# src: https://github.com/facebookresearch/DrQA/blob/master/scripts/reader/train.py

import sys

sys.path.append(".")
sys.path.append("..")

import os
import json
import torch
import logging
import subprocess
import argparse
import numpy as np

import config
from nqa.inputters.utils import AverageMeter, Timer
import vector
import data

from abstractor import Abstractor
import utils as util
from nqa.eval.bleu import compute_bleu
from nqa.eval.rouge import Rouge
from nqa.eval import f1_score, exact_match_score

def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    if(args.reader_type!='summarizer'):
	    args.dev_json = os.path.join(args.data_dir, args.dev_json)
	    if not os.path.isfile(args.dev_json):
	        raise IOError('No such file: %s' % args.dev_json)
    args.train_file = os.path.join(args.data_dir, args.train_file)
    if not os.path.isfile(args.train_file):
        raise IOError('No such file: %s' % args.train_file)
    args.dev_file = os.path.join(args.data_dir, args.dev_file)
    if not os.path.isfile(args.dev_file):
        raise IOError('No such file: %s' % args.dev_file)
    if args.embedding_file:
        args.embedding_file = os.path.join(args.embed_dir, args.embedding_file)
        if not os.path.isfile(args.embedding_file):
            raise IOError('No such file: %s' % args.embedding_file)

    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    args.log_file = os.path.join(args.model_dir, args.model_name + '.txt')
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')
    args.pred_file = os.path.join(args.model_dir, args.model_name + '_predictions.txt')

    # Embeddings options
    if args.embedding_file:
        with open(args.embedding_file) as f:
            dim = len(f.readline().strip().split(' ')) - 1
        args.emsize = dim
    elif not args.emsize:
        raise RuntimeError('Either embedding_file or embedding_dim '
                           'needs to be specified.')

    # Make sure tune_partial and fix_embeddings are consistent.
    if args.tune_partial > 0:
        logger.warning('WARN: partial tuning is not supported in abstractive model.')

    # Make sure fix_embeddings and embedding_file are consistent
    if args.fix_embeddings:
        if not (args.embedding_file or args.pretrained):
            logger.warning('WARN: fix_embeddings set to False '
                           'as embeddings are random.')
            args.fix_embeddings = False
    return args



def train(args, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    train_loss = AverageMeter()
    epoch_time = Timer()
    model.optimizer.param_groups[0]['lr'] = model.optimizer.param_groups[0]['lr'] * args.lr_decay

    # Run one epoch
    for idx, ex in enumerate(data_loader):
        train_loss.update(*model.update(ex))

        if idx % args.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d | ' %
                        (global_stats['epoch'], idx, len(data_loader)) +
                        'loss = %.2f | elapsed time = %.2f (s)' %
                        (train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()

    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                (global_stats['epoch'], epoch_time.time()))

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint',
                         global_stats['epoch'] + 1)

def init_from_scratch(args, train_exs, dev_exs):
	"""New model, new data, new dictionary."""
	# Build a dictionary from the data questions + words (train/dev splits)
	logger.info('-' * 100)
	logger.info('Build word dictionary')
	if(args.reader_type=='summarizer'):
		src_dict = util.build_word_and_char_dict(args, train_exs + dev_exs,
											 fields=['summary', 'document'],
											 dict_size=args.src_vocab_size)
		tgt_dict = util.build_word_and_char_dict(args, train_exs + dev_exs,
												 fields=['summary', 'document'],
												 dict_size=args.tgt_vocab_size)
		logger.info('Num words in source = %d and target = %d' % (len(src_dict),
																  len(tgt_dict)))
	else:
		src_dict = util.build_word_and_char_dict(args, train_exs + dev_exs,
											 fields=['question', 'document'],
											 dict_size=args.src_vocab_size)
		tgt_dict = util.build_word_and_char_dict(args, train_exs + dev_exs,
												 fields=['question', 'document'],
												 dict_size=args.tgt_vocab_size)
		logger.info('Num words in source = %d and target = %d' % (len(src_dict),
																  len(tgt_dict)))
	

	# Initialize model
	model = Abstractor(config.get_model_args(args, 'abstractor'), src_dict, tgt_dict)

	# Load pretrained embeddings for words in dictionary
	if args.embedding_file:
		model.load_src_embeddings(src_dict.tokens(), args.embedding_file)
		model.load_tgt_embeddings(tgt_dict.tokens(), args.embedding_file)

	return model

def eval_accuracies(prediction, target, fw=None):
    """An unofficial evalutation helper.
    Compute ROUGE and BLEU score from list of predictions and targets.
    """
    # Compute accuracies from targets
    tgt = [[t] for t in target]
    score, precisions, _, _, _, _ = compute_bleu(tgt, prediction)
    rouge_calculator = Rouge()
    rouge_l, _ = rouge_calculator.compute_score(tgt, prediction)

    f1 = AverageMeter()
    exact_match = AverageMeter()
    for i in range(len(prediction)):
        pred, tgt = ' '.join(prediction[i]), ' '.join(target[i])
        exact_match.update(exact_match_score(pred, tgt))
        f1.update(f1_score(pred, tgt))
        if fw:
            fw.write(pred + ' |||| ' + tgt + '\n')

    return rouge_l * 100, precisions[0] * 100, exact_match.avg * 100, f1.avg * 100


def validate_official_QA(args, data_loader, model, global_stats, answers):
	"""Run one full official validation. Uses exact spans and same
	exact match/F1 score computation as in the SQuAD script.
	Extra arguments:
		offsets: The character start/end indices for the tokens in each context.
		texts: Map of qid --> raw text of examples context (matches offsets).
		answers: Map of qid --> list of accepted answers.
	"""
	eval_time = Timer()
	rouge = AverageMeter()
	bleu = AverageMeter()
	f1 = AverageMeter()
	exact_match = AverageMeter()

	# Run through examples
	examples = 0
	fw = open('temp_log.txt', 'w', encoding="utf-8")
	for ex in data_loader:
		ids, batch_size = ex['ids'], ex['doc_rep'].size(0)
		# FIXME: why do evaluation for one answer span?
		targets = [answers[eid][0] for eid in ids]
		predictions, _ = model.predict(ex)

		accuracies = eval_accuracies(predictions, targets, fw)
		rouge.update(accuracies[0], batch_size)
		bleu.update(accuracies[1], batch_size)
		exact_match.update(accuracies[2], batch_size)
		f1.update(accuracies[3], batch_size)

		examples += batch_size

	fw.close()
	logger.info('dev valid official: Epoch = %d | rouge_l = %.2f | ' %
				(global_stats['epoch'], rouge.avg) +
				'bleu_1 = %.2f | EM = %.2f | F1 = %.2f | examples = %d | ' %
				(bleu.avg, exact_match.avg, f1.avg, examples) +
				'valid time = %.2f (s)' % eval_time.time())

	return {'rouge': rouge.avg, 'bleu': bleu.avg,
			'exact_match': exact_match.avg, 'f1': f1.avg}


def validate_official_summarization(args, data_loader, model, global_stats, mode):
    """Run one full official validation for summarization
    Unofficial = doesn't use SQuAD script.
    """
    eval_time = Timer()
    rouge = AverageMeter()
    bleu = AverageMeter()
    f1 = AverageMeter()
    exact_match = AverageMeter()

    # Make predictions
    examples = 0
    for ex in data_loader:
        batch_size = ex['doc_rep'].size(0)
        predictions, targets = model.predict(ex)

        accuracies = eval_accuracies(predictions, targets)
        rouge.update(accuracies[0], batch_size)
        bleu.update(accuracies[1], batch_size)
        exact_match.update(accuracies[2], batch_size)
        f1.update(accuracies[3], batch_size)

        # If getting train accuracies, sample max 10k
        examples += batch_size
        if mode == 'train' and examples >= 1e4:
            break

    logger.info('%s valid unofficial: Epoch = %d | rouge_l = %.2f | ' %
                (mode, global_stats['epoch'], rouge.avg) +
                'bleu_1 = %.2f | EM = %.2f | F1 = %.2f | examples = %d | ' %
                (bleu.avg, exact_match.avg, f1.avg, examples) +
                'valid time = %.2f (s)' % eval_time.time())

    return {'rouge': rouge.avg, 'bleu': bleu.avg,
            'exact_match': exact_match.avg, 'f1': f1.avg}


if __name__ == '__main__':

	print("In the program...")
	logger = logging.getLogger()

	parser = argparse.ArgumentParser(
		'Neural QA Reader',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	config.add_train_args(parser)
	config.add_model_args(parser)
	args = parser.parse_args()
	set_defaults(args)

	print("Parser parsed..")

	# Set cuda
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	#if args.cuda:
	#	torch.cuda.set_device(args.gpu)

	# Set random state
	np.random.seed(args.random_seed)
	torch.manual_seed(args.random_seed)
	if args.cuda:
		torch.cuda.manual_seed(args.random_seed)

	# Set logging
	logger.setLevel(logging.INFO)
	fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
							'%m/%d/%Y %I:%M:%S %p')
	console = logging.StreamHandler()
	console.setFormatter(fmt)
	logger.addHandler(console)
	if args.log_file:
		if args.checkpoint:
			logfile = logging.FileHandler(args.log_file, 'a')
		else:
			logfile = logging.FileHandler(args.log_file, 'w')
		logfile.setFormatter(fmt)
		logger.addHandler(logfile)
	logger.info('COMMAND: %s' % ' '.join(sys.argv))

	# Start->Run!
	logger.info('-' * 100)
	logger.info('Load data files')

	if(args.dataset_name=="MSMARCO" or args.dataset_name=="SQuAD"):
		train_exs = util.load_QA_data(args, args.train_file, skip_no_answer=args.skip_no_answer,
								   max_examples=args.max_examples,
								   dataset_name=args.dataset_name)
		logger.info('Num train examples = %d' % len(train_exs))
		dev_exs = util.load_QA_data(args, args.dev_file,
								 skip_no_answer=args.skip_no_answer,
								 dataset_name=args.dataset_name)
		logger.info('Num dev examples = %d' % len(dev_exs))
		dev_answers = util.load_answers(args.dev_json, dataset_name=args.dataset_name)
	else:
		train_exs = util.load_summ_data(args, args.train_file, args.max_examples)
		logger.info('Num train examples = %d' % len(train_exs))
		dev_exs = util.load_summ_data(args, args.dev_file, args.max_examples)
		logger.info('Num dev examples = %d' % len(dev_exs))

	logger.info('-' * 100)
	start_epoch = 0
	if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
		# Just resume training, no modifications.
		logger.info('Found a checkpoint...')
		checkpoint_file = args.model_file + '.checkpoint'
		model, start_epoch = Abstractor.load_checkpoint(checkpoint_file, args.cuda)
	else:
		# Training starts fresh. But the model state is either pretrained or
		# newly (randomly) initialized.
		if args.pretrained:
			logger.info('Using pretrained model...')
			model = Abstractor.load(args.pretrained, args)
		else:
			logger.info('Training model from scratch...')
			model = init_from_scratch(args, train_exs, dev_exs)

		# Set up optimizer
		model.init_optimizer()

	# Use the GPU?
	if args.cuda:
		model.cuda()

	# Use multiple GPUs?
	if args.parallel:
		model.parallelize()


	# --------------------------------------------------------------------------
	# DATA ITERATORS
	# Two datasets: train and dev. If we sort by length it's faster.

	logger.info('-' * 100)
	logger.info('Make data loaders')

	if(args.reader_type=='summarizer'):
		train_dataset = data.ReaderDatasetSummarization(train_exs,
										   model)
		if args.sort_by_len:
			train_sampler = data.SortedBatchSamplerSummarization(train_dataset.lengths(),
													args.batch_size,
													shuffle=True)
		else:
			train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
	else:
		train_dataset = data.ReaderDatasetQA(train_exs,
										   model,
										   single_answer=True)
		if args.sort_by_len:
			train_sampler = data.SortedBatchSamplerQA(train_dataset.lengths(),
													args.batch_size,
													shuffle=True)
		else:
			train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
	

	if(args.reader_type=='summarizer'):
		train_loader = torch.utils.data.DataLoader(
			train_dataset,
			batch_size=args.batch_size,
			sampler=train_sampler,
			num_workers=args.data_workers,
			collate_fn=vector.batchify_summary,
			pin_memory=args.cuda,
		)
	else:
		train_loader = torch.utils.data.DataLoader(
			train_dataset,
			batch_size=args.batch_size,
			sampler=train_sampler,
			num_workers=args.data_workers,
			collate_fn=vector.batchify_QA,
			pin_memory=args.cuda,
		)


	## DEV DATASET

	if(args.reader_type=='summarizer'):
		dev_dataset = data.ReaderDatasetSummarization(dev_exs,
										   model)
		if args.sort_by_len:
			dev_sampler = data.SortedBatchSamplerSummarization(dev_dataset.lengths(),
													args.batch_size,
													shuffle=True)
		else:
			dev_sampler = torch.utils.data.sampler.RandomSampler(dev_dataset)
	else:
		dev_dataset = data.ReaderDatasetQA(dev_exs,
										   model,
										   single_answer=True)
		if args.sort_by_len:
			dev_sampler = data.SortedBatchSamplerQA(dev_dataset.lengths(),
													args.batch_size,
													shuffle=True)
		else:
			dev_sampler = torch.utils.data.sampler.RandomSampler(dev_dataset)
	

	if(args.reader_type=='summarizer'):
		dev_loader = torch.utils.data.DataLoader(
			dev_dataset,
			batch_size=args.batch_size,
			sampler=dev_sampler,
			num_workers=args.data_workers,
			collate_fn=vector.batchify_summary,
			pin_memory=args.cuda,
		)
	else:
		dev_loader = torch.utils.data.DataLoader(
			dev_dataset,
			batch_size=args.batch_size,
			sampler=dev_sampler,
			num_workers=args.data_workers,
			collate_fn=vector.batchify_QA,
			pin_memory=args.cuda,
		)

	# -------------------------------------------------------------------------
	# PRINT CONFIG
	logger.info('-' * 100)
	logger.info('CONFIG:\n%s' %
				json.dumps(vars(args), indent=4, sort_keys=True))

	# --------------------------------------------------------------------------
	# DO TEST

	if args.only_test:
		stats = {'timer': Timer(), 'epoch': 0, 'best_valid': 0, 'no_improvement': 0}
		if(args.reader_type=='summarizer'):
			result = validate_official_summarization(args, dev_loader, model, stats, mode="train")
		else:
			result = validate_official_QA(args, dev_loader, model, stats, dev_answers)

	logger.info('-' * 100)
	logger.info('Starting training...')
	stats = {'timer': Timer(), 'epoch': 0, 'best_valid': 0, 'no_improvement': 0}
	for epoch in range(start_epoch, args.num_epochs):
		stats['epoch'] = epoch

		# Train
		train(args, train_loader, model, stats)

		# Validate unofficial (train)
		# validate_unofficial(args, train_loader, model, stats, mode='train')

		# Validate unofficial (dev)
		# validate_unofficial(args, dev_loader, model, stats, mode='dev')

		if(args.reader_type=='summarizer'):
			result = validate_official_summarization(args, dev_loader, model, stats, mode="train")
		else:
			result = validate_official_QA(args, dev_loader, model, stats, dev_answers)

		# Save best valid
		if result[args.valid_metric] > stats['best_valid']:
			logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
						(args.valid_metric, result[args.valid_metric],
						 stats['epoch'], model.updates))
			model.save(args.model_file)
			stats['best_valid'] = result[args.valid_metric]
			stats['no_improvement'] = 0
		else:
			stats['no_improvement'] += 1
			if stats['no_improvement'] >= args.early_stop:
				break

