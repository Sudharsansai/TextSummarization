# src: https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/config.py
""" Implementation of all available options """
from __future__ import print_function

"""Model architecture/optimization options for DrQA document reader."""

import argparse
import logging

logger = logging.getLogger(__name__)

# Index of arguments concerning the core model architecture
MODEL_ARCHITECTURE = {
    'model_type', 'reader_type', 'emsize', 'rnn_type',
    'nhid', 'nlayers', 'bidirection'
}

BIDAF_MODEL_ARCHITECTURE = {
    'n_characters', 'char_emsize', 'filter_size', 'nfilters'
}

IBIDAF_MODEL_ARCHITECTURE = {'fc_dim'}

SEQ2SEQ_ARCHITECTURE = {'attn_type', 'coverage_attn', 'copy_attn',
                        'reuse_copy_attn', 'context_gate', 'max_ans_len',
                        'share_decoder_embeddings'}

ADVANCED_OPTIONS = {'use_elmo', 'optfile', 'wgtfile'}

# Index of arguments concerning the model optimizer/training
MODEL_OPTIMIZER = {
    'fix_embeddings', 'optimizer', 'learning_rate', 'momentum', 'weight_decay',
    'rnn_padding', 'dropout_rnn', 'dropout', 'dropout_emb',
    'max_len', 'grad_clipping', 'tune_partial', 'lr_decay', 'ema'
}


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_model_args(parser):
    parser.register('type', 'bool', str2bool)

    # Model architecture
    model = parser.add_argument_group('Neural QA Reader Architecture')
    model.add_argument('--model_type', type=str, default='rnn',
                       help='Model architecture type')
    model.add_argument('--reader_type', type=str, default='bidaf',
                       help='Model name: bidaf, ibidaf, mlstm, areader')
    model.add_argument('--emsize', type=int, default=300,
                       help='Embedding size if embedding_file is not given')
    model.add_argument('--rnn_type', type=str, default='LSTM',
                       help='RNN type: LSTM, GRU')
    model.add_argument('--nhid', type=int, default=200,
                       help='Hidden size of RNN units')
    model.add_argument('--bidirection', type='bool', default=True,
                       help='use bidirectional recurrent unit')
    model.add_argument('--nlayers', type=int, default=1,
                       help='Number of encoding layers')

    # Model specific details
    bidaf = parser.add_argument_group('BIDAF Specific Model Params')
    bidaf.add_argument('--n_characters', type=int, default=260,
                       help='Character vocabulary size')
    bidaf.add_argument('--char_emsize', type=int, default=16,
                       help='Character embedding size')
    bidaf.add_argument('--filter_size', nargs='+', type=list,
                       default=[5], action='append',
                       help='Char convolution filter sizes')
    bidaf.add_argument('--nfilters', nargs='+', type=list,
                       default=[100], action='append',
                       help='Number of char convolution filters')

    ibidaf = parser.add_argument_group('iBIDAF Specific Model Params')
    ibidaf.add_argument('--fc_dim', type=int, default=200,
                        help='Number of hidden units per ReLU linear layer '
                             '(only applicable for ibidaf)')

    seq2seq = parser.add_argument_group('Seq2seq Model Specific Params')
    seq2seq.add_argument('--attn_type', type=str, default='general',
                         help='Attention type for the seq2seq [dot, general, mlp]')
    seq2seq.add_argument('--coverage_attn', type='bool', default=False,
                         help='Use coverage attention')
    seq2seq.add_argument('--copy_attn', type='bool', default=False,
                         help='Use copy attention')
    seq2seq.add_argument('--reuse_copy_attn', type='bool', default=False,
                         help='Reuse encoder attention')
    seq2seq.add_argument('--context_gate', type=str, default=None,
                         choices=[None, 'source', 'target', 'both'],
                         help='Use context gate')
    seq2seq.add_argument('--max_ans_len', type=int, default=50,
                         help='Maximum allowed length of an answer')
    seq2seq.add_argument('--share_decoder_embeddings', type='bool', default=False,
                         help='Share decoder embeddings weight with softmax layer')

    advanced = parser.add_argument_group('Advanced Optional Params')
    advanced.add_argument('--use_elmo', type='bool', default=False,
                          help='Use elmo as the input layer')
    advanced.add_argument('--optfile', type=str, default='',
                          help='Required if ELMo is used')
    advanced.add_argument('--wgtfile', type=str, default='',
                          help='Required if ELMo is used')

    # Optimization details
    optim = parser.add_argument_group('Neural QA Reader Optimization')
    optim.add_argument('--dropout_emb', type=float, default=0.2,
                       help='Dropout rate for word embeddings')
    optim.add_argument('--dropout_rnn', type=float, default=0.2,
                       help='Dropout rate for RNN states')
    optim.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout for NN layers')
    optim.add_argument('--optimizer', type=str, default='adamax',
                       help='Optimizer: sgd or adamax')
    optim.add_argument('--learning_rate', type=float, default=0.1,
                       help='Learning rate for SGD only')
    parser.add_argument('--lr_decay', type=float, default=0.95,
                        help='Decay ratio for learning rate')
    parser.add_argument('--ema', type='bool', default=False,
                        help='Maintain moving averages of the trained parameters')
    optim.add_argument('--grad_clipping', type=float, default=10,
                       help='Gradient clipping')
    parser.add_argument('--early_stop', type=int, default=5,
                        help='Stop training if performance doesn\'t improve')
    optim.add_argument('--weight_decay', type=float, default=0,
                       help='Weight decay factor')
    optim.add_argument('--momentum', type=float, default=0,
                       help='Momentum factor')
    optim.add_argument('--fix_embeddings', type='bool', default=True,
                       help='Keep word embeddings fixed (use pretrained)')
    optim.add_argument('--tune_partial', type=int, default=0,
                       help='Backprop through only the top N question words')
    optim.add_argument('--max_len', type=int, default=15,
                       help='The max span allowed during decoding')


def get_model_args(args, _type):
    """Filter args for model ones.
    From a args Namespace, return a new Namespace with *only* the args specific
    to the model architecture or optimization. (i.e. the ones defined here.)
    """
    global MODEL_ARCHITECTURE, MODEL_OPTIMIZER, BIDAF_MODEL_ARCHITECTURE, \
        IBIDAF_MODEL_ARCHITECTURE, ADVANCED_OPTIONS, SEQ2SEQ

    required_args = MODEL_ARCHITECTURE | MODEL_OPTIMIZER | ADVANCED_OPTIONS
    if args.reader_type == 'bidaf' or args.reader_type=='summarizer':
        required_args = required_args | BIDAF_MODEL_ARCHITECTURE
    elif args.reader_type == 'ibidaf':
        required_args = required_args | BIDAF_MODEL_ARCHITECTURE | IBIDAF_MODEL_ARCHITECTURE


    if _type == 'abstractor':
        required_args = required_args | SEQ2SEQ_ARCHITECTURE

    arg_values = {k: v for k, v in vars(args).items() if k in required_args}
    return argparse.Namespace(**arg_values)


def override_model_args(old_args, new_args):
    """Set args to new parameters.
    Decide which model args to keep and which to override when resolving a set
    of saved args and new args.
    We keep the new optimation, but leave the model architecture alone.
    """
    global MODEL_OPTIMIZER
    old_args, new_args = vars(old_args), vars(new_args)
    for k in old_args.keys():
        if k in new_args and old_args[k] != new_args[k]:
            if k in MODEL_OPTIMIZER:
                logger.info('Overriding saved %s: %s --> %s' %
                            (k, old_args[k], new_args[k]))
                old_args[k] = new_args[k]
            else:
                logger.info('Keeping saved %s: %s' % (k, old_args[k]))
    return argparse.Namespace(**old_args)


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--no_cuda', type='bool', default=False,
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--gpu', type=int, default=-1,
                         help='Run on a specific GPU')
    runtime.add_argument('--data_workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--parallel', type='bool', default=False,
                         help='Use DataParallel on all available GPUs')
    runtime.add_argument('--random_seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num_epochs', type=int, default=40,
                         help='Train data iterations')
    runtime.add_argument('--batch_size', type=int, default=32,
                         help='Batch size for training')
    runtime.add_argument('--test_batch_size', type=int, default=128,
                         help='Batch size during validation/testing')

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--dataset_name', type=str, choices=['SQuAD', 'MSMARCO', 'CNN-DM'],
                       default='SQuAD', help='Name of the experimental dataset')
    files.add_argument('--model_dir', type=str, default='/tmp/qa_models/',
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model_name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data_dir', type=str, default='/data/SQuAD/',
                       help='Directory of training/validation data')
    files.add_argument('--train_file', type=str,
                       default='train-v2.0-processed.txt',
                       help='Preprocessed train file')
    files.add_argument('--dev_file', type=str,
                       default='dev-v2.0-processed.txt',
                       help='Preprocessed dev file')
    files.add_argument('--dev_json', type=str, default='dev-v2.0.json',
                       help=('Unprocessed dev file to run validation '
                             'while training on'))
    files.add_argument('--embed_dir', type=str, default='/data/glove/',
                       help='Directory of pre-trained embedding files')
    files.add_argument('--embedding_file', type=str, default='',
                       help='Space-separated pretrained embeddings file')

    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default='',
                           help='Path to a pretrained model to warm-start with')

    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--max_examples', type=int, default=-1,
                            help='Maximum number of examples for training')
    preprocess.add_argument('--uncased_question', type='bool', default=False,
                            help='Question words will be lower-cased')
    preprocess.add_argument('--uncased_doc', type='bool', default=False,
                            help='Document words will be lower-cased')
    preprocess.add_argument('--uncased_answer', type='bool', default=False,
                            help='Answer words will be lower-cased')
    preprocess.add_argument('--restrict_vocab', type='bool', default=True,
                            help='Only use pre-trained words in embedding_file')
    preprocess.add_argument('--src_vocab_size', type=int, default=None,
                            help='Maximum allowed length for src dictionary')
    preprocess.add_argument('--tgt_vocab_size', type=int, default=None,
                            help='Maximum allowed length for tgt dictionary')
    preprocess.add_argument('--skip_no_answer', type='bool', default=False,
                            help='Skip unanswerable questions')
    preprocess.add_argument('--max_characters_per_token', type=int, default=30,
                            help='Maximum number of characters allowed per token')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--valid_metric', type=str, default='bleu',
                         help='The evaluation metric used for model selection')
    general.add_argument('--display_iter', type=int, default=25,
                         help='Log state after every <display_iter> batches')
    general.add_argument('--sort_by_len', type='bool', default=True,
                         help='Sort batches by length for speed')
    general.add_argument('--only_test', type='bool', default=False,
                         help='Only do testing')

