# src: https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/model.py
"""DrQA Document Reader model"""

import copy
import logging

import torch
import torch.optim as optim
from torch.autograd import Variable

from config import override_model_args
from nqa.inputters import BOS, EOS
from writer import Writer
from nqa.modules.ema import ExponentialMovingAverage

logger = logging.getLogger(__name__)


class Abstractor(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, src_dict, tgt_dict, state_dict=None):
        # Book-keeping.
        self.args = args
        self.word_dict = src_dict
        self.args.vocab_size = len(src_dict)
        self.tgt_dict = tgt_dict
        self.args.tgt_vocab_size = len(tgt_dict)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False

        if args.model_type == 'rnn':
            self.network = Writer(args, tgt_dict)
        else:
            raise RuntimeError('Unsupported model: %s' % args.model_type)

        # Load saved state
        if state_dict:
            # Load buffer separately
            if 'fixed_embedding' in state_dict:
                fixed_embedding = state_dict.pop('fixed_embedding')
                self.network.load_state_dict(state_dict)
                self.network.register_buffer('fixed_embedding', fixed_embedding)
            else:
                self.network.load_state_dict(state_dict)

        if self.args.ema:
            self.ema = ExponentialMovingAverage(0.999)
            for name, param in self.network.named_parameters():
                if param.requires_grad:
                    self.ema.register(name, param.data)

    def expand_dictionary(self, words):
        # TODO: at present, only supported for source dictionary
        """Add words to the Reader dictionary if they do not exist. The
        underlying embedding matrix is also expanded (with random embeddings).
        Args:
            words: iterable of tokens to add to the dictionary.
        Output:
            added: set of tokens that were added.
        """
        to_add = {self.word_dict.normalize(w) for w in words
                  if w not in self.word_dict}

        # Add words to dictionary and expand embedding layer
        if len(to_add) > 0:
            logger.info('Adding %d new words to dictionary...' % len(to_add))
            for w in to_add:
                self.word_dict.add(w)
            self.args.vocab_size = len(self.word_dict)
            logger.info('New vocab size: %d' % len(self.word_dict))

            old_embedding = self.network.encoder.word_embeddings.embedding.weight.data
            self.network.encoder.word_embeddings.embedding = torch.nn.Embedding(self.args.vocab_size,
                                                                                self.args.emsize,
                                                                                padding_idx=0)
            new_embedding = self.network.encoder.word_embeddings.embedding.weight.data
            new_embedding[:old_embedding.size(0)] = old_embedding

        # Return added words
        return to_add

    @staticmethod
    def load_embeddings(word_dict, words, embedding_file,
                        emb_layer, fix_embeddings):
        """Load pretrained embeddings for a given list of words, if they exist.
        #TODO: update args
        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in words if w in word_dict}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts, embedding = {}, {}
        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert (len(parsed) == emb_layer.word_vec_size + 1)
                w = word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[w] = vec
                    else:
                        logging.warning(
                            'WARN: Duplicate embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[w].add_(vec)

        for w, c in vec_counts.items():
            embedding[w].div_(c)

        emb_layer.init_word_vectors(word_dict, embedding, fix_embeddings)
        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def load_src_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.
        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        self.load_embeddings(self.word_dict, words, embedding_file,
                             self.network.encoder.word_embeddings.embedding,
                             self.args.fix_embeddings)

    def load_tgt_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.
        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        fix_embeddings = not self.args.share_decoder_embeddings and self.args.fix_embeddings
        self.load_embeddings(self.tgt_dict, words, embedding_file,
                             self.network.word_embeddings.embedding,
                             fix_embeddings)

    def init_optimizer(self, state_dict=None, use_gpu=True):
        """Initialize an optimizer for the free parameters of the network.
        Args:
            state_dict: optimizer's state dict
            use_gpu: required to move state_dict to GPU
        """
        if self.args.fix_embeddings:
            for p in self.network.encoder.word_embeddings.embedding.parameters():
                p.requires_grad = False

        if not self.args.share_decoder_embeddings and self.args.fix_embeddings:
            for p in self.network.word_embeddings.embedding.parameters():
                p.requires_grad = False

        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.args.optimizer)

        if state_dict is not None:
            self.optimizer.load_state_dict(state_dict)
            # TODO: fix me [temporary: https://github.com/pytorch/pytorch/issues/2830]
            if use_gpu:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def _make_src_map(self, data):
        """ ? """
        src_size = max([t.size(0) for t in data])
        src_vocab_size = max([t.max() for t in data]) + 1
        alignment = torch.zeros(len(data), src_size, src_vocab_size)
        for i, sent in enumerate(data):
            for j, t in enumerate(sent):
                alignment[i, j, t] = 1
        return alignment

    def _align(self, data):
        """ ? """
        tgt_size = max([t.size(0) for t in data])
        alignment = torch.zeros(len(data), tgt_size).long()
        for i, sent in enumerate(data):
            alignment[i, :sent.size(0)] = sent
        return alignment

    def update(self, ex):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        # To enable copy attn, collect source map and alignment info
        if self.args.copy_attn:
            assert 'src_map' in ex and 'alignment' in ex
            source_map = self._make_src_map(ex['src_map'])
            alignment = self._align(ex['alignment'])
            source_map = Variable(source_map.cuda(async=True)) if self.use_cuda \
                else Variable(source_map)
            alignment = Variable(alignment.cuda(async=True)) if self.use_cuda \
                else Variable(alignment)
        else:
            source_map, alignment = None, None

        inputs = []
        params = self.network.get_forward_params()
        for name in params[:-2]:
            #print(params[:-2])
            assert name in ex
            # Transfer to GPU
            network_param = ex[name] if ex[name] is None \
                else Variable(ex[name].cuda(async=True)) if self.use_cuda \
                else Variable(ex[name])
            inputs.append(network_param)

        # Run forward
        if(self.args.reader_type=="summarizer"):
            loss = self.network(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], source_map, alignment)
        else:
            loss = self.network(inputs[0], inputs[1], inputs[2], inputs[6], inputs[7], \
                source_map, alignment, inputs[3], inputs[4], inputs[5])
        # Run forward
        #loss = self.network(*inputs, source_map, alignment)

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.args.grad_clipping)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Maintain exponential moving average
        if self.args.ema:
            # maintain exponential moving average
            for name, param in self.network.named_parameters():
                if param.requires_grad:
                    param.data = self.ema(name, param.data)

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

        return loss.data[0], ex['doc_rep'].size(0)

    def reset_parameters(self):
        """Reset any partially fixed parameters to original states."""

        # Reset fixed embeddings to original value
        if self.args.tune_partial > 0:
            if self.parallel:
                embedding = self.network.encoder.word_embeddings.embedding.weight.data
                fixed_embedding = self.network.fixed_embedding
            else:
                embedding = self.network.encoder.word_embeddings.embedding.weight.data
                fixed_embedding = self.network.fixed_embedding

            # Embeddings to fix are the last indices
            offset = embedding.size(0) - fixed_embedding.size(0)
            if offset >= 0:
                embedding[offset:] = fixed_embedding

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex, top_n=1):
        """Forward a batch of examples only to get predictions.
        Args:
            ex: the batch
            top_n: Number of predictions to return per batch element.
        Output:
            predictions: batch * top_n predicted sequences
        """

        def tens2sen(t, word_dict, src_vocabs=None):
            sentences = []
            # loop over the batch elements
            for idx, s in enumerate(t):
                sentence = []
                for wt in s:
                    word = wt if isinstance(wt, int) \
                        else wt.data[0]
                    if word in [BOS]:
                        continue
                    if word in [EOS]:
                        break
                    if word < len(word_dict):
                        sentence += [word_dict[word]]
                    elif src_vocabs:
                        word = word - len(word_dict)
                        sentence += [src_vocabs[idx][1][word]]

                # Sentence can't be empty list
                assert len(sentence) > 0
                sentences += [sentence]
            return sentences

        # Eval mode
        self.network.eval()

        # To enable copy attn, collect source map and alignment info
        if self.args.copy_attn:
            assert 'src_map' in ex and 'alignment' in ex
            source_map = self._make_src_map(ex['src_map'])
            alignment = self._align(ex['alignment'])
            source_map = Variable(source_map.cuda(async=True), volatile=True) \
                if self.use_cuda else Variable(source_map, volatile=True)
            alignment = Variable(alignment.cuda(async=True), volatile=True) \
                if self.use_cuda else Variable(alignment, volatile=True)
        else:
            source_map, alignment = None, None

        inputs = []
        params = self.network.get_decoding_params()
        for name in params[:-2]:
            assert name in ex
            # Transfer to GPU
            network_param = ex[name] if ex[name] is None \
                else Variable(ex[name].cuda(async=True), volatile=True) if self.use_cuda \
                else Variable(ex[name], volatile=True)
            inputs.append(network_param)

        # Run forward
        if(self.args.reader_type=="summarizer"):
            predictions = self.network.decode(inputs[0], inputs[1], inputs[2], source_map, alignment,
                                              max_len=self.args.max_len,
                                              tgt_dict=self.tgt_dict)
        else:
            predictions = self.network.decode(inputs[0], inputs[1], inputs[2], source_map, alignment,\
                                              self.args.max_len, self.tgt_dict, inputs[3], inputs[4], inputs[5])
        # FIXME: Copied tokens are not considered
        predictions = tens2sen(predictions, self.tgt_dict, ex['source_vocabs'])
        if(self.args.reader_type=="summarizer"):
            targets = tens2sen(ex['summ_rep'], self.tgt_dict)
        else:
            targets = tens2sen(ex['ans_rep'], self.tgt_dict)

        return predictions, targets

    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        if self.parallel:
            network = self.network
        else:
            network = self.network
        state_dict = copy.copy(network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'word_dict': self.word_dict,
            'tgt_dict': self.tgt_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        if self.parallel:
            network = self.network
        else:
            network = self.network
        params = {
            'state_dict': network.state_dict(),
            'word_dict': self.word_dict,
            'tgt_dict': self.tgt_dict,
            'args': self.args,
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, new_args=None):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        tgt_dict = saved_params['tgt_dict']
        state_dict = saved_params['state_dict']
        args = saved_params['args']
        if new_args:
            args = override_model_args(args, new_args)
        return Abstractor(args, word_dict, tgt_dict, state_dict)

    @staticmethod
    def load_checkpoint(filename, use_gpu=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        tgt_dict = saved_params['tgt_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        optimizer = saved_params['optimizer']
        args = saved_params['args']
        model = Abstractor(args, word_dict, tgt_dict, state_dict)
        model.init_optimizer(optimizer, use_gpu)
        return model, epoch

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network, device_ids=range(2))
        self.network = self.network.module
