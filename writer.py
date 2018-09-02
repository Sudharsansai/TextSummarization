import torch
import torch.nn as nn
import torch.nn.functional as f
from collections import OrderedDict
from torch.autograd import Variable

from nqa.inputters import BOS, PAD, UNK
from nqa.models.reader.areader import AttentiveReader
from bidaf import BIDAF
from nqa.models.reader.ibidaf import ImprovedBIDAF
from nqa.models.reader.mlstm import mLSTM
from nqa.modules.embeddings import Embeddings
from nqa.decoders import RNNDecoder
from nqa.utils import sequence_mask


class Writer(nn.Module):
    """Module that writes an answer for the question given a passage."""

    def __init__(self, args, tgt_dict):
        """"Constructor of the class."""
        super(Writer, self).__init__()

        self.word_embeddings = nn.Sequential(OrderedDict([
            ('embedding', Embeddings(args.emsize,
                                     args.tgt_vocab_size,
                                     PAD)),
            ('dropout', nn.Dropout(p=args.dropout_emb))
        ]))

        self.reader_type = args.reader_type
        if self.reader_type == 'bidaf' or self.reader_type == 'ibidaf':
            self.encoder = BIDAF(args) if self.reader_type == 'bidaf' else ImprovedBIDAF(args)
            input_size = args.nhid * 5 if self.reader_type == 'bidaf' else args.nhid + args.fc_dim
            self.transform = nn.Linear(input_size, args.nhid)
        elif self.reader_type == 'mlstm':
            self.encoder = mLSTM(args)
        elif self.reader_type == 'areader':
            self.encoder = AttentiveReader(args)
        elif self.reader_type == 'summarizer':
            self.encoder = BIDAF(args)
            input_size = args.nhid * args.nlayers
            self.transform = nn.Linear(input_size, args.nhid)
        else:
            raise RuntimeError('Unsupported reader: %s' % args.reader_type)

        self.decoder = RNNDecoder(args.rnn_type,
                                  args.emsize,
                                  args.bidirection,
                                  1,
                                  args.nhid,
                                  attn_type=args.attn_type,
                                  coverage_attn=args.coverage_attn,
                                  copy_attn=args.copy_attn,
                                  reuse_copy_attn=args.reuse_copy_attn,
                                  context_gate=args.context_gate,
                                  dropout=args.dropout_rnn)

        self.dropout = nn.Dropout(args.dropout)
        self.generator = nn.Sequential(
            nn.Linear(args.nhid, args.emsize),
            nn.Linear(args.emsize, args.tgt_vocab_size)
        )
        self.softmax = nn.LogSoftmax(dim=-1)

        self.copy_attn = args.copy_attn
        if self.copy_attn:
            from nqa.modules.copy_generator import CopyGenerator, CopyGeneratorCriterion
            self.copy_generator = CopyGenerator(args.nhid, tgt_dict, self.generator)
            self.criterion = CopyGeneratorCriterion(vocab_size=len(tgt_dict),
                                                    force_copy=False)

        if args.share_decoder_embeddings:
            self.generator[1].weight = self.word_embeddings.embedding.word_lut.weight

    def get_forward_params(self):
        if(self.reader_type=="summarizer"):
            return ['doc_rep', 'doc_char_rep', 'doc_len',
                    'summ_rep', 'summ_len', 'src_map', 'alignment']
        else:
            return ['doc_rep', 'doc_char_rep', 'doc_len',
                    'que_rep', 'que_char_rep', 'que_len',
                    'ans_rep', 'ans_len', 'src_map', 'alignment']

    def forward(self, document, doc_char_tensor, doc_len, answer,
                answer_len, src_map, alignment, question=None, ques_char_tensor=None, ques_len=None):
        """
        Input:
            - document: ``(batch_size, max_doc_len)``
            - doc_char_tensor: ``(batch_size, max_doc_len, max_word_len)``
            - doc_len: ``(batch_size)``
            - question: ``(batch_size, max_que_len)``
            - ques_char_tensor: ``(batch_size, max_que_len, max_word_len)``
            - ques_len: ``(batch_size)``
            - answer: ``(batch_size, max_ans_len)``
            - answer_len: ``(batch_size)``
        Output:
            - ``(batch_size, P_LEN)``, ``(batch_size, P_LEN)``
        """
        # memory_bank: B x P x h; hidden: l*num_directions x B x h
        if self.reader_type == 'summarizer':
            encoder_out = self.encoder(document, doc_char_tensor, doc_len)
        else:
            encoder_out = self.encoder(document, doc_char_tensor, doc_len,
                question, ques_char_tensor, ques_len)

        if self.reader_type == 'bidaf' or self.reader_type == 'ibidaf':
            G, M, hidden = encoder_out
            G_M = torch.cat((G, M), 2)
            # B`xPx5h/(l + h)` ---> `BxPxh`
            memory_bank = self.dropout(self.transform(G_M))
        elif self.reader_type == 'mlstm':
            memory_bank, hidden = encoder_out
        elif self.reader_type == 'areader':
            memory_bank, _, hidden = encoder_out
        elif self.reader_type == 'summarizer':
            memory_bank, hidden = encoder_out

        init_decoder_state = self.decoder.init_decoder_state(hidden)
        tgt = self.dropout(self.word_embeddings(answer.unsqueeze(2)))

        decoder_outputs, state, attns = self.decoder(tgt,
                                                     memory_bank,
                                                     init_decoder_state,
                                                     memory_lengths=doc_len.data)

        if self.copy_attn:
            scores = self.copy_generator(decoder_outputs, attns["copy"], src_map)
            scores = scores[:, :-1, :].contiguous()
            loss = self.criterion(scores,
                                  alignment[:, 1:].contiguous(),
                                  answer[:, 1:].contiguous())
            loss = loss.view(*scores.size()[:2])
        else:
            dec_preds = self.generator(decoder_outputs)  # batch x tgt_len x vocab_size
            dec_preds = self.softmax(dec_preds)
            dec_preds = dec_preds[:, :-1, :].contiguous()  # batch x tgt_len - 1 x vocab_size

            loss = f.nll_loss(dec_preds.view(-1, dec_preds.size(2)),
                              answer[:, 1:].contiguous().view(-1),
                              reduce=False)
            loss = loss.view(*dec_preds.size()[:2])
            mask = sequence_mask(answer_len.data - 1)
            loss = loss * Variable(mask).float()

        loss = loss.sum(1).mean()
        return loss

    def get_decoding_params(self):
        if(self.reader_type=='summarizer'):
            return ['doc_rep', 'doc_char_rep', 'doc_len',
                    'src_map', 'alignment']
        else:
            return ['doc_rep', 'doc_char_rep', 'doc_len',
                    'que_rep', 'que_char_rep', 'que_len',
                    'src_map', 'alignment']

    def decode(self, document, doc_char_tensor, doc_len,
               src_map, alignment, max_len, tgt_dict, question=None, ques_char_tensor=None, ques_len=None):
        if self.reader_type == 'summarizer':
            encoder_out = self.encoder(document, doc_char_tensor, doc_len)
        else:
            encoder_out = self.encoder(document, doc_char_tensor, doc_len,
                question, ques_char_tensor, ques_len)

        if self.reader_type == 'bidaf' or self.reader_type == 'ibidaf':
            G, M, hidden = encoder_out
            G_M = torch.cat((G, M), 2)
            memory_bank = self.transform(G_M)
        elif self.reader_type == 'mlstm':
            memory_bank, hidden = encoder_out
        elif self.reader_type == 'areader':
            memory_bank, _, hidden = encoder_out
        elif self.reader_type == 'summarizer':
            memory_bank, hidden = encoder_out

        init_decoder_state = self.decoder.init_decoder_state(hidden)
        tgt = Variable(torch.LongTensor([BOS]))
        if document.is_cuda:
            tgt = tgt.cuda()
        tgt = tgt.expand(document.size(0)).unsqueeze(1)  # B x 1

        dec_preds = []
        for idx in range(max_len):
            tgt = self.word_embeddings(tgt.unsqueeze(2))
            decoder_outputs, state, attns = self.decoder(tgt,
                                                         memory_bank,
                                                         init_decoder_state,
                                                         memory_lengths=doc_len.data)
            if self.copy_attn:
                prediction = self.copy_generator(decoder_outputs, attns["copy"], src_map)
                prediction = prediction.squeeze(1)
            else:
                prediction = self.generator(decoder_outputs.squeeze(1))
                prediction = self.softmax(prediction)

            _, tgt = torch.max(prediction, dim=1, keepdim=True)
            dec_preds.append(tgt.squeeze(1).clone())
            if self.copy_attn:
                # ref: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/translator.py#L546
                tgt = tgt.masked_fill(tgt.gt(len(tgt_dict) - 1), UNK)

        return torch.stack(dec_preds, dim=1)
