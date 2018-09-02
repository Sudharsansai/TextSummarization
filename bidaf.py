import torch
import torch.nn as nn
import torch.nn.functional as f
from collections import OrderedDict

from nqa.models.elmo import ELMo

from nqa.inputters import PAD
from nqa.modules.embeddings import Embeddings
from nqa.encoders import RNNEncoder
from nqa.modules.highway import Highway
from nqa.modules.char_embedding import CharEmbedding
from nqa.modules.matrix_attn import MatrixAttention


# model details can be found at https://arxiv.org/pdf/1611.01603.pdf
class BIDAF(nn.Module):
    """Bidirectional Attention Flow Network that finds answer span for the question from the given passage."""

    def __init__(self, args):
        """"Constructor of the class."""
        super(BIDAF, self).__init__()

        self.use_elmo = args.use_elmo
        self.model_type = args.reader_type
        input_size = args.emsize
        if args.use_elmo:
            self.elmo = ELMo(args.optfile,
                             args.wgtfile,
                             proj_dim=args.emsize,
                             dropout=args.dropout_emb,
                             requires_grad=False)
        else:
            self.word_embeddings = nn.Sequential(OrderedDict([
                ('embedding', Embeddings(args.emsize,
                                         args.vocab_size,
                                         PAD)),
                ('dropout', nn.Dropout(p=args.dropout_emb))
            ]))
            self.char_embeddings = nn.Sequential(OrderedDict([
                ('embedding', CharEmbedding(args.n_characters,
                                            args.char_emsize,
                                            args.filter_size,
                                            args.nfilters)),
                ('dropout', nn.Dropout(p=args.dropout_emb))
            ]))
            input_size += sum(list(map(int, args.nfilters)))
            self.highway_net = Highway(input_size, num_layers=2)

        self.ctx_embd_layer = RNNEncoder(args.rnn_type,
                                         input_size,
                                         args.bidirection,
                                         args.nlayers,
                                         args.nhid,
                                         args.dropout_rnn)

        self.matrix_attn_layer = MatrixAttention(args.nhid)
        self.modeling_layer = RNNEncoder(args.rnn_type,
                                         args.nhid * 4,
                                         args.bidirection,
                                         args.nlayers,
                                         args.nhid,
                                         args.dropout_rnn)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, document, doc_char_tensor, doc_len, question=None, ques_char_tensor=None, ques_len=None):
        """
        Input:
            - document: ``(batch_size, max_doc_len)``
            - doc_char_tensor: ``(batch_size, max_doc_len, max_word_len)``
            - doc_len: ``(batch_size)``
            - question: ``(batch_size, max_que_len)``
            - ques_char_tensor: ``(batch_size, max_que_len, max_word_len)``
            - ques_len: ``(batch_size)``
        Output:
            - ``(batch_size, P_LEN)``, ``(batch_size, P_LEN)``
        """
        if self.use_elmo:
            # ------------- Optional ELMo Layer -------------
            X = self.elmo(document)  # (batch_size, num_timesteps, input_size)
            if(self.model_type != 'summarizer'):
                Q = self.elmo(question)  # (batch_size, num_timesteps, input_size)
        else:
            # ------------- Character Embedding Layer -------------
            X_char = self.char_embeddings(doc_char_tensor)  # B x P x d
            if(self.model_type != 'summarizer'):
                Q_char = self.char_embeddings(ques_char_tensor)  # B x Q x d

            # ------------- Word Embedding Layer -------------
            X_word = self.word_embeddings(document.unsqueeze(2))  # B x P x d
            if(self.model_type != 'summarizer'):
                Q_word = self.word_embeddings(question.unsqueeze(2))  # B x Q x d

            # combine word embeddings
            X = torch.cat((X_word, X_char), 2)  # B x P x d+f
            if(self.model_type != 'summarizer'):
                Q = torch.cat((Q_word, Q_char), 2)  # B x Q x d+f

            # pass combined embeddings through the highway network
            X = self.dropout(self.highway_net(X))  # B x P x d+f
            if(self.model_type != 'summarizer'):
                Q = self.dropout(self.highway_net(Q))  # B x Q x d+f

        # ------------- Contextual Embedding Layer -------------

        if(self.model_type == 'summarizer'):
            encoder_out = self.ctx_embd_layer(X, doc_len.data)
            tensor_H = self.dropout(encoder_out[1])  # B x P x hs
            # return memory_bank, hidden
            return tensor_H, encoder_out[0]
            
        tensor_H = self.dropout(self.ctx_embd_layer(X, doc_len.data)[1])  # B x P x h
        tensor_U = self.dropout(self.ctx_embd_layer(Q, ques_len.data)[1])  # B x Q x h

        # ------------- Attention Flow Layer -------------
        # compute \alpha(h,u)
        S = self.matrix_attn_layer(tensor_H, tensor_U)  # B x P x Q

        # c2q: context-to-query attention
        c2q = torch.bmm(f.softmax(S, dim=2), tensor_U)  # B x P x h

        # q2c: query-to-context attention
        b = f.softmax(torch.max(S, 2)[0], dim=-1)  # B x P
        q2c = torch.bmm(b.unsqueeze(1), tensor_H)  # B x 1 x h
        q2c = q2c.repeat(1, tensor_H.size(1), 1)  # B x P x h , tiled P times

        # G: query aware representation of each context word
        G = torch.cat((tensor_H, c2q, tensor_H.mul(c2q), tensor_H.mul(q2c)), 2)  # B x P x 4h

        # ------------- Modeling Layer -------------
        hidden, M = self.modeling_layer(G, None)
        M = self.dropout(M)  # M: B x P x h

        return G, M, hidden
