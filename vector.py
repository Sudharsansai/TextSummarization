# src: https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/vector.py

import torch
import numpy as np

from nqa.inputters.constants import UNK
from nqa.models import elmo_sent_mapper


def vectorize(ex, model, single_answer=False):
    """Torchify a single example."""
    word_dict = model.word_dict
    tgt_dict = model.tgt_dict

    if model.args.use_elmo:
        # Index words
        document = torch.from_numpy(np.asarray(elmo_sent_mapper(ex['document']), dtype=np.int))
        if(model.args.reader_type!='summarizer'):
            question = torch.from_numpy(np.asarray(elmo_sent_mapper(ex['question']), dtype=np.int))

        # When using elmo, char representations are not required
        document_chars, question_chars = -1, -1
    else:
        # Index words
        document = torch.LongTensor([word_dict[w] for w in ex['document']])
        if(model.args.reader_type!='summarizer'):
            question = torch.LongTensor([word_dict[w] for w in ex['question']])

        # Index chars
        if model.args.reader_type == 'mlstm' or \
                        model.args.reader_type == 'areader':
            document_chars, question_chars = -2, -2
        else:
            document_chars = torch.LongTensor([word_dict.word_to_char_ids(w).tolist() for w in ex['document']])
            if(model.args.reader_type!='summarizer'):
                question_chars = torch.LongTensor([word_dict.word_to_char_ids(w).tolist() for w in ex['question']])

    # ...or with target(s) (might still be empty if answers is empty)
    # handle if questions are unanswerable [SQuAD v2.0]
    if(model.args.reader_type!='summarizer'):
            
        if single_answer:
            assert (len(ex['answers']) > 0)
            answer = torch.LongTensor([tgt_dict[w] for w in ex['answers'][0]])
        else:
            # FIXME: multiple answers are possible, fix batchify also.
            # answer = [torch.LongTensor([word_dict[w] for w in ans]) for ans in ex['answers']]
            answer = torch.LongTensor([tgt_dict[w] for w in ex['answers'][0]])
    else:
        summary = torch.LongTensor([tgt_dict[w] for w in ex['summary']])


    if(model.args.reader_type!='summarizer'):
        return document, document_chars, question, question_chars, answer, \
        ex['id'], ex['document'], ex['question'], ex['answers'], \
        ex['src_vocab']
    else:
        return document, document_chars, summary, ex['id'], ex['document'], ex['summary'], ex['src_vocab']


def batchify_QA(batch):
    """Gather a batch of individual examples into one batch."""

    if type(batch[0][1]) != torch.LongTensor:
        no_elmo, use_char = (True, False) if batch[0][1] == -2 else (False, False)
    else:
        no_elmo, use_char = True, True

    docs = [ex[0] for ex in batch]
    docs_char = [ex[1] for ex in batch]
    ques = [ex[2] for ex in batch]
    questions_char = [ex[3] for ex in batch]
    answers = [ex[4] for ex in batch]

    # Batch documents
    max_doc_length = max([d.size(0) for d in docs])
    x1_len = torch.LongTensor(len(docs)).zero_()
    x1 = torch.LongTensor(len(docs),
                          max_doc_length).zero_() if no_elmo else torch.LongTensor(len(docs),
                                                                                   max_doc_length,
                                                                                   50).zero_()
    x1_char = torch.LongTensor(len(docs),
                               max_doc_length,
                               docs_char[0].size(1)).zero_() if (no_elmo and use_char) else None
    for i, d in enumerate(docs):
        x1_len[i] = d.size(0)
        x1[i, :d.size(0)].copy_(d)
        if not no_elmo:
            x1_char[i, :d.size(0), :].copy_(docs_char[i])

    # Batch questions
    max_ques_length = max([q.size(0) for q in ques])
    x2_len = torch.LongTensor(len(ques)).zero_()
    x2 = torch.LongTensor(len(ques), max_ques_length).zero_() if no_elmo else torch.LongTensor(len(ques),
                                                                                               max_ques_length,
                                                                                               50).zero_()
    x2_char = torch.LongTensor(len(ques),
                               max_ques_length,
                               questions_char[0].size(1)).zero_() if (no_elmo and use_char) else None
    for i, q in enumerate(ques):
        x2_len[i] = q.size(0)
        x2[i, :q.size(0)].copy_(q)
        if not no_elmo:
            x2_char[i, :q.size(0), :].copy_(questions_char[i])

    # Batch answers
    max_ans_length = max([a.size(0) for a in answers])
    ans_len = torch.LongTensor(len(answers)).zero_()
    ans = torch.LongTensor(len(answers), max_ans_length).zero_()
    for i, a in enumerate(answers):
        ans_len[i] = a.size(0)
        ans[i, :a.size(0)].copy_(a)

    ids = [ex[5] for ex in batch]
    contexts = [ex[6] for ex in batch]
    # FIXME: multiple answers are possible, fix vectorize also.
    targets = [ex[8][0] for ex in batch]
    src_vocabs = [ex[9] for ex in batch]
    source_maps = []
    alignments = []

    # Prepare source vocabs, alignment [required for Copy Attention]
    for eid, context, target, (token2idx, idx2token) in \
            zip(ids, contexts, targets, src_vocabs):
        # Mapping source tokens to indices in the dynamic dict.
        src_map = torch.LongTensor([token2idx[w] for w in context])
        source_maps.append(src_map)

        # TODO: does skipping the first and last token in answer valid?
        mask = torch.LongTensor([token2idx[w] if w in token2idx
                                 else UNK for w in target])
        alignments.append(mask)

    return {'doc_rep': x1,
            'doc_char_rep': x1_char,
            'doc_len': x1_len,
            'que_rep': x2,
            'que_char_rep': x2_char,
            'que_len': x2_len,
            'ans_rep': ans,
            'ans_len': ans_len,
            'ids': ids,
            'documents': contexts,
            'questions': [ex[7] for ex in batch],
            'answers': targets,
            'source_vocabs': src_vocabs,
            'src_map': source_maps,
            'alignment': alignments}


'''
document, document_chars, summary, ex['id'], ex['document'], ex['summary'], ex['src_vocab']
'''
def batchify_summary(batch):
    """Gather a batch of individual examples into one batch."""

    if type(batch[0][1]) != torch.LongTensor:
        no_elmo, use_char = (True, False) if batch[0][1] == -2 else (False, False)
    else:
        no_elmo, use_char = True, True

    docs = [ex[0] for ex in batch]
    docs_char = [ex[1] for ex in batch]
    summaries = [ex[2] for ex in batch]

    # Batch documents
    max_doc_length = max([d.size(0) for d in docs])
    x1_len = torch.LongTensor(len(docs)).zero_()
    x1 = torch.LongTensor(len(docs),
                          max_doc_length).zero_() if no_elmo else torch.LongTensor(len(docs),
                                                                                   max_doc_length,
                                                                                   50).zero_()
    x1_char = torch.LongTensor(len(docs),
                               max_doc_length,
                               docs_char[0].size(1)).zero_() if (no_elmo and use_char) else None
    for i, d in enumerate(docs):
        x1_len[i] = d.size(0)
        x1[i, :d.size(0)].copy_(d)
        if not no_elmo:
            x1_char[i, :d.size(0), :].copy_(docs_char[i])

    # Batch answers
    max_ans_length = max([a.size(0) for a in summaries])
    ans_len = torch.LongTensor(len(summaries)).zero_()
    ans = torch.LongTensor(len(summaries), max_ans_length).zero_()
    for i, a in enumerate(summaries):
        ans_len[i] = a.size(0)
        ans[i, :a.size(0)].copy_(a)

    ids = [ex[3] for ex in batch]
    contexts = [ex[4] for ex in batch]
    # FIXME: multiple answers are possible, fix vectorize also.
    targets = [ex[5] for ex in batch]
    src_vocabs = [ex[6] for ex in batch]
    source_maps = []
    alignments = []

    # Prepare source vocabs, alignment [required for Copy Attention]
    for eid, context, target, (token2idx, idx2token) in \
            zip(ids, contexts, targets, src_vocabs):
        # Mapping source tokens to indices in the dynamic dict.
        src_map = torch.LongTensor([token2idx[w] for w in context])
        source_maps.append(src_map)

        # TODO: does skipping the first and last token in answer valid?
        mask = torch.LongTensor([token2idx[w] if w in token2idx
                                 else UNK for w in target])
        alignments.append(mask)

    return {'doc_rep': x1,
            'doc_char_rep': x1_char,
            'doc_len': x1_len,
            'summ_rep': ans,
            'summ_len': ans_len,
            'ids': ids,
            'documents': contexts,
            'answers': targets,
            'source_vocabs': src_vocabs,
            'src_map': source_maps,
            'alignment': alignments}
