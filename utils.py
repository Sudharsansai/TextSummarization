import json
import logging
from collections import Counter

from nqa.inputters.vocabulary import Vocabulary, UnicodeCharsVocabulary
from nqa.inputters.constants import BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD

logger = logging.getLogger(__name__)

def load_QA_data(args, filename, skip_no_answer=False,
              max_examples=-1, dataset_name='SQuAD'):
    """Load examples from preprocessed file.
    One example per line, JSON encoded.
    """
    # Load JSON lines
    with open(filename) as f:
        examples = [json.loads(line) for line in f]

    for ex in examples:
        # TODO: only a single passage is used in examples
        if dataset_name == 'MSMARCO':
            ex['document'] = ex['document'][0]

        if args.uncased_question:
            ex['question'] = [w.lower() for w in ex['question']]

        if args.uncased_doc:
            ex['document'] = [w.lower() for w in ex['document']]

        if ex['answers']:
            if dataset_name == 'MSMARCO' and args.uncased_answer:
                ex['answers'] = [[w.lower() for w in ans] for ans in ex['answers']]
            elif dataset_name == 'SQuAD':
                ex['answers'] = [ex['document'][a[0]: a[1] + 1] for a in ex['answers']]
            ex['answers'] = [[BOS_WORD] + ans + [EOS_WORD] for ans in ex['answers']]

        elif not skip_no_answer:
            ex['answers'] = [BOS_WORD, EOS_WORD]

        idx2token = list(set(ex['document'] + [UNK_WORD, PAD_WORD]))
        token2idx = {k: v for v, k in enumerate(idx2token)}
        ex['src_vocab'] = (token2idx, idx2token)

    # Skip unparsed (start/end) examples
    if skip_no_answer:
        examples = [ex for ex in examples if len(ex['answers']) > 0]

    if max_examples != -1:
        examples = [examples[i] for i in range(min(max_examples, len(examples)))]

    return examples

def load_summ_data(args, filename,max_examples=-1):
    """Load examples from preprocessed file.
    One example per line, JSON encoded.
    """
    # Load JSON lines
    with open(filename) as f:
        examples = [json.loads(line) for line in f]

    for ex in examples:
        # TODO: only a single passage is used in examples

        if args.uncased_doc:
            ex['document'] = [w.lower() for w in ex['document']]

        if args.uncased_answer:
            ex['summary'] = [w.lower() for w in ex['summary']]
        
        ex['summary'] = [BOS_WORD] + ex['summary'] + [EOS_WORD] 

        idx2token = list(set(ex['document'] + [UNK_WORD, PAD_WORD]))
        token2idx = {k: v for v, k in enumerate(idx2token)}
        ex['src_vocab'] = (token2idx, idx2token)


    if max_examples != -1:
        examples = [examples[i] for i in range(min(max_examples, len(examples)))]

    return examples

def load_answers(filename, dataset_name='SQuAD'):
    """Load the answers only of a SQuAD dataset. Store as qid -> [answers]."""
    # Load JSON file
    with open(filename) as f:
        examples = [json.loads(line) for line in f] if dataset_name == 'MSMARCO' \
            else json.load(f)['data']

    ans = {}
    if dataset_name == 'MSMARCO':
        for ex in examples:
            ans[ex['id']] = ex['answers']
    else:
        for article in examples:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    if qa['is_impossible']:
                        ans[qa['id']] = ['']
                    else:
                        ans[qa['id']] = list(map(lambda x: x['text'], qa['answers']))
    return ans

def index_embedding_words(embedding_file):
    """Put all the words in embedding_file into a set."""
    words = set()
    with open(embedding_file) as f:
        for line in f:
            w = Vocabulary.normalize(line.rstrip().split(' ')[0])
            words.add(w)

    words.update([BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD])
    return words


def load_words(args, examples, fields, dict_size=None):
    """Iterate and index all the words in examples (documents + questions)."""

    def _insert(iterable):
        words = []
        for w in iterable:
            w = Vocabulary.normalize(w)
            if valid_words and w not in valid_words:
                continue
            words.append(w)
        word_count.update(words)

    if args.restrict_vocab and args.embedding_file:
        logger.info('Restricting to words in %s' % args.embedding_file)
        valid_words = index_embedding_words(args.embedding_file)
        logger.info('Num words in set = %d' % len(valid_words))
    else:
        valid_words = None

    word_count = Counter()
    for ex in examples:
        for field in fields:
            _insert(ex[field])

    dict_size = dict_size - 4 if dict_size and dict_size > 4 else dict_size
    most_common = word_count.most_common(dict_size)
    words = set(word for word, _ in most_common)
    return words


def build_word_dict(args, examples, fields, dict_size=None):
    """Return a dictionary from question and document words in
    provided examples.
    """
    word_dict = Vocabulary()
    for w in load_words(args, examples, fields, dict_size):
        word_dict.add(w)
    return word_dict


def build_word_and_char_dict(args, examples, fields, dict_size=None):
    """Return a dictionary from question and document words in
    provided examples.
    """
    words = load_words(args, examples, fields, dict_size)
    dictionary = UnicodeCharsVocabulary(words, args.max_characters_per_token)
    return dictionary


def top_question_words(args, examples, word_dict):
    """Count and return the most common question words in provided examples."""
    word_count = Counter()
    for ex in examples:
        for w in ex['question']:
            w = Vocabulary.normalize(w)
            if w in word_dict:
                word_count.update([w])
    return word_count.most_common(args.tune_partial)
