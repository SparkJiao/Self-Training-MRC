import collections
import json
import pickle
from typing import List, Tuple

import nltk
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer, whitespace_tokenize
from tqdm import tqdm

from data.data_instance import QAFullExample, QAFullInputFeatures, RawResultChoice, WeightResultChoice, ModelState, ReadState, \
    FullResult
from general_util import utils
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)

"""
The reader is used to read dataset which contains evidence labels at first, so that we can use slide window
to split the passage into many segments while the length of passage exceeds the max length.
At the same time, we can label each segment with the initial Yes/No label if the segment contains
the evidence or "Unknown" on the other hand.
"""


class ReaderE(object):
    def __init__(self, vocab_file):
        super(ReaderE, self).__init__()
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.bert_tokenizer = BertTokenizer.from_pretrained(vocab_file)

    def read(self, input_file, sentence_id_file: str = None) -> List[QAFullExample]:
        logger.info(f'Reading data from {input_file}')

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        def is_whitespace(ch):
            if ch == " " or ch == "\t" or ch == "\r" or ch == "\n" or ord(ch) == 0x202F:
                return True
            return False

        if sentence_id_file is not None:
            with open(sentence_id_file, 'r') as f:
                sentence_ids = json.load(f)
        else:
            sentence_ids = None

        examples = []
        for instance in data:
            article = instance['article']
            data_id = instance['id']
            question = instance['question']
            answer = instance['answer']

            doc_tokens = []
            prev_is_whitespace = True
            char_to_word_offset = []
            for c in article:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            sentence_start_list, sentence_end_list = utils.split_sentence(article, self.sentence_tokenizer)
            sentence_span_list = []
            for c_start, c_end in zip(sentence_start_list, sentence_end_list):
                t_start = char_to_word_offset[c_start]
                t_end = char_to_word_offset[c_end]
                sentence_span_list.append((t_start, t_end))

            sentence_id = instance['id'] if sentence_ids is None else sentence_ids[data_id]

            example = QAFullExample(
                qas_id=data_id,
                question_text=question,
                doc_tokens=doc_tokens,
                sentence_span_list=sentence_span_list,
                sentence_id=sentence_id,
                is_impossible=answer
            )
            examples.append(example)

        return examples

    def convert_examples_to_features(self, examples: List[QAFullExample], max_seq_length: int, doc_stride: int, max_query_length: int):
        unique_id = 1000000000
        features = []

        for example_index, example in tqdm(enumerate(examples), desc='Convert examples to features', total=len(examples)):
            query_tokens = self.bert_tokenizer.toke

            if len(query_tokens) > max_query_length:
                # query_tokens = query_tokens[0:max_query_length]
                # Remove the tokens appended at the front of query, which may belong to last query and answer.
                query_tokens = query_tokens[-max_query_length:]

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example.doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = self.bert_tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            sentence_spans = []
            for (start, end) in example.sentence_span_list:
                piece_start = orig_to_tok_index[start]
                if end < len(example.doc_tokens) - 1:
                    piece_end = orig_to_tok_index[end + 1] - 1
                else:
                    piece_end = len(all_doc_tokens) - 1
                sentence_spans.append((piece_start, piece_end))

            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

            _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)

            sentence_spans_list = []
            sentence_ids_list = []
            for span_id, doc_span in enumerate(doc_spans):
                span_start = doc_span.start
                span_end = span_start + doc_span.length - 1

                span_sentence = []
                sen_ids = []

                for sen_idx, (sen_start, sen_end) in enumerate(sentence_spans):
                    if sen_end < span_start:
                        continue
                    if sen_start > span_end:
                        break
                    span_sentence.append((max(sen_start, span_start), min(sen_end, span_end)))
                    sen_ids.append(sen_idx)

                sentence_spans_list.append(span_sentence)
                sentence_ids_list.append(sen_ids)

            ini_sen_id = example.sentence_id
            for doc_span_index, doc_span in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                doc_start = doc_span.start
                doc_offset = len(query_tokens) + 2
                sentence_list = sentence_spans_list[doc_span_index]
                cur_sentence_list = []
                for sen_id, sen in enumerate(sentence_list):
                    new_sen = (sen[0] - doc_start + doc_offset, sen[1] - doc_start + doc_offset)
                    cur_sentence_list.append(new_sen)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i  # Original index of word piece in all_doc_tokens
                    # Index of word piece in input sequence -> Original word index in doc_tokens
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                    # Check if the word piece has the max context in all doc spans.
                    is_max_context = utils.check_is_max_context(doc_spans, doc_span_index, split_token_index)

                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                span_sen_id = -1
                for piece_sen_id, sen_id in enumerate(sentence_ids_list[doc_span_index]):
                    if ini_sen_id == sen_id:
                        span_sen_id = piece_sen_id

