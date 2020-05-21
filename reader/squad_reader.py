import collections
import json
from typing import List, Tuple, Dict

import nltk
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from tqdm import tqdm

from data.data_instance import SQuADFullExample, QAFullInputFeatures, WeightResult, ModelState
from general_util import utils
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)

"""
This reader has several usages:
    - Read SQuAD examples to pretrain Hierarchical Attention.
"""


class SQuADReader(object):
    def __init__(self):
        super(SQuADReader, self).__init__()
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def read(self, input_file):
        """
        :param input_file: input file to load data. The format is in CoQA style
        """
        logger.info('Reading data set from {}...'.format(input_file))
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)['data']

        def is_whitespace(ch):
            if ch == " " or ch == "\t" or ch == "\r" or ch == "\n" or ord(ch) == 0x202F:
                return True
            return False

        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                # Split context into sentences
                sentence_start_list, sentence_end_list = utils.split_sentence(paragraph_text, self.sentence_tokenizer)
                sentence_span_list = []
                for c_start, c_end in zip(sentence_start_list, sentence_end_list):
                    t_start = char_to_word_offset[c_start]
                    t_end = char_to_word_offset[c_end]
                    sentence_span_list.append((t_start, t_end))

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    answer = qa["answers"][0]
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]

                    sentence_id = utils.find_evidence_sentence(sentence_span_list, start_position, end_position)

                    example = SQuADFullExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        sentence_span_list=sentence_span_list,
                        orig_answer_text="",
                        start_position=None,
                        end_position=None,
                        sentence_id=sentence_id,
                        is_impossible=-1,
                        ral_start_position=start_position,
                        ral_end_position=end_position)
                    examples.append(example)

        return examples

    @staticmethod
    def convert_examples_to_features(examples: List[SQuADFullExample], tokenizer: BertTokenizer, max_seq_length, doc_stride,
                                     max_query_length, is_training: bool):
        """Loads a data file into a list of `InputBatch`s."""

        unique_id = 1000000000
        features = []
        for (example_index, example) in tqdm(enumerate(examples), desc='Convert examples to features', total=len(examples)):
            query_tokens = tokenizer.tokenize(example.question_text)

            if len(query_tokens) > max_query_length:
                # query_tokens = query_tokens[0:max_query_length]
                # Remove the tokens appended at the front of query, which may belong to last query and answer.
                query_tokens = query_tokens[-max_query_length:]

            # word piece index -> token index
            tok_to_orig_index = []
            # token index -> word pieces group start index
            # BertTokenizer.tokenize(doc_tokens[i]) = all_doc_tokens[orig_to_tok_index[i]: orig_to_tok_index[i + 1]]
            orig_to_tok_index = []
            # word pieces for all doc tokens
            all_doc_tokens = []
            for (i, token) in enumerate(example.doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            # Process sentence span list
            sentence_spans = []
            for (start, end) in example.sentence_span_list:
                piece_start = orig_to_tok_index[start]
                if end < len(example.doc_tokens) - 1:
                    piece_end = orig_to_tok_index[end + 1] - 1
                else:
                    piece_end = len(all_doc_tokens) - 1
                sentence_spans.append((piece_start, piece_end))

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
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
            for (doc_span_index, doc_span) in enumerate(doc_spans):
                # Store the input tokens to transform into input ids later.
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

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                # ral_start = None
                # ral_end = None
                # answer_choice = None

                answer_choice = -1

                # Process sentence id
                span_sen_id = -1
                for piece_sen_id, sen_id in enumerate(sentence_ids_list[doc_span_index]):
                    if ini_sen_id == sen_id:
                        span_sen_id = piece_sen_id
                meta_data = {'span_sen_to_orig_sen_map': sentence_ids_list[doc_span_index]}

                if example_index < 0:
                    logger.info("*** Example ***")
                    logger.info("unique_id: %s" % unique_id)
                    logger.info("example_index: %s" % example_index)
                    logger.info("doc_span_index: %s" % doc_span_index)
                    logger.info(
                        "sentence_spans_list: %s" % " ".join(
                            [(str(x[0]) + '-' + str(x[1])) for x in cur_sentence_list]))

                    logger.info("answer choice: %s" % str(answer_choice))

                features.append(QAFullInputFeatures(
                    qas_id=example.qas_id,
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    sentence_span_list=cur_sentence_list,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    is_impossible=answer_choice,
                    sentence_id=span_sen_id,
                    start_position=None,
                    end_position=None,
                    ral_start_position=-1,
                    ral_end_position=-1,
                    meta_data=meta_data))

                unique_id += 1

        return features

    @staticmethod
    def write_sentence_predictions(all_examples, all_features: List[QAFullInputFeatures], all_results: List[WeightResult],
                                   output_prediction_file=None, null_score_diff_threshold=0.0):
        """Write final predictions to the json file and log-odds of null if needed."""
        logger.info("Writing predictions to: %s" % output_prediction_file)

        sentence_pred = utils.AverageMeter()

        example_index_to_features = collections.defaultdict(list)
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        all_predictions = collections.OrderedDict()

        for (example_index, example) in enumerate(all_examples):
            features = example_index_to_features[example_index]

            max_weight_logit = 0
            max_weight_feature_index = -1
            max_weight_index = -1
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                feature_max_weight = result.max_weight
                if feature_max_weight > max_weight_logit:
                    max_weight_logit = feature_max_weight
                    max_weight_feature_index = feature_index
                    max_weight_index = result.max_weight_index

            target_feature = features[max_weight_feature_index]
            sentence_id = target_feature.meta_data['span_sen_to_orig_sen_map'][max_weight_index]
            target_result = unique_id_to_result[target_feature.unique_id]
            all_predictions[example.qas_id] = {
                'sentence_id': sentence_id,
                'max_weight': max_weight_logit,
                'max_weight_index': max_weight_index,
                'sentence_logits': target_result.sentence_logits,
                'feature_gold_sentence': target_feature.sentence_id
            }

            if sentence_id == example.sentence_id:
                sentence_pred.update(1, 1)
            else:
                sentence_pred.update(0, 1)

        logger.info('Sentence Prediction Result: {}'.format(sentence_pred.avg))

        if output_prediction_file is not None:
            with open(output_prediction_file, 'w') as f:
                json.dump(all_predictions, f, indent=2)
        return {'predictions': all_predictions, 'acc': sentence_pred.avg}

    @staticmethod
    def data_to_tensors(all_features: List[QAFullInputFeatures]):

        all_input_ids = torch.tensor([f.input_ids for f in all_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in all_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in all_features], dtype=torch.long)
        all_sentence_ids = torch.tensor([f.sentence_id for f in all_features], dtype=torch.long)
        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        return all_input_ids, all_segment_ids, all_input_mask, all_sentence_ids, all_feature_index

    @staticmethod
    def generate_inputs(batch: Tuple, all_features: List[QAFullInputFeatures], model_state):
        assert model_state in ModelState
        feature_index = batch[4].tolist()
        sentence_span_list = [all_features[idx].sentence_span_list for idx in feature_index]
        inputs = {
            "input_ids": batch[0],
            "token_type_ids": batch[1],
            "attention_mask": batch[2],
            "sentence_span_list": sentence_span_list
        }
        if model_state == ModelState.Train or model_state == ModelState.Evaluate:
            inputs['sentence_ids'] = batch[3]

        return inputs
