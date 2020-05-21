import collections
import json
import logging
from typing import List, Tuple

import nltk
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from tqdm import tqdm

from data.data_instance import QAFullInputFeatures, QAFullExample, ModelState, FullResult
from general_util import utils

logger = logging.getLogger(__name__)


class FeverReader(object):
    def __init__(self):
        super(FeverReader, self).__init__()
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.yesno_cate = utils.CategoricalAccuracy(['yes', 'no'])

    def read(self, input_file):
        logger.info(f'Reading data set from {input_file}...')
        with open(input_file, 'r') as f:
            data = json.load(f)

        def is_whitespace(ch):
            if ch == " " or ch == "\t" or ch == "\r" or ch == "\n" or ord(ch) == 0x202F:
                return True
            return False

        examples = []
        for instance_id in tqdm(data, desc=f'Reading examples from {input_file}...', total=len(data)):
            claim = data[instance_id]['claim']
            sentence_id = data[instance_id]['evidence']
            label = data[instance_id]['label'].lower()
            passage = data[instance_id]['passage']

            doc_tokens = []
            prev_is_whitespace = True
            char_to_word_offset = []
            for c in passage:
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
            sentence_start_list, sentence_end_list = utils.split_sentence(passage, self.sentence_tokenizer)
            sentence_span_list = []
            for c_start, c_end in zip(sentence_start_list, sentence_end_list):
                t_start = char_to_word_offset[c_start]
                t_end = char_to_word_offset[c_end]
                sentence_span_list.append((t_start, t_end))

            if label == 'yes':
                answer_choice = 0
            elif label == 'no':
                answer_choice = 1
            else:
                raise RuntimeError(f'Wrong label for {label}')

            example = QAFullExample(
                qas_id=instance_id,
                question_text=claim,
                doc_tokens=doc_tokens,
                sentence_span_list=sentence_span_list,
                orig_answer_text="",
                start_position=None,
                end_position=None,
                sentence_id=sentence_id,
                is_impossible=answer_choice,
                ral_start_position=None,
                ral_end_position=None
            )
            examples.append(example)
        return examples

    @staticmethod
    def convert_examples_to_features(examples: List[QAFullExample], tokenizer: BertTokenizer, max_seq_length: int, doc_stride: int,
                                     max_query_length: int, is_training: bool):
        unique_id = 1000000000
        features = []
        for (example_index, example) in tqdm(enumerate(examples), desc='Converting examples to features..', total=len(examples)):
            query_tokens = tokenizer.tokenize(example.question_text)

            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[-max_query_length:]

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example.doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
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

            ini_sen_id: List[int] = example.sentence_id
            for (doc_span_index, doc_span) in enumerate(doc_spans):

                token_to_orig_map = {}
                token_is_max_context = {}
                tokens = ["[CLS]"] + query_tokens + ["[SEP]"]
                segment_ids = [0] * len(tokens)

                doc_start = doc_span.start
                doc_offset = len(query_tokens) + 2
                sentence_list = sentence_spans_list[doc_span_index]
                cur_sentence_list = []
                for sen_id, sen in enumerate(sentence_list):
                    new_sen = (sen[0] - doc_start + doc_offset, sen[1] - doc_start + doc_offset)
                    cur_sentence_list.append(new_sen)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                    is_max_context = utils.check_is_max_context(doc_spans, doc_span_index, split_token_index)

                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == len(input_mask) == len(segment_ids) == max_seq_length

                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1

                """
                There are multiple evidence sentences in some examples. To avoid multi-label setting,
                we choose to use the evidence sentence with the max length.
                """
                span_sen_id = -1
                max_evidence_length = 0
                for piece_sen_id, sen_id in enumerate(sentence_ids_list[doc_span_index]):
                    if sen_id in ini_sen_id:
                        evidence_length = cur_sentence_list[piece_sen_id][1] - cur_sentence_list[piece_sen_id][0]
                        if evidence_length > max_evidence_length:
                            max_evidence_length = evidence_length
                            span_sen_id = piece_sen_id
                meta_data = {
                    'span_sen_to_orig_sen_map': sentence_ids_list[doc_span_index]
                }

                if span_sen_id == -1:
                    answer_choice = 0
                else:
                    answer_choice = example.is_impossible + 1

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
                    ral_start_position=None,
                    ral_end_position=None,
                    meta_data=meta_data
                ))
            unique_id += 1
        return features

    def write_predictions(self, all_examples, all_features, all_results: List[FullResult], output_prediction_file: str = None,
                          null_score_diff_threshold: float = 0.0):
        self.yesno_cate.reset()
        logger.info("Writing predictions to: %s" % output_prediction_file)

        example_index_to_features = collections.defaultdict(list)
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        all_predictions = collections.OrderedDict()

        for (example_index, example) in enumerate(all_examples):
            features = example_index_to_features[example_index]

            max_diff = -1000000
            max_diff_yes_logit = 0
            max_diff_no_logit = 0
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                choice_logits = result.choice_logits
                non_null_logit = choice_logits[1] + choice_logits[2]
                yes_logit = choice_logits[1]
                no_logit = choice_logits[2]
                null_logit = choice_logits[0]
                diff = non_null_logit - null_logit
                if diff > max_diff:
                    max_diff = diff
                    max_diff_yes_logit = yes_logit
                    max_diff_no_logit = no_logit

            if max_diff_yes_logit > max_diff_no_logit:
                final_text = 'yes'
                scores = max_diff_yes_logit
            else:
                final_text = 'no'
                scores = max_diff_no_logit
            all_predictions[example.qas_id] = {
                'answer': final_text,
                'scores': scores
            }

            gold_label = 'yes' if example.is_impossible == 0 else 'no'
            self.yesno_cate.update(gold_label, final_text)

        logger.info('Yes/No Metric: %s' % self.yesno_cate)

        if output_prediction_file is not None:
            with open(output_prediction_file, 'w') as f:
                json.dump(all_predictions, f, indent=2)
        return all_predictions

    @staticmethod
    def data_to_tensors(all_features: List[QAFullInputFeatures]):
        all_input_ids = torch.LongTensor([f.input_ids for f in all_features])
        all_segment_ids = torch.LongTensor([f.segment_ids for f in all_features])
        all_input_mask = torch.LongTensor([f.input_mask for f in all_features])
        all_answer_choice = torch.LongTensor([f.is_impossible for f in all_features])
        all_sentence_ids = torch.LongTensor([f.sentence_id for f in all_features])
        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        return all_input_ids, all_segment_ids, all_input_mask, all_answer_choice, all_sentence_ids, all_feature_index

    @staticmethod
    def generate_inputs(batch: Tuple, all_features: List[QAFullInputFeatures], model_state, do_label=False):
        assert model_state in ModelState
        feature_index = batch[5].tolist()
        sentence_span_list = [all_features[idx].sentence_span_list for idx in feature_index]
        inputs = {
            'input_ids': batch[0],
            'token_type_ids': batch[1],
            'attention_mask': batch[2],
            'sentence_span_list': sentence_span_list
        }
        if model_state == ModelState.Test:
            return inputs
        if model_state == ModelState.Train or model_state == ModelState.Evaluate:
            inputs['answer_choice'] = batch[3]
            inputs['sentence_ids'] = batch[4]

        return inputs
