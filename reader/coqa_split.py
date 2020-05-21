import collections
import json
from typing import List, Tuple

import nltk
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from tqdm import tqdm

from data.data_instance import QAFullExample, RawResultChoice, ModelState, ReadState, \
    SingleSentenceFeature
from general_util import utils
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)

"""
This reader has several usages:
    - Read simple CoQA Yes/No data ** with sentence supervision **.
    - Mask or re-write new sentence id supervision as you need.
    - The reader will add negative sample sentences into the examples and features.
    
TODO:
    - There are about 10% questions whose rationale contain more than one sentences. Currently we select the longest sentence.
    We could also consider about multi-label later on.
"""


class CoQASplitSentenceReader(object):
    def __init__(self):
        super(CoQASplitSentenceReader, self).__init__()
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.yesno_cate = utils.CategoricalAccuracy(['yes', 'no'])

    def read(self, input_file, dialog_turns: int = 2) -> List[QAFullExample]:
        """
        :param input_file: input file to load data. The format is in CoQA style
        :param read_state: If read extra sentences from CoQA dataset.
        :param dialog_turns:
        """
        logger.info('Reading data set from {}...'.format(input_file))
        logger.info('Read parameters:')
        logger.info('Dialog turns: {}'.format(dialog_turns))
        # logger.info('Read state: {}'.format(read_state))
        # assert read_state in ReadState
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)['data']

        def is_whitespace(ch):
            if ch == " " or ch == "\t" or ch == "\r" or ch == "\n" or ord(ch) == 0x202F:
                return True
            return False

        examples = []
        for paragraph in input_data:
            paragraph_text = paragraph["story"]
            story_id = paragraph['id']
            doc_tokens = []
            prev_is_whitespace = True
            char_to_word_offset = []
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

            doc_sentence_tokens = [doc_tokens[span[0]: (span[1] + 1)] for span in sentence_span_list]

            questions = paragraph['questions']
            answers = paragraph['answers']
            for i, (question, answer) in enumerate(zip(questions, answers)):
                question_text = question['input_text']

                # We are only concerned about questions with Yes/No as answers
                answer_type = utils.normalize_answer(answer['input_text'])
                if answer_type not in ['yes', 'no']:
                    continue
                if answer_type == 'yes':
                    answer_choice = 0
                else:
                    answer_choice = 1

                for j in range(dialog_turns):
                    pre_idx = i - (j + 1)
                    if pre_idx >= 0:
                        question_text = questions[pre_idx]['input_text'] + '<Q>' + answers[pre_idx][
                            'input_text'] + '<A>' + question_text

                qas_id = story_id + '--' + str(i + 1)

                # Add rationale start and end as extra supervised label.
                rationale_start_position = char_to_word_offset[answer['span_start']]
                rationale_end_position = char_to_word_offset[answer['span_end'] - 1]

                sentence_id = utils.find_evidence_sentence(sentence_span_list, rationale_start_position, rationale_end_position)

                example = QAFullExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_sentence_tokens,
                    sentence_span_list=sentence_span_list,
                    orig_answer_text="",
                    start_position=None,
                    end_position=None,
                    sentence_id=sentence_id,
                    is_impossible=answer_choice,
                    ral_start_position=rationale_start_position,
                    ral_end_position=rationale_end_position)
                examples.append(example)
        return examples

    @staticmethod
    def convert_examples_to_features(examples: List[QAFullExample], tokenizer: BertTokenizer, max_seq_length, doc_stride,
                                     max_query_length, is_training: bool):
        """Loads a data file into a list of `InputBatch`s."""

        unique_id = 1000000000
        features = []
        drop = 0
        for (example_index, example) in tqdm(enumerate(examples), desc='Convert examples to features', total=len(examples)):
            query_tokens = tokenizer.tokenize(example.question_text)

            # if len(query_tokens) > max_query_length:
            # query_tokens = query_tokens[0:max_query_length]
            # Remove the tokens appended at the front of query, which may belong to last query and answer.
            # query_tokens = query_tokens[-max_query_length:]
            query_tokens = ["[CLS]"] + query_tokens + ["[SEP]"]
            ques_input_ids = tokenizer.convert_tokens_to_ids(query_tokens)
            ques_input_mask = [1] * len(ques_input_ids)
            assert len(ques_input_ids) <= max_query_length
            while len(ques_input_ids) < max_query_length:
                ques_input_ids.append(0)
                ques_input_mask.append(0)
            assert len(ques_input_ids) == max_query_length
            assert len(ques_input_mask) == max_query_length

            doc_sen_tokens = example.doc_tokens
            all_doc_tokens = []
            for sentence in doc_sen_tokens:
                cur_sen_doc_tokens = ["[CLS]"]
                for token in sentence:
                    sub_tokens = tokenizer.tokenize(token)
                    if len(cur_sen_doc_tokens) + 1 + len(sub_tokens) > max_seq_length:
                        drop += 1
                        break
                    cur_sen_doc_tokens.extend(sub_tokens)
                cur_sen_doc_tokens.append("[SEP]")
                all_doc_tokens.append(cur_sen_doc_tokens)

            pass_input_ids = []
            pass_input_mask = []
            for sentence in all_doc_tokens:
                sentence_input_ids = tokenizer.convert_tokens_to_ids(sentence)
                sentence_input_mask = [1] * len(sentence_input_ids)

                assert len(sentence_input_ids) <= max_seq_length, len(sentence_input_ids)

                while len(sentence_input_ids) < max_seq_length:
                    sentence_input_ids.append(0)
                    sentence_input_mask.append(0)
                assert len(sentence_input_ids) == max_seq_length
                assert len(sentence_input_mask) == max_seq_length

                pass_input_ids.append(sentence_input_ids)
                pass_input_mask.append(sentence_input_mask)

            features.append(SingleSentenceFeature(qas_id=example.qas_id, unique_id=unique_id, example_index=example_index,
                                                  tokens=all_doc_tokens,
                                                  ques_input_ids=ques_input_ids, ques_input_mask=ques_input_mask,
                                                  pass_input_ids=pass_input_ids, pass_input_mask=pass_input_mask,
                                                  is_impossible=example.is_impossible, sentence_id=example.sentence_id))
            unique_id += 1
        logger.info(f'Read {len(features)} features and trunk {drop} sentences')
        return features

    def write_predictions(self, all_examples, all_features, all_results: List[RawResultChoice], output_prediction_file=None,
                          null_score_diff_threshold=0.0):
        """Write final predictions to the json file and log-odds of null if needed."""
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
            # max_diff_feature_index = 0
            max_diff_yes_logit = 0
            max_diff_no_logit = 0
            # max_diff_null_logit = 0
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
                    # max_diff_feature_index = feature_index
                    max_diff_yes_logit = yes_logit
                    max_diff_no_logit = no_logit
                    # max_diff_null_logit = null_logit

            # if max_diff > null_score_diff_threshold:
            #     final_text = 'unknown'
            # Here we only consider questions with Yes/No as answers
            if max_diff_yes_logit > max_diff_no_logit:
                final_text = 'yes'
            else:
                final_text = 'no'
            all_predictions[example.qas_id] = final_text

            gold_label = 'yes' if example.is_impossible == 0 else 'no'
            self.yesno_cate.update(gold_label, final_text)

        output = []
        for prediction in all_predictions:
            keys = prediction.split('--')
            pred = dict()
            pred['id'] = keys[0]
            pred['turn_id'] = int(keys[1])
            pred['answer'] = all_predictions[prediction]
            output.append(pred)

        logger.info('Yes/No Metric: %s' % self.yesno_cate)

        if output_prediction_file is not None:
            with open(output_prediction_file, 'w') as f:
                json.dump(output, f, indent=2)
        return output

    @staticmethod
    def data_to_tensors(all_features: List[SingleSentenceFeature]):
        ques_input_ids = torch.tensor([f.ques_input_ids for f in all_features], dtype=torch.long)
        ques_input_mask = torch.tensor([f.ques_input_mask for f in all_features], dtype=torch.long)

        max_sentence_num = 0
        for feature in all_features:
            max_sentence_num = max(max_sentence_num, len(feature.pass_input_ids))
        pass_input_ids = torch.LongTensor(len(all_features), max_sentence_num, len(all_features[0].pass_input_ids[0])).fill_(0)
        pass_input_mask = torch.LongTensor(len(all_features), max_sentence_num, len(all_features[0].pass_input_ids[0])).fill_(0)
        for feature_id, feature in enumerate(all_features):
            sentence_num = len(feature.pass_input_ids)
            pass_input_ids[feature_id, :sentence_num] = torch.LongTensor(feature.pass_input_ids)
            pass_input_mask[feature_id, :sentence_num] = torch.LongTensor(feature.pass_input_mask)

        all_answer_choices = torch.LongTensor([f.is_impossible for f in all_features])
        all_sentence_ids = torch.LongTensor([f.sentence_id for f in all_features])
        all_feature_index = torch.arange(ques_input_ids.size(0), dtype=torch.long)

        return ques_input_ids, ques_input_mask, pass_input_ids, pass_input_mask, \
            all_answer_choices, all_sentence_ids, all_feature_index

    @staticmethod
    def generate_inputs(batch: Tuple, all_features: List[SingleSentenceFeature], model_state):
        assert model_state in ModelState

        inputs = {
            "ques_input_ids": batch[0],
            "ques_input_mask": batch[1],
            "pass_input_ids": batch[2],
            "pass_input_mask": batch[3]
        }

        if model_state == ModelState.Test:
            return inputs
        elif model_state == ModelState.Train or model_state == ModelState.Evaluate:
            inputs['answer_choice'] = batch[4]
            inputs['sentence_ids'] = batch[5]

        return inputs
