import collections
import json
from typing import List, Tuple

import nltk
import numpy as np
import torch
from tqdm import tqdm

from data.data_instance import MultiChoiceFullExample, MultiChoiceFullFeature, RawResultChoice, RawOutput, ModelState
from general_util import utils
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)

"""
TODO:
- Add sliding window while reading features while evaluation or testing.
"""


class RACEReader(object):
    def __init__(self):
        super(RACEReader, self).__init__()
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.data_state_dict = dict()

    def read(self, input_file):
        """Read a SQuAD json file into a list of SquadExample."""
        logger.info('Reading data set from {}...'.format(input_file))
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)

        def is_whitespace(ch):
            if ch == " " or ch == "\t" or ch == "\r" or ch == "\n" or ord(ch) == 0x202F:
                return True
            return False

        examples = []
        for instance in tqdm(input_data):
            passage = instance['article']
            article_id = instance['id']

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

            questions = instance['questions']
            answers = list(map(lambda x: {'A': 0, 'B': 1, 'C': 2, 'D': 3}[x], instance['answers']))
            options = instance['options']

            for q_id, (question, answer, option_list) in enumerate(zip(questions, answers, options)):
                qas_id = f"{article_id}--{q_id}"
                example = MultiChoiceFullExample(
                    qas_id=qas_id,
                    question_text=question,
                    options=option_list,
                    doc_tokens=doc_tokens,
                    sentence_span_list=sentence_span_list,
                    answer=answer
                )
                examples.append(example)

        logger.info('Finish reading {} examples from {}'.format(len(examples), input_file))
        return examples

    @staticmethod
    def convert_examples_to_features(examples: List[MultiChoiceFullExample], tokenizer, max_seq_length: int = 512):
        unique_id = 1000000000
        features = []
        for (example_index, example) in tqdm(enumerate(examples), desc='Convert examples to features', total=len(examples)):
            query_tokens = tokenizer.tokenize(example.question_text)

            # word piece index -> token index
            tok_to_orig_index = []
            # token index -> word pieces group start index
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

            options = example.options
            choice_features = []
            for option_id, option in enumerate(options):
                q_op_tokens = query_tokens + tokenizer.tokenize(option)
                doc_tokens = all_doc_tokens[:]
                utils.truncate_seq_pair(q_op_tokens, doc_tokens, max_seq_length - 3)

                tokens = ["[CLS]"] + q_op_tokens + ["[SEP]"]
                segment_ids = [0] * len(tokens)
                tokens = tokens + doc_tokens + ["[SEP]"]
                segment_ids += [1] * (len(doc_tokens) + 1)

                sentence_list = []
                doc_offset = len(q_op_tokens) + 2
                for (start, end) in sentence_spans:
                    assert start <= end
                    if start >= len(doc_tokens):
                        break
                    if end >= len(doc_tokens):
                        end = len(doc_tokens) - 1
                    start = doc_offset + start
                    end = doc_offset + end
                    sentence_list.append((start, end))
                    assert start < max_seq_length and end < max_seq_length

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                choice_features.append({
                    "input_ids": input_ids,
                    "input_mask": input_mask,
                    "segment_ids": segment_ids,
                    "sentence_span_list": sentence_list
                })
            features.append(MultiChoiceFullFeature(
                example_index=example_index,
                qas_id=example.qas_id,
                unique_id=unique_id,
                choice_features=choice_features,
                answer=example.answer
            ))
            unique_id += 1

        logger.info(f'Reading {len(features)} features.')

        return features

    @staticmethod
    def write_predictions(all_examples: List[MultiChoiceFullExample], all_features: List[MultiChoiceFullFeature],
                          all_results: List[RawResultChoice], output_prediction_file=None):
        logger.info("Writing predictions to: %s" % output_prediction_file)

        example_index_to_features = collections.defaultdict(list)
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        all_predictions = collections.OrderedDict()

        for (example_index, example) in enumerate(all_examples):
            feature = example_index_to_features[example_index][0]
            result = unique_id_to_result[feature.unique_id]
            choice_logits = result.choice_logits
            prediction = int(np.array(choice_logits).argmax())
            all_predictions[example.qas_id] = {
                'prediction': prediction,
                'pred_prob': choice_logits[prediction]
            }
            if example.answer is not None:
                all_predictions[example.qas_id]['answer'] = example.answer

        if output_prediction_file is not None:
            with open(output_prediction_file, 'w') as f:
                json.dump(all_predictions, f, indent=2)
        return all_predictions

    @staticmethod
    def generate_features_sentence_ids(features: List[MultiChoiceFullFeature], sentence_label_file):
        """
        The sentence label file may contains two attribute. The first is a hard label, which indicates weather a sentence
        is an evidence sentence(0/1/-1), which could be optimized by nll_loss. The other is the probability distribution over
        all sentences, which could be optimized by KL divergence.
        For multiple evidence sentences, the labels will be generated by sigmoid. For single evidence sentence, the label
        will be generated by softmax, so the input of sentence labels will be a one-hot vector.
        For KL divergence seems similar between single and multiple. And because the KLDivLoss doesn't contain 'ignore_index',
        the sentence_label_file need to mark the unlabeled parts as -1 so you can generate a zero mask.
        :param features:
        :param sentence_label_file:
        :return:
        """
        logger.info('Reading sentence labels from {}'.format(sentence_label_file))
        labeled_data = 0

        with open(sentence_label_file, 'r') as f:
            sentence_label_dict = json.load(f)

        for feature_id, feature in enumerate(features):
            # If some option or question doesn't contain labels, you still need to fill it with -1
            qas_id = feature.qas_id
            if qas_id in sentence_label_dict:
                # labeled_data += 1
                sentence_label = sentence_label_dict[qas_id]["sentence_label"]
                sentence_prob = sentence_label_dict[qas_id]["sentence_prob"]
                sentence_id = sentence_label_dict[qas_id]["max_weight_index"]
                flag = False
                for option_index, option_att in enumerate(feature.choice_features):
                    if sentence_label[option_index] is not None:
                        option_att["sentence_label"] = sentence_label[option_index]
                        flag = True
                    if sentence_prob[option_index] is not None:
                        option_att["sentence_prob"] = sentence_prob[option_index]
                        flag = True
                    if sentence_id[option_index] is not None:
                        option_att["sentence_id"] = sentence_id[option_index]
                        flag = True
                    if flag:
                        labeled_data += 1
                        flag = False
            # else:
            #     for option_index, option_att in enumerate(feature.choice_features):
            #         option_att["sentence_label"] = []
            #         option_att["sentence_prob"] = []
            #         option_att["sentence_id"] = []

        logger.info(f"Labeled {labeled_data} data in total.")
        return features

    @staticmethod
    def predict_sentence_ids(all_examples, all_features: List[MultiChoiceFullFeature], all_results: List[RawOutput],
                             output_prediction_file=None, multi_evidence: bool = False,
                             weight_threshold: float = 0.0, only_correct: bool = True, label_threshold: float = 0.0):
        logger.info("Predicting sentence id to: %s" % output_prediction_file)
        logger.info("Weight threshold: {}".format(weight_threshold))
        logger.info("Use ids with true yes/no prediction only: {}".format(only_correct))
        logger.info("Evidence prediction probability threshold : {}, only make sense while only_correct=True.".format(label_threshold))

        example_index_to_features = collections.defaultdict(list)
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        all_predictions = collections.OrderedDict()

        total = 0
        labeled = 0
        for (example_index, example) in enumerate(all_examples):
            total += 1
            feature = example_index_to_features[example_index][0]

            model_output = unique_id_to_result[feature.unique_id].model_output
            choice_output = model_output["choice_logits"].numpy()
            choice_prediction = int(choice_output.argmax())
            choice_prob = float(choice_output.max())

            sentence_logits = model_output["sentence_logits"]  # option_num * max_sentence_num

            if (only_correct and choice_prediction == example.answer and choice_prob > label_threshold) or not only_correct:
                weight_mask = sentence_logits > weight_threshold
                # Only keep selected values as the same as before, the other values are set to 0.
                masked_sentence_logits = sentence_logits.masked_fill(1 - weight_mask, 0)

                max_weight, max_weight_index = masked_sentence_logits.max(dim=-1)
                max_weight = max_weight.float().tolist()
                max_weight_index = max_weight_index.int().tolist()

                sentence_labels = torch.zeros_like(sentence_logits).fill_(-1)
                # if multi_evidence:
                sentence_labels = sentence_labels.masked_fill(weight_mask, 1)
                # else:
                for op_index, op_w in enumerate(max_weight):
                    if op_w == 0:
                        max_weight_index[op_index] = -1
                    else:
                        sentence_labels[op_index][max_weight_index[op_index]] = 1
                # labeled += 1
                # if weight_mask.float().sum() > 0:
                #     labeled += 1
            else:
                masked_sentence_logits = torch.zeros_like(sentence_logits).fill_(0)
                sentence_labels = torch.zeros_like(sentence_logits).fill_(-1)
                max_weight = [0.] * sentence_labels.size(0)
                max_weight_index = [-1] * sentence_labels.size(0)

            for option_index, sentence_id in enumerate(max_weight_index):
                assert sentence_id == -1 or sentence_id < len(feature.choice_features[option_index]['sentence_span_list'])
                if sentence_id != -1:
                    labeled += 1

            all_predictions[example.qas_id] = {
                'choice_prediction': choice_prediction,
                'choice_prob': choice_prob,
                'answer': example.answer,
                'raw_sentence_prob': sentence_logits.tolist(),
                'sentence_prob': masked_sentence_logits.tolist(),
                'sentence_label': sentence_labels.tolist(),
                'sentence_prob_sum': masked_sentence_logits.sum(dim=-1).tolist(),
                'max_weight': max_weight,
                'max_weight_index': max_weight_index
            }

        logger.info("Labeling {} instances of {} in total".format(labeled, len(all_examples)))

        if output_prediction_file is not None:
            with open(output_prediction_file, 'w') as f:
                json.dump(all_predictions, f, indent=2)
        return all_predictions

    def data_to_tensors(self, all_features: List[MultiChoiceFullFeature]):

        max_sentence_num = 0
        for feature in all_features:
            choice_features = feature.choice_features
            max_sentence_num = max(max_sentence_num, max(map(lambda x: len(x["sentence_span_list"]), choice_features)))
        self.data_state_dict['max_sentences'] = max_sentence_num

        all_input_ids = torch.LongTensor([[choice["input_ids"] for choice in feature.choice_features] for feature in all_features])
        all_input_mask = torch.LongTensor([[choice["input_mask"] for choice in feature.choice_features] for feature in all_features])
        all_segment_ids = torch.LongTensor([[choice["segment_ids"] for choice in feature.choice_features] for feature in all_features])
        all_answers = torch.LongTensor([feature.answer for feature in all_features])

        all_sentence_labels = torch.zeros(all_input_ids.size(0), all_input_ids.size(1), max_sentence_num).fill_(-1).long()
        all_sentence_prob = torch.zeros_like(all_sentence_labels).float()
        all_sentence_ids = torch.zeros(all_input_ids.size(0), all_input_ids.size(1)).fill_(-1).long()
        for feature_index, feature in enumerate(all_features):
            for option_index, option in enumerate(feature.choice_features):
                if "sentence_label" in option and option["sentence_label"] is not None:
                    all_sentence_labels[feature_index][option_index] = torch.LongTensor(option["sentence_label"])
                if "sentence_prob" in option and option["sentence_prob"] is not None:
                    all_sentence_prob[feature_index][option_index] = torch.FloatTensor(option["sentence_prob"])
                if "sentence_id" in option:
                    all_sentence_ids[feature_index][option_index] = option["sentence_id"]
        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        return all_input_ids, all_segment_ids, all_input_mask, all_answers, \
               all_sentence_labels, all_sentence_prob, all_sentence_ids, all_feature_index

    def generate_inputs(self, batch: Tuple, all_features: List[MultiChoiceFullFeature], model_state):
        assert model_state in ModelState
        feature_index = batch[-1].tolist()
        batch_features = [all_features[index] for index in feature_index]
        sentence_span_list = []
        # For convenience
        for feature in batch_features:
            sentence_span_list.extend([choice["sentence_span_list"] for choice in feature.choice_features])
        assert len(sentence_span_list) == len(batch_features) * 4

        inputs = {
            "input_ids": batch[0],
            "token_type_ids": batch[1],
            "attention_mask": batch[2],
            "sentence_span_list": sentence_span_list,
            "max_sentences": self.data_state_dict['max_sentences']
        }
        if model_state == ModelState.Test:
            return inputs
        elif model_state == ModelState.Evaluate:
            inputs["labels"] = batch[3]
        elif model_state == ModelState.Train:
            inputs["labels"] = batch[3]
            inputs["sentence_labels"] = batch[4]
            inputs["sentence_prob"] = batch[5]
            inputs["sentence_ids"] = batch[6]

        return inputs
