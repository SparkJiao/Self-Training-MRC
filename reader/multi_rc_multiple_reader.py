import collections
import json
from typing import List, Tuple

import nltk
import numpy as np
import torch
from tqdm import tqdm

from data.data_instance import MultiRCExample, MultiRCFeature, RawResultChoice, RawOutput, ModelState
from general_util import utils
from general_util.multi_rc.measure import Measures
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)

"""
TODO:
- Add sliding window while reading features while evaluation or testing.
"""


class MultiRCMultipleReader(object):
    def __init__(self):
        super(MultiRCMultipleReader, self).__init__()
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
            sentences = instance['article']
            article_id = instance['id']

            sentence_start_list, sentence_end_list = [], []
            passage = ''
            for sentence in sentences:
                sentence = sentence.strip()
                sentence_start_list.append(len(passage))
                passage = passage + sentence + ' '
                sentence_end_list.append(len(passage) - 1)
                assert sentence_start_list[-1] <= sentence_end_list[-1]
                # if len(examples) == 453:
                #     print(sentence)
                #     print(sentence_start_list[-1], sentence_end_list[-1])

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
            # sentence_start_list, sentence_end_list = utils.split_sentence(passage, self.sentence_tokenizer)
            sentence_span_list = []
            for c_start, c_end in zip(sentence_start_list, sentence_end_list):
                t_start = char_to_word_offset[c_start]
                t_end = char_to_word_offset[c_end]
                sentence_span_list.append((t_start, t_end))
                assert t_start <= t_end

            questions = instance['questions']
            answers = instance['answers']
            options = instance['options']
            evidences = instance['evidences']
            # multi_sents = instance['multi_sents']

            for q_id, (question, option_list, option_answer_list, evidence) in enumerate(zip(questions, options, answers, evidences)):
                # qas_id = f"{article_id}--{q_id}"
                for op_id, (option, answer) in enumerate(zip(option_list, option_answer_list)):
                    qas_id = f"{article_id}--{q_id}--{op_id}"
                    example = MultiRCExample(
                        qas_id=qas_id,
                        doc_tokens=doc_tokens,
                        question_text=question,
                        sentence_span_list=sentence_span_list,
                        option_text=option,
                        answer=answer,
                        sentence_ids=evidence
                    )
                    examples.append(example)

        logger.info('Finish reading {} examples from {}'.format(len(examples), input_file))
        return examples

    @staticmethod
    def convert_examples_to_features(examples: List[MultiRCExample], tokenizer, max_seq_length: int = 512):
        unique_id = 1000000000
        features = []
        for (example_index, example) in tqdm(enumerate(examples), desc='Convert examples to features', total=len(examples)):
            query_tokens = tokenizer.tokenize(example.question_text)
            if query_tokens[-1] != '?':
                query_tokens.append('?')

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

            # Process all tokens
            q_op_tokens = query_tokens + tokenizer.tokenize(example.option_text)
            doc_tokens = all_doc_tokens[:]
            utils.truncate_seq_pair(q_op_tokens, doc_tokens, max_seq_length - 3)

            tokens = ["[CLS]"] + q_op_tokens + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            tokens = tokens + doc_tokens + ["[SEP]"]
            segment_ids += [1] * (len(doc_tokens) + 1)

            sentence_list = []
            collected_sentence_indices = []
            doc_offset = len(q_op_tokens) + 2
            for sentence_index, (start, end) in enumerate(sentence_spans):
                assert start <= end, (example_index, sentence_index, start, end)
                if start >= len(doc_tokens):
                    break
                if end >= len(doc_tokens):
                    end = len(doc_tokens) - 1
                start = doc_offset + start
                end = doc_offset + end
                sentence_list.append((start, end))
                assert start < max_seq_length and end < max_seq_length
                collected_sentence_indices.append(sentence_index)

            sentence_ids = []
            for sentence_id in example.sentence_ids:
                if sentence_id in collected_sentence_indices:
                    sentence_ids.append(sentence_id)

            # For multiple style, append 0 at last and for each sentence id, +1
            sentence_ids = [x + 1 for x in sentence_ids]
            sentence_ids.append(0)

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

            features.append(MultiRCFeature(
                example_index=example_index,
                qas_id=example.qas_id,
                unique_id=unique_id,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                sentence_span_list=sentence_list,
                answer=example.answer + 1,  # In bert_hierarchical model, the output size is 3.
                sentence_ids=sentence_ids
            ))
            unique_id += 1

        logger.info(f'Reading {len(features)} features.')

        return features

    def write_predictions(self, all_examples: List[MultiRCExample], all_features: List[MultiRCFeature],
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
            choice_logits = np.array(result.choice_logits[1:])
            choice_logits = choice_logits / choice_logits.sum()  # Recalculate probability after remove index 0.
            prediction = int(choice_logits.argmax())
            all_predictions[example.qas_id] = {
                'prediction': prediction,
                'pred_prob': choice_logits[prediction]
            }
            if example.answer is not None:
                all_predictions[example.qas_id]['answer'] = example.answer

        if output_prediction_file is not None:
            with open(output_prediction_file, 'w') as f:
                json.dump(all_predictions, f, indent=2)
        metric, save_metric = self.get_metric(all_examples, all_predictions)
        return all_predictions, metric, save_metric

    @staticmethod
    def get_metric(examples: List[MultiRCExample], all_predictions):
        golden = collections.defaultdict(lambda: collections.defaultdict(dict))
        for example in examples:
            article_id, question_id, option_id = example.qas_id.split('--')
            golden[article_id][question_id][int(option_id)] = example.answer

        predictions = collections.defaultdict(lambda: collections.defaultdict(dict))
        for pred_id, prediction in all_predictions.items():
            article_id, question_id, option_id = pred_id.split('--')
            predictions[article_id][question_id][option_id] = prediction['prediction']

        metric = {
            'f1_m': Measures.per_question_metrics(golden, predictions)[-1],
            'f1_a': Measures.per_dataset_metric(golden, predictions)[-1],
            'em0': Measures.exact_match_metrics(golden, predictions, 0),
            'em1': Measures.exact_match_metrics(golden, predictions, 1)
        }
        save_metric = ('f1_a', metric['f1_a'])
        return metric, save_metric

    @staticmethod
    def generate_features_sentence_ids(features: List[MultiRCFeature], sentence_label_file):
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
                # sentence_ids = sentence_label_dict[qas_id]['sentence_ids']
                # for option_index, option_att in enumerate(feature.choice_features):
                #     feature.choice_features[option_index]["sentence_ids"] = sentence_ids[option_index]
                #     if feature.choice_features[option_index]["sentence_ids"]:
                #         labeled_data += 1
                sentence_ids = sentence_label_dict[qas_id]['sentence_ids']
                if sentence_ids:
                    labeled_data += 1
                feature.sentence_ids = sentence_ids
            else:
                feature.sentence_ids = []

        logger.info(f"Labeled {labeled_data} data in total.")
        return features

    @staticmethod
    def predict_sentence_ids(all_examples, all_features: List[MultiRCFeature], all_results: List[RawOutput],
                             output_prediction_file=None,
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
        # weight_cnt = collections.Counter()
        for (example_index, example) in enumerate(all_examples):
            total += 1
            feature = example_index_to_features[example_index][0]

            model_output = unique_id_to_result[feature.unique_id].model_output
            choice_output = np.array(model_output["choice_logits"][1:])
            choice_output = choice_output / choice_output.sum()
            choice_prediction = int(choice_output.argmax())
            choice_prob = float(choice_output.max())

            evidence = model_output["evidence"]
            if (only_correct and choice_prediction == example.answer and choice_prob > label_threshold) or not only_correct:
                if evidence['value'] > weight_threshold:
                    sentence_ids = evidence['sentences']
                    weight = evidence['value']
                    labeled += 1
                else:
                    sentence_ids = []
                    weight = 0.0
            else:
                sentence_ids = []
                weight = 0.0

            all_predictions[example.qas_id] = {
                'choice_prob': choice_prob,
                'choice_prediction': choice_prediction,
                'weight': weight,
                'sentence_ids': sentence_ids,
                'answer': example.answer
            }

        logger.info("Labeling {} instances of {} in total".format(labeled, len(all_examples)))
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        if output_prediction_file is not None:
            with open(output_prediction_file, 'w') as f:
                json.dump(all_predictions, f, indent=2)
        return all_predictions

    def data_to_tensors(self, all_features: List[MultiRCFeature]):

        max_sentence_num = max(map(lambda x: len(x.sentence_span_list), all_features))
        # for feature in all_features:
        #     choice_features = feature.choice_features
        #     max_sentence_num = max(max_sentence_num, max(map(lambda x: len(x["sentence_span_list"]), choice_features)))
        self.data_state_dict['max_sentences'] = max_sentence_num

        all_input_ids = torch.LongTensor([feature.input_ids for feature in all_features])
        all_input_mask = torch.LongTensor([feature.input_mask for feature in all_features])
        all_segment_ids = torch.LongTensor([feature.segment_ids for feature in all_features])
        all_answers = torch.LongTensor([feature.answer for feature in all_features])

        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        return all_input_ids, all_segment_ids, all_input_mask, all_answers, all_feature_index

    def generate_inputs(self, batch: Tuple, all_features: List[MultiRCFeature], model_state):
        assert model_state in ModelState
        feature_index = batch[-1].tolist()
        batch_features = [all_features[index] for index in feature_index]
        sentence_span_list = []
        sentence_ids = []
        # For convenience
        for feature in batch_features:
            sentence_span_list.append(feature.sentence_span_list)
            sentence_ids.append(feature.sentence_ids)
        # assert len(sentence_span_list) == len(batch_features) * 4

        inputs = {
            "input_ids": batch[0],
            "token_type_ids": batch[1],
            "attention_mask": batch[2],
            "sentence_span_list": sentence_span_list
            # "max_sentences": self.data_state_dict['max_sentences']
        }
        if model_state == ModelState.Test:
            return inputs
        elif model_state == ModelState.Evaluate:
            inputs["answer_choice"] = batch[3]
        elif model_state == ModelState.Train:
            inputs["answer_choice"] = batch[3]
            inputs["sentence_ids"] = sentence_ids

        return inputs
