import collections
import json
import logging
from typing import List, Tuple

import nltk
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from tqdm import tqdm

from data.data_instance import QAFullExample, QAFullInputFeatures, RawResultChoice, WeightResultChoice
from general_util import utils

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

"""
This reader has several usages:
    - Read simple CoQA Yes/No data ** with sentence supervision **.
    - ~~Mask or re-write new sentence id supervision as you need.~~
    - The reader will add [SEP] or other separator between each sentence in passage.
"""


class CoQAYesNoReader(object):
    def __init__(self):
        super(CoQAYesNoReader, self).__init__()
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.yesno_cate = utils.CategoricalAccuracy(['yes', 'no'])

    def read(self, input_file, dialog_turns: int = 2):
        """
        :param input_file: input file to load data. The format is in CoQA style
        :param dialog_turns:  Decide how many turns' questions and answers will be appended before current question.
        """
        logger.info('Reading data set from {}...'.format(input_file))
        logger.info('Read parameters:')
        logger.info('Dialog turns: {}'.format(dialog_turns))
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
                    doc_tokens=doc_tokens,
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
    def generate_features_sentence_ids(features: List[QAFullInputFeatures], sentence_id_input_file):
        logger.info('Reading sentence ids from {}'.format(sentence_id_input_file))
        with open(sentence_id_input_file, 'r') as f:
            id_dict = json.load(f)
        labeled_data = 0
        for feature in features:
            qas_id = feature.qas_id
            doc_span_index = feature.doc_span_index
            if qas_id in id_dict and doc_span_index == id_dict[qas_id]['doc_span_index']:
                feature.sentence_id = id_dict[qas_id]['sentence_id']
                if id_dict[qas_id]['sentence_id'] != -1:
                    labeled_data += 1
            else:
                feature.sentence_id = -1
        logger.info('Labeling {} data in total.'.format(labeled_data))
        return features

    @staticmethod
    def mask_all_sentence_ids(features: List[QAFullInputFeatures]):
        logger.info('Mask all sentence_ids...')
        for feature in features:
            feature.sentence_id = -1
        return features

    @staticmethod
    def convert_examples_to_features(examples: List[QAFullExample], tokenizer: BertTokenizer, max_seq_length, doc_stride,
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

            # Add [SEP] between sentences in doc
            doc_tokens, separator_poss, sentence_spans_list = utils.add_sentence_separator(example.doc_tokens,
                                                                                           example.sentence_span_list, '[SEP]')
            example.doc_tokens = doc_tokens
            example.sentence_span_list = sentence_spans_list

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

            separator_positions = [orig_to_tok_index[pos] for pos in separator_poss]
            for pos in separator_positions:
                assert all_doc_tokens[pos] == '[SEP]'

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

                # process separator positions
                doc_end = doc_start + doc_span.length - 1
                span_separators = [doc_offset - 1]
                for pos in separator_positions:
                    if doc_start <= pos <= doc_end:
                        new_offset = pos - doc_start + doc_offset
                        if new_offset == doc_offset:
                            continue
                        span_separators.append(new_offset)
                    else:
                        break

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

                # Process sentence id
                span_sen_id = -1
                for piece_sen_id, sen_id in enumerate(sentence_ids_list[doc_span_index]):
                    if ini_sen_id == sen_id:
                        span_sen_id = piece_sen_id
                if span_sen_id == -1:
                    answer_choice = 0
                else:
                    answer_choice = example.is_impossible + 1

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
                    ral_start_position=None,
                    ral_end_position=None,
                    meta_data={'separator_position': span_separators}))

                unique_id += 1

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
    def predict_sentence_ids(all_examples, all_features: List[QAFullInputFeatures], all_results: List[WeightResultChoice],
                             output_prediction_file=None,
                             weight_threshold: float = 0.0, only_correct: bool = False, label_threshold: float = 0.0):
        """
        :param all_results:
        :param all_examples:
        :param all_features:
        :param output_prediction_file:
        :param weight_threshold: The threshold for attention weights, only id predictions with a higher weight than this will
               be added.
        :param only_correct: If true, only id predictions with final choices predicted which are correct will be added.
               Otherwise, all the id predictions will be added no matter if the yes/no prediction is correct.
        :param label_threshold: Only make sense while only_correct=True, which means that only the id predictions with true
               yes/no prediction and the probability for yes/no is higher than this will be added.
        :return:
        """
        logger.info("Predicting sentence id to: %s" % output_prediction_file)
        logger.info("Weight threshold: {}".format(weight_threshold))
        logger.info("Use ids with true yes/no prediction only: {}".format(only_correct))
        logger.info("Yes/No prediction probability threshold : {}, only make sense while only_correct=True.".format(label_threshold))

        example_index_to_features = collections.defaultdict(list)
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        all_predictions = collections.OrderedDict()

        no_label = 0
        total = 0
        correct = 0
        for (example_index, example) in enumerate(all_examples):
            features = example_index_to_features[example_index]

            max_diff = -1000000
            max_diff_feature_index = 0
            max_yes_logit = 0
            max_no_logit = 0
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                choice_logits = result.choice_logits
                non_null_logit = choice_logits[1] + choice_logits[2]
                null_logit = choice_logits[0]
                diff = non_null_logit - null_logit
                if diff > max_diff:
                    max_diff = diff
                    max_diff_feature_index = feature_index
                    max_yes_logit = choice_logits[1]
                    max_no_logit = choice_logits[2]

            # if max_diff > null_score_diff_threshold:
            #     final_text = 'unknown'
            # Here we only consider questions with Yes/No as answers
            target_feature = features[max_diff_feature_index]
            sentence_id = unique_id_to_result[target_feature.unique_id].max_weight_index
            # Attention weights threshold
            max_weight = unique_id_to_result[target_feature.unique_id].max_weight
            if max_weight <= weight_threshold:
                sentence_id = -1

            # Yes/No prediction restriction
            yesno_scores = utils.compute_softmax([max_yes_logit, max_no_logit])
            if max_yes_logit > max_no_logit:
                final_choice = 0
                prediction_prob = yesno_scores[0]
            else:
                final_choice = 1
                prediction_prob = yesno_scores[1]
            # if final_choice != example.is_impossible:
            #     sentence_id = -1
            if only_correct:
                if (final_choice + 1) != target_feature.is_impossible:
                    sentence_id = -1
                if prediction_prob <= label_threshold:
                    sentence_id = -1

            feature_doc_span_index = target_feature.doc_span_index
            all_predictions[example.qas_id] = {'sentence_id': sentence_id,
                                               'doc_span_index': feature_doc_span_index,
                                               'weight': max_weight,
                                               'choice_prediction': 'yes' if final_choice == 0 else 'no',
                                               'prediction_prob': prediction_prob}

            total += 1
            if sentence_id == -1:
                no_label += 1
            else:
                if sentence_id == target_feature.sentence_id:
                    correct += 1

        logger.info("Labeling {} instances of {} in total".format(total - no_label, len(all_examples)))
        logger.info(
            "Labeling accuracy: Correct / Total: {} / {},  {}".format(correct, total - no_label, correct * 1.0 / (total - no_label)))

        if output_prediction_file is not None:
            with open(output_prediction_file, 'w') as f:
                json.dump(all_predictions, f, indent=2)
        return all_predictions

    @staticmethod
    def data_to_tensors(all_features: List[QAFullInputFeatures]):

        all_input_ids = torch.tensor([f.input_ids for f in all_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in all_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in all_features], dtype=torch.long)
        all_answer_choices = torch.tensor([f.is_impossible for f in all_features], dtype=torch.long)
        all_sentence_ids = torch.tensor([f.sentence_id for f in all_features], dtype=torch.long)
        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        return all_input_ids, all_segment_ids, all_input_mask, all_answer_choices, all_sentence_ids, all_feature_index

    @staticmethod
    def generate_inputs(batch: Tuple, all_features: List[QAFullInputFeatures], do_label: bool, is_training: bool):
        feature_index = batch[5].tolist()
        sentence_span_list = [all_features[idx].sentence_span_list for idx in feature_index]
        inputs = {
            "input_ids": batch[0],
            "token_type_ids": batch[1],
            "attention_mask": batch[2],
            "answer_choice": batch[3],
            "sentence_span_list": sentence_span_list
        }
        if do_label and not is_training:
            return inputs
        inputs['sentence_ids'] = batch[4]

        return inputs
