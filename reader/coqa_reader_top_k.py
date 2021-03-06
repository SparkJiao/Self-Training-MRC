import collections
import json
import logging
from typing import List, Tuple

import nltk
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from tqdm import tqdm

from data.data_instance import SQuADFullExample, SQuADFullInputFeatures, RawResultChoice, WeightResultChoice, WeightResult, \
    QAFullInputFeatures, QAFullExample, ModelState, RawOutput
from general_util import utils
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)

"""
This reader has several usages:
    - Read simple CoQA Yes/No data ** with sentence supervision **.
    - Mask or re-write new sentence id supervision as you need.
"""


class CoQATopKReader(object):
    def __init__(self):
        super(CoQATopKReader, self).__init__()
        self.yesno_cate = utils.CategoricalAccuracy(['yes', 'no'])
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def read(self, input_file, dialog_turns: int = 2, remove_evidence=False, remove_question=False, remove_passage=False,
             remove_dict=None):
        """
        :param input_file: input file to load data. The format is in CoQA style
        :param dialog_turns:  Decide how many turns' questions and answers will be appended before current question.
        """
        logger.info('Reading data set from {}...'.format(input_file))
        logger.info('Read parameters:')
        logger.info('Dialog turns: {}'.format(dialog_turns))
        logger.info('Remove evidence during test: {}'.format(remove_evidence))
        logger.info('Remove dict: {}'.format(remove_dict))
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)['data']

        def is_whitespace(ch):
            if ch == " " or ch == "\t" or ch == "\r" or ch == "\n" or ord(ch) == 0x202F:
                return True
            return False

        if remove_dict is None:
            remove_dict = {}
        else:
            remove_dict = json.load(open(remove_dict, 'r'))
            logger.info(len(remove_dict))

        rule_labels_acc = utils.AverageMeter()

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

                if story_id in remove_dict and str(i + 1) in remove_dict[story_id]:
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

                # Remove evidence sentence for experiments while evaluation only.
                if remove_evidence and sentence_id != -1 and 'train' not in input_file:
                    evi_token_s, evi_token_e = sentence_span_list[sentence_id]
                    new_doc_tokens = doc_tokens[:evi_token_s] + doc_tokens[(evi_token_e + 1):]
                    rationale_start_position = rationale_end_position = -1
                    reduce_offset = evi_token_e - evi_token_s + 1
                    new_sentence_span_list = sentence_span_list[:sentence_id] + [(s - reduce_offset, e - reduce_offset)
                                                                                 for s, e in sentence_span_list[(sentence_id + 1):]]
                    sentence_id = -1
                else:
                    new_doc_tokens = doc_tokens
                    new_sentence_span_list = sentence_span_list


                if 'sentence_id' in question:
                    pseudo_sentence_id = question['sentence_id']
                    if pseudo_sentence_id == sentence_id:
                        rule_labels_acc.update(1)
                    else:
                        rule_labels_acc.update(0)
                    sentence_id = pseudo_sentence_id

                # example = SQuADFullExample(
                #     qas_id=qas_id,
                #     question_text=question_text,
                #     doc_tokens=doc_tokens,
                #     sentence_span_list=sentence_span_list,
                #     orig_answer_text="",
                #     start_position=None,
                #     end_position=None,
                #     sentence_id=sentence_id,
                #     is_impossible=answer_choice,
                #     ral_start_position=rationale_start_position,
                #     ral_end_position=rationale_end_position)
                example = SQuADFullExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=new_doc_tokens,
                    sentence_span_list=new_sentence_span_list,
                    orig_answer_text="",
                    start_position=None,
                    end_position=None,
                    sentence_id=sentence_id,
                    is_impossible=answer_choice,
                    ral_start_position=rationale_start_position,
                    ral_end_position=rationale_end_position)
                examples.append(example)

        if rule_labels_acc.count > 0:
            logger.info('Read labels generated by rules.')
            logger.info(f'Accuracy of labels: {rule_labels_acc.avg}')
        return examples

    @staticmethod
    def generate_features_sentence_ids(features: List[SQuADFullInputFeatures], sentence_id_input_file):
        logger.info('Reading sentence ids from {}'.format(sentence_id_input_file))
        with open(sentence_id_input_file, 'r') as f:
            id_dict = json.load(f)
        labeled_data = 0
        for feature in features:
            qas_id = feature.qas_id
            doc_span_index = feature.doc_span_index
            if qas_id in id_dict and doc_span_index == id_dict[qas_id]['doc_span_index']:
                feature.sentence_id = id_dict[qas_id]['sentence_id']
                if id_dict[qas_id]['sentence_id']:
                    labeled_data += 1
            else:
                feature.sentence_id = []
        logger.info('Labeling {} data in total.'.format(labeled_data))
        return features

    @staticmethod
    def mask_all_sentence_ids(features: List[SQuADFullInputFeatures]):
        logger.info('Warning: the method `mask_all_sentence_ids` is Deprecated.')
        logger.info('Mask all sentence_ids...')
        for feature in features:
            feature.sentence_id = []
        return features

    @staticmethod
    def convert_examples_to_features(examples: List[QAFullExample], tokenizer: BertTokenizer, max_seq_length, doc_stride,
                                     max_query_length):
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

            # Rationale start and end position in chunk, where is calculated from the start of current chunk.
            # ral_start_position = None
            # ral_end_position = None

            ral_start_position = orig_to_tok_index[example.ral_start_position]
            if example.ral_end_position < len(example.doc_tokens) - 1:
                ral_end_position = orig_to_tok_index[example.ral_end_position + 1] - 1
            else:
                ral_end_position = len(all_doc_tokens) - 1
            ral_start_position, ral_end_position = utils.improve_answer_span(
                all_doc_tokens, ral_start_position, ral_end_position, tokenizer,
                example.orig_answer_text)

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

                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1

                # Process rationale
                out_of_span = False
                if not (ral_start_position >= doc_start and ral_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    # TODO:
                    #  Considering how to set rationale start and end positions for out of span instances.
                    ral_start = 0
                    ral_end = 0
                    answer_choice = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    ral_start = ral_start_position - doc_start + doc_offset
                    ral_end = ral_end_position - doc_start + doc_offset
                    answer_choice = example.is_impossible + 1

                # Process sentence id
                span_sen_id = -1
                for piece_sen_id, sen_id in enumerate(sentence_ids_list[doc_span_index]):
                    if ini_sen_id == sen_id:
                        span_sen_id = piece_sen_id
                # # For no sentence id feature, replace it with []
                if span_sen_id == -1:
                    span_sen_id = []
                else:
                    span_sen_id = [span_sen_id]
                meta_data = {'span_sen_to_orig_sen_map': sentence_ids_list[doc_span_index]}

                if example_index < 0:
                    logger.info("*** Example ***")
                    logger.info("unique_id: %s" % unique_id)
                    logger.info("example_index: %s" % example_index)
                    logger.info("doc_span_index: %s" % doc_span_index)
                    logger.info(
                        "sentence_spans_list: %s" % " ".join(
                            [(str(x[0]) + '-' + str(x[1])) for x in cur_sentence_list]))

                    rationale_text = " ".join(tokens[ral_start:(ral_end + 1)])
                    logger.info("answer choice: %s" % str(answer_choice))
                    logger.info("rationale start position: %s" % str(ral_start))
                    logger.info("rationale end position: %s" % str(ral_end))
                    logger.info("rationale: %s" % rationale_text)

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
                    ral_start_position=ral_start,
                    ral_end_position=ral_end,
                    meta_data=meta_data))

                unique_id += 1

        return features

    def write_predictions(self, all_examples, all_features, all_results: List[RawResultChoice], output_prediction_file=None):
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
            max_diff_choice_logits = 0
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
                    max_diff_choice_logits = choice_logits

            # if max_diff > null_score_diff_threshold:
            #     final_text = 'unknown'
            # Here we only consider questions with Yes/No as answers
            if max_diff_yes_logit > max_diff_no_logit:
                final_text = 'yes'
            else:
                final_text = 'no'
            # all_predictions[example.qas_id] = final_text
            all_predictions[example.qas_id] = {
                'prediction': final_text,
                'gold_answer': example.is_impossible,
                'raw_choice_logits': max_diff_choice_logits
            }

            gold_label = 'yes' if example.is_impossible == 0 else 'no'
            self.yesno_cate.update(gold_label, final_text)

        output = []
        for prediction in all_predictions:
            keys = prediction.split('--')
            pred = dict()
            pred['id'] = keys[0]
            pred['turn_id'] = int(keys[1])
            pred['answer'] = all_predictions[prediction]['prediction']
            pred['gold_answer'] = all_predictions[prediction]['gold_answer']
            pred['raw_choice_logits'] = all_predictions[prediction]['raw_choice_logits']
            output.append(pred)

        logger.info('Yes/No Metric: %s' % self.yesno_cate)

        if output_prediction_file is not None:
            with open(output_prediction_file, 'w') as f:
                json.dump(output, f, indent=2)
        metric, save_metric = self.get_metric(all_examples, output)
        return output, metric, save_metric

    def get_metric(self, examples: List[QAFullExample], all_predictions):
        self.yesno_cate.reset()
        for prediction in all_predictions:
            if prediction['gold_answer'] == 0:
                gold = 'yes'
            elif prediction['gold_answer'] == 1:
                gold = 'no'
            else:
                raise RuntimeError(f'Wrong gold answer type: {prediction["gold_answer"]}')
            self.yesno_cate.update(gold, prediction['answer'])
        yes_metric = self.yesno_cate.f1_measure('yes', 'no')
        no_metric = self.yesno_cate.f1_measure('no', 'yes')
        metric = {
            'yes_f1': yes_metric['f1'],
            'yes_recall': yes_metric['recall'],
            'yes_precision': yes_metric['precision'],
            'no_f1': no_metric['f1'],
            'no_recall': no_metric['recall'],
            'no_precision': no_metric['precision'],
            'yesno_acc': yes_metric['accuracy']
        }
        return metric, ('yesno_acc', metric['yesno_acc'])

    @staticmethod
    def write_sentence_predictions(all_examples, all_features: List[QAFullInputFeatures], all_results: List[WeightResult],
                                   output_prediction_file=None, null_score_diff_threshold=0.0):
        logger.info("Writing evidence id predictions to: %s" % output_prediction_file)

        sentence_pred = utils.AverageMeter()
        yesno_pred = utils.AverageMeter()

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
            max_diff_feature_index = 0
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                choice_logits = result.choice_logits
                non_null_logit = choice_logits[1] + choice_logits[2]
                null_logit = choice_logits[0]
                diff = non_null_logit - null_logit
                if diff > max_diff:
                    max_diff = diff
                    max_diff_feature_index = feature_index

            target_feature = features[max_diff_feature_index]
            target_result = unique_id_to_result[target_feature.unique_id]
            sentence_id_pred = target_feature.meta_data['span_sen_to_orig_sen_map'][target_result.max_weight_index]

            target_choice_prob = utils.compute_softmax([target_result.choice_logits[1], target_result.choice_logits[2]])
            if target_choice_prob[0] > target_choice_prob[1]:
                target_choice = 0
            else:
                target_choice = 1

            if target_choice == example.is_impossible:
                yesno_pred.update(1, 1)
            else:
                yesno_pred.update(0, 1)

            if sentence_id_pred == example.sentence_id:
                sentence_pred.update(1, 1)
            else:
                sentence_pred.update(0, 1)

            # Find top 5 sentences
            sentence_logits = target_result.sentence_logits
            sentence_num = len(target_feature.sentence_span_list)
            sentence_logits_item = [(i, x) for i, x in enumerate(sentence_logits[:sentence_num])]

            # Find sentence
            gold_span_sen_id = -1
            for p_sen_idx, orig_sen_id in enumerate(target_feature.meta_data["span_sen_to_orig_sen_map"]):
                if orig_sen_id == example.sentence_id:
                    gold_span_sen_id = p_sen_idx
            if gold_span_sen_id != -1:
                gold_span_sen_logit = target_result.sentence_logits[gold_span_sen_id]
            else:
                gold_span_sen_logit = 0

            sorted_sentence_logits_item = sorted(sentence_logits_item, key=lambda x: x[1], reverse=True)
            top_5_sentences = [target_feature.meta_data["span_sen_to_orig_sen_map"][x[0]] for x in sorted_sentence_logits_item[:5]]

            assert sentence_id_pred in top_5_sentences

            all_predictions[example.qas_id] = {
                'sentence_id': sentence_id_pred,
                'gold_sentence_logit': gold_span_sen_logit,
                'max_weight': target_result.max_weight,
                'max_weight_index': target_result.max_weight_index,
                'yesno_pred': target_choice,
                'yesno_gold': example.is_impossible,
                'sentence_logits': target_result.sentence_logits,
                'feature_gold_sentence': target_feature.sentence_id,
                'gold_sentence_id': example.sentence_id,
                'top_5_sentence_ids': top_5_sentences,
                'yesno_pred_prob': target_choice_prob,
                'doc_span_index': target_feature.doc_span_index,
                'span_sen_to_orig_sen_map': target_feature.meta_data['span_sen_to_orig_sen_map']
            }

        logger.info(f'Yesno Prediction Result: {yesno_pred.avg}')
        logger.info('Sentence Prediction Result: {}'.format(sentence_pred.avg))

        if output_prediction_file is not None:
            with open(output_prediction_file, 'w') as f:
                json.dump(all_predictions, f, indent=2)
        return {'predictions': all_predictions, 'acc': sentence_pred.avg}

    @staticmethod
    def predict_sentence_ids(all_examples, all_features: List[SQuADFullInputFeatures], all_results: List[RawOutput],
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
        for (example_index, example) in enumerate(all_examples):
            features = example_index_to_features[example_index]

            max_diff = -1000000
            max_diff_feature_index = 0
            max_yes_logit = 0
            max_no_logit = 0
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                choice_logits = result.model_output['choice_logits']
                non_null_logit = choice_logits[1] + choice_logits[2]
                null_logit = choice_logits[0]
                diff = non_null_logit - null_logit
                if diff > max_diff:
                    max_diff = diff
                    max_diff_feature_index = feature_index
                    max_yes_logit = choice_logits[1]
                    max_no_logit = choice_logits[2]

            # Yes/No prediction restriction
            yesno_scores = utils.compute_softmax([max_yes_logit, max_no_logit])
            if max_yes_logit > max_no_logit:
                final_choice = 0
                prediction_prob = yesno_scores[0]
            else:
                final_choice = 1
                prediction_prob = yesno_scores[1]

            target_feature = features[max_diff_feature_index]
            evidence = unique_id_to_result[target_feature.unique_id].model_output['evidence']
            evidence_value = evidence['value']
            if (only_correct and final_choice + 1 == target_feature.is_impossible and prediction_prob > label_threshold) \
                    or not only_correct:
                if evidence_value > weight_threshold:
                    sentence_id = evidence['sentences']
                else:
                    sentence_id = []
            else:
                sentence_id = []
            if not sentence_id:
                no_label += 1

            feature_doc_span_index = target_feature.doc_span_index
            all_predictions[example.qas_id] = {
                'sentence_id': sentence_id,
                'doc_span_index': feature_doc_span_index,
                'weight': evidence_value,
                'choice_prediction': 'yes' if final_choice == 0 else 'no',
                'prediction_prob': prediction_prob
            }

            total += 1

        logger.info("Labeling {} instances of {} in total".format(total - no_label, len(all_examples)))

        if output_prediction_file is not None:
            with open(output_prediction_file, 'w') as f:
                json.dump(all_predictions, f, indent=2)
        return all_predictions

    @staticmethod
    def data_to_tensors(all_features: List[SQuADFullInputFeatures]):

        all_input_ids = torch.tensor([f.input_ids for f in all_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in all_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in all_features], dtype=torch.long)
        all_answer_choices = torch.tensor([f.is_impossible for f in all_features], dtype=torch.long)
        # all_sentence_ids = torch.tensor([f.sentence_id for f in all_features], dtype=torch.long)
        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        return all_input_ids, all_segment_ids, all_input_mask, all_answer_choices, all_feature_index

    @staticmethod
    def data_to_tensors_sent_pretrain(all_features: List[SQuADFullInputFeatures]):
        all_features = [feature for feature in all_features if feature.sentence_id]

        all_input_ids = torch.tensor([f.input_ids for f in all_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in all_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in all_features], dtype=torch.long)
        all_answer_choices = torch.tensor([f.is_impossible for f in all_features], dtype=torch.long)
        # all_sentence_ids = torch.tensor([f.sentence_id for f in all_features], dtype=torch.long)
        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        return all_features, (all_input_ids, all_segment_ids, all_input_mask, all_answer_choices, all_feature_index)

    @staticmethod
    def generate_inputs(batch: Tuple, all_features: List[SQuADFullInputFeatures], model_state):
        assert model_state in ModelState
        feature_index = batch[4].tolist()
        sentence_span_list = [all_features[idx].sentence_span_list for idx in feature_index]
        sentence_ids = [all_features[idx].sentence_id for idx in feature_index]
        for sentence_id in sentence_ids:
            assert isinstance(sentence_id, list)
        inputs = {
            "input_ids": batch[0],
            "token_type_ids": batch[1],
            "attention_mask": batch[2],
            "sentence_span_list": sentence_span_list
        }
        if model_state == ModelState.Test:
            return inputs
        elif model_state == ModelState.Train or model_state == ModelState.Evaluate:
            inputs['answer_choice'] = batch[3]
            inputs['sentence_ids'] = sentence_ids

        return inputs
