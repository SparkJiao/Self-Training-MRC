import collections
import json
import math
import random
import re
import string
from collections import Counter
from typing import List, Callable, Tuple, Any

import torch
from pytorch_pretrained_bert.tokenization import BasicTokenizer
from torch.nn.functional import softmax
from allennlp.training import metrics

from eval_utils.CoQA_eval import CoQAEvaluator
from data.data_instance import ModelState

# Named Turple List
DocSpan = collections.namedtuple("DocSpan", ["start", "length"])


def add_sentence_separator(doc_tokens: List[str], sentence_span_list: List[Tuple[int, int]], separator: str = '[SEP]'):
    new_doc_tokens = []
    separator_positions = []
    new_sentence_span_list = []
    for sen_idx, (span_start, span_end) in enumerate(sentence_span_list):
        new_doc_tokens.extend(doc_tokens[span_start: span_end + 1])
        if sen_idx != 0:
            span_start = span_start - 1
            new_sentence_span_list.append((span_start, span_end))
        separator_positions.append(len(new_doc_tokens))
        new_doc_tokens.append(separator)
    return new_doc_tokens, separator_positions[:-1], new_sentence_span_list


# def set_random_seed(seed: int = None):
#     random.seed(seed)

def remove_all_evidence(sentence_span_list, doc_tokens, evidences):
    evidences.sort(reverse=False)
    for index, evidence in enumerate(evidences):
        evi_token_s, evi_token_e = sentence_span_list[evidence]
        doc_tokens = doc_tokens[:evi_token_s] + doc_tokens[(evi_token_e + 1):]
        reduce_offset = evi_token_e - evi_token_s + 1
        sentence_span_list = sentence_span_list[:evidence] + [(s - reduce_offset, e - reduce_offset)
                                                                                 for s, e in sentence_span_list[(evidence + 1):]]
        for pointer in range(index + 1, len(evidences)):
            evidences[pointer] -= 1
    return doc_tokens, sentence_span_list



def generate_random_seq(seq_len_a: int, seq_len_b: int):
    seq_a = [0] * seq_len_a
    seq_b = [1] * seq_len_b
    seq = seq_a + seq_b

    # _set_random_seed(seed)
    random.shuffle(seq)
    return seq


def random_sample(seq, sample_length: int):
    # _set_random_seed(seed)
    return random.sample(seq, sample_length)


def generate_seq_with_negative_sample(initial_seq: List[Any], negative_seq: List[Any], sample_ratio: float,
                                      target_index: int = -1):
    sampling_length = int(len(initial_seq) * sample_ratio)
    negative_samples = random_sample(negative_seq, sampling_length)
    random_new_seq_label = generate_random_seq(len(initial_seq), sampling_length)
    random_new_seq = []
    new_target_index = -1
    positive_pointer = 0
    negative_pointer = 0
    orig_token_map = []

    orig_total_tokens = 0
    new_total_tokens = 0
    for idx, num in enumerate(random_new_seq_label):
        if num == 0:
            for i in range(len(initial_seq[positive_pointer])):
                orig_token_map.append(new_total_tokens + i)
            orig_total_tokens += len(initial_seq[positive_pointer])
            new_total_tokens += len(initial_seq[positive_pointer])

            random_new_seq.append(initial_seq[positive_pointer])
            if new_target_index == -1 and positive_pointer == target_index:
                new_target_index = len(random_new_seq) - 1
            positive_pointer += 1

        else:
            new_total_tokens += len(negative_samples[negative_pointer])
            random_new_seq.append(negative_samples[negative_pointer])
            negative_pointer += 1

    random_new_tokens = []
    sentence_span_list = []
    for sentence in random_new_seq:
        start = len(random_new_tokens)
        end = start + len(sentence) - 1
        sentence_span_list.append((start, end))
        random_new_tokens.extend(sentence)

    assert len(sentence_span_list) == len(random_new_seq)
    assert len(sentence_span_list) == len(random_new_seq_label)

    return random_new_tokens, random_new_seq_label, new_target_index, sentence_span_list, orig_token_map


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def is_punctuation(c):
    # Don't contains '-' compared with string.punctuation
    punc = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
            ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{',
            '|', '}', '~']
    if c in punc:
        return True
    return False


def split_sentence(context, sen_tokenizer):
    sentences = sen_tokenizer.tokenize(context)
    sen_start_list = []
    sen_end_list = []
    for sen in sentences:
        s = context.find(sen)
        assert s != -1
        e = s + len(sen) - 1
        sen_start_list.append(s)
        sen_end_list.append(e)
    return sen_start_list, sen_end_list


def improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                        orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


def check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def get_final_text(pred_text, orig_text, do_lower_case, logger, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return ns_text, ns_to_s_map

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def find_evidence_sentence(sentence_span_list: List[Tuple], rationale_start_position: int, rationale_end_position: int):
    sentence_id = -1
    over_size = 0
    for sen_idx, (t_start, t_end) in enumerate(sentence_span_list):
        if t_end < rationale_start_position:
            continue
        if t_start > rationale_end_position:
            break
        if rationale_start_position <= t_end <= rationale_end_position:
            cur_size = t_end - max(rationale_start_position, t_start) + 1
            if cur_size > over_size:
                over_size = cur_size
                sentence_id = sen_idx
        elif rationale_start_position <= t_start <= rationale_end_position:
            cur_size = rationale_end_position - max(rationale_start_position, t_start) + 1
            if cur_size > over_size:
                over_size = cur_size
                sentence_id = sen_idx
    return sentence_id


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def save(self):
        return {
            'val': self.val,
            'avg': self.avg,
            'sum': self.sum,
            'count': self.count
        }

    def load(self, value: dict):
        if value is None:
            self.reset()
        self.val = value['val'] if 'val' in value else 0
        self.avg = value['avg'] if 'avg' in value else 0
        self.sum = value['sum'] if 'sum' in value else 0
        self.count = value['count'] if 'count' in value else 0


class CategoricalAccuracy(object):
    def __init__(self, label_list: List[str]):
        self.predictions = Counter()
        self.label_list = [label.lower() for label in label_list]
        self.reset()

    def reset(self):
        self.predictions.clear()

    @staticmethod
    def _get_key(gold, pred) -> str:
        return '{} - {}'.format(str(gold).lower(), str(pred).lower())

    @staticmethod
    def _split_key(key: str) -> (str, str):
        strs = key.split(' - ')
        return strs[0], strs[1]

    def update(self, gold, pred):
        self.predictions[self._get_key(gold, pred)] += 1

    def __repr__(self):
        return json.dumps(self.predictions, indent=2)

    def f1_measure(self, positive_label, negative_label):
        true_positive = self.predictions[self._get_key(positive_label, positive_label)]
        false_positive = self.predictions[self._get_key(negative_label, positive_label)]
        true_negative = self.predictions[self._get_key(negative_label, negative_label)]
        false_negative = self.predictions[self._get_key(positive_label, negative_label)]

        precision = float(true_positive) / float(true_positive + false_positive + 1e-13)
        recall = float(true_positive) / float(true_positive + false_negative + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        accuracy = 1.0 * (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        result = {'precision': precision, 'recall': recall, 'f1': f1_measure, 'accuracy': accuracy}
        return result

    def read_predictions(self, ground_truths, predictions):
        """
        :param ground_truths: ground_truths[(story_id, qid)]=List[List[answer_text]]
        :param predictions: official format predictions
        :return:
        """
        for pred in predictions:
            story_id = pred['id']
            turn_id = pred['turn_id']

            pred_text = CoQAEvaluator.normalize_answer(pred['answer'])
            gold_text = CoQAEvaluator.normalize_answer(ground_truths[(story_id, turn_id)][0])

            label_list = self.label_list
            if pred_text not in label_list:
                pred_label = 'not'
            else:
                pred_label = pred_text
            if gold_text not in label_list:
                gold_label = 'not'
            else:
                gold_label = gold_text

            self.update(gold_label, pred_label)


class AttentionWeightWriter(object):
    def __init__(self, log_file):
        self.log_file = open(log_file, 'w')

    def write_weights(self, attention_matrix: torch.Tensor, col_ids: torch.Tensor = None, row_ids: torch.Tensor = None,
                      col_mask: torch.Tensor = None, row_mask: torch.Tensor = None, id_to_str: Callable = None,
                      do_softmax: bool = False):

        attn_matrix = attention_matrix.detach().cpu()
        if do_softmax:
            attn_matrix = softmax(attn_matrix, dim=-1)
        else:
            attn_matrix.exp_()
        batch, len1, len2 = attn_matrix.size()
        if col_ids is not None:
            col_ids = col_ids.detach().cpu()
        if row_ids is not None:
            row_ids = row_ids.detach().cpu()
        if col_mask is None:
            col_mask = torch.zeros(batch, len1)
        else:
            col_mask = col_mask.detach().cpu()
        if row_mask is None:
            row_mask = torch.zeros(batch, len2)
        else:
            row_mask = row_mask.detach().cpu()

        for batch_id in range(batch):
            print('batch_id = {}\t'.format(batch_id), file=self.log_file)
            row_is_null = []
            for j in range(len2):
                t_str = self.index_to_token(index=(batch_id, j), ids=row_ids, mask=row_mask, id_to_str=id_to_str)
                if t_str is None:
                    row_is_null.append(True)
                    continue
                else:
                    row_is_null.append(False)
                    print(t_str, end='\t', file=self.log_file)
            print(file=self.log_file)
            for i in range(len1):
                col_t_str = self.index_to_token(index=(batch_id, i), ids=col_ids, mask=col_mask, id_to_str=id_to_str)
                if col_t_str is None:
                    continue
                else:
                    print(col_t_str, end='\t', file=self.log_file)
                for j in range(len2):
                    if row_is_null[j]:
                        continue
                    else:
                        print(attn_matrix[batch_id, i, j].item(), end='\t', file=self.log_file)
                print(file=self.log_file)
            print('======================', file=self.log_file)

    @staticmethod
    def index_to_token(index, ids: torch.Tensor, mask: torch.Tensor, id_to_str: Callable = None):
        if mask[index] == 1:
            return None
        else:
            if ids is None:
                token_id = index[-1]
                return token_id

            token_id = ids[index].item()
            if id_to_str is not None:
                return id_to_str(token_id)
            else:
                return token_id
