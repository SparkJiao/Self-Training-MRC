from collections import namedtuple
from typing import Dict, Any, List
from enum import Enum, unique

RawResultChoice = namedtuple('RawResultChoice', ['unique_id', 'choice_logits'])
WeightResultChoice = namedtuple('WeightResultChoice', ['unique_id', 'choice_logits', 'max_weight_index', 'max_weight'])
WeightResult = namedtuple('WeightResult', ['unique_id', 'max_weight_index', 'max_weight', 'sentence_logits', 'choice_logits'])
FullResult = namedtuple('FullResult', ['unique_id', 'full_logits', 'full_labels'])  # full_labels is a one-hot vector
RawOutput = namedtuple('RawOutput', ['unique_id', 'model_output'])


@unique
class ModelState(Enum):
    Train = 777
    Evaluate = 7777
    Test = 77777


@unique
class ReadState(Enum):
    NoNegative = 0
    SampleFromSelf = 1
    SampleFromExternal = 2


class FullResult:

    def __init__(self, unique_id, choice_logits: List[float] = None, sentence_logits: List[float] = None,
                 max_weight_index: int = None, max_weight: float = None):
        self.unique_id = unique_id
        self.choice_logits = choice_logits
        self.sentence_logits = sentence_logits
        self.max_weight_index = max_weight_index
        self.max_weight = max_weight


class SQuADFullExample:
    """
    # The example contains the document and the QA pair, besides, the document's sentences' start and end position.
    # is_impossible can be viewed as answerable/unanswerable or answer type(Yes/No/Extractive/Unknown)
    The example contains all features that may be used.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 sentence_span_list,
                 sentence_id=None,
                 is_impossible=None,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 ral_start_position=None,
                 ral_end_position=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.sentence_span_list = sentence_span_list
        self.sentence_id = sentence_id
        self.is_impossible = is_impossible
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.ral_start_position = ral_start_position
        self.ral_end_position = ral_end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % self.qas_id
        s += ", question_text: %s" % self.question_text
        s += ", doc_tokens: [%s]" % " ".join(self.doc_tokens)
        if self.start_position:
            s += ", start_position: %d" % self.start_position
        if self.start_position:
            s += ", end_position: %d" % self.end_position
        if self.is_impossible:
            s += ", is_impossible: %d" % self.is_impossible
        return s


class SQuADFullInputFeatures(object):
    """A single set of features of Squad sentence example."""

    def __init__(self,
                 qas_id,
                 unique_id,
                 example_index,
                 doc_span_index,
                 sentence_span_list,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 is_impossible=None,
                 sentence_id=None,
                 start_position=None,
                 end_position=None,
                 ral_start_position=None,
                 ral_end_position=None):
        self.qas_id = qas_id
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.sentence_span_list = sentence_span_list
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_impossible = is_impossible
        self.sentence_id = sentence_id
        self.start_position = start_position
        self.end_position = end_position
        self.ral_start_position = ral_start_position
        self.ral_end_position = ral_end_position


class QAFullExample:
    """
    The example contains all features that may be used in QA task.
    Besides, the example add meta data domain, which is a dict, to add additional data in special task.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 sentence_span_list,
                 sentence_id=None,
                 is_impossible=None,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 ral_start_position=None,
                 ral_end_position=None,
                 meta_data: Dict[str, Any] = None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.sentence_span_list = sentence_span_list
        self.sentence_id = sentence_id
        self.is_impossible = is_impossible
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.ral_start_position = ral_start_position
        self.ral_end_position = ral_end_position
        self.meta_data = meta_data

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % self.qas_id
        s += ", question_text: %s" % self.question_text
        s += ", doc_tokens: [%s]" % " ".join(self.doc_tokens)
        if self.start_position:
            s += ", start_position: %d" % self.start_position
        if self.start_position:
            s += ", end_position: %d" % self.end_position
        if self.is_impossible:
            s += ", is_impossible: %d" % self.is_impossible
        return s


class QAFullInputFeatures(object):
    """A single set of features of Squad sentence example."""

    def __init__(self,
                 qas_id,
                 unique_id,
                 example_index,
                 doc_span_index,
                 sentence_span_list,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 is_impossible=None,
                 sentence_id=None,
                 start_position=None,
                 end_position=None,
                 ral_start_position=None,
                 ral_end_position=None,
                 meta_data: Dict[str, Any] = None):
        self.qas_id = qas_id
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.sentence_span_list = sentence_span_list
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_impossible = is_impossible
        self.sentence_id = sentence_id
        self.start_position = start_position
        self.end_position = end_position
        self.ral_start_position = ral_start_position
        self.ral_end_position = ral_end_position
        self.meta_data = meta_data


class BoolQFullExample:
    """
    # The example contains the document and the QA pair, besides, the document's sentences' start and end position.
    # is_impossible can be viewed as answerable/unanswerable or answer type(Yes/No/Extractive/Unknown)
    The example contains all features that may be used.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 sentence_span_list,
                 sentence_id=None,
                 is_impossible=None,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.sentence_span_list = sentence_span_list
        self.sentence_id = sentence_id
        self.is_impossible = is_impossible
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % self.qas_id
        s += ", question_text: %s" % self.question_text
        s += ", doc_tokens: [%s]" % " ".join(self.doc_tokens)
        if self.start_position:
            s += ", start_position: %d" % self.start_position
        if self.start_position:
            s += ", end_position: %d" % self.end_position
        if self.is_impossible:
            s += ", is_impossible: %d" % self.is_impossible
        return s


class BoolQFullInputFeatures(object):
    """A single set of features of Squad sentence example."""

    def __init__(self,
                 qas_id,
                 unique_id,
                 example_index,
                 doc_span_index,
                 sentence_span_list,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 is_impossible=None,
                 sentence_id=None,
                 start_position=None,
                 end_position=None,
                 meta_data=None):
        self.qas_id = qas_id
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.sentence_span_list = sentence_span_list
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_impossible = is_impossible
        self.sentence_id = sentence_id
        self.start_position = start_position
        self.end_position = end_position
        self.meta_data = meta_data


class SingleSentenceFeature(object):

    def __init__(self,
                 qas_id,
                 unique_id,
                 example_index,
                 tokens,
                 ques_input_ids,
                 ques_input_mask,
                 pass_input_ids,
                 pass_input_mask,
                 is_impossible=None,
                 sentence_id=None,
                 meta_data: Dict[str, Any] = None):
        self.qas_id = qas_id
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.ques_input_ids = ques_input_ids
        self.ques_input_mask = ques_input_mask
        self.pass_input_ids = pass_input_ids
        self.pass_input_mask = pass_input_mask
        self.is_impossible = is_impossible
        self.sentence_id = sentence_id
        self.meta_data = meta_data


class MultiChoiceFullExample:
    def __init__(self,
                 qas_id,
                 question_text,
                 options,
                 doc_tokens,
                 sentence_span_list,
                 answer=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.options = options
        self.doc_tokens = doc_tokens
        self.sentence_span_list = sentence_span_list
        self.answer = answer


class MultiChoiceFullFeature:
    def __init__(self,
                 example_index,
                 qas_id,
                 unique_id,
                 choice_features,
                 answer=None):
        self.example_index = example_index
        self.qas_id = qas_id
        self.unique_id = unique_id
        self.choice_features = choice_features
        self.answer = answer


class MultiRCExample:
    def __init__(self, qas_id, doc_tokens, question_text, sentence_span_list, option_text, answer=None, sentence_ids=None):
        self.qas_id = qas_id
        self.doc_tokens = doc_tokens
        self.question_text = question_text
        self.sentence_span_list = sentence_span_list
        self.option_text = option_text
        self.answer = answer
        self.sentence_ids = sentence_ids


class MultiRCFeature:
    def __init__(self, example_index, qas_id, unique_id, input_ids, input_mask, segment_ids, sentence_span_list, answer=None,
                 sentence_ids=None):
        self.example_index = example_index
        self.qas_id = qas_id
        self.unique_id = unique_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.sentence_span_list = sentence_span_list
        self.answer = answer
        self.sentence_ids = sentence_ids
