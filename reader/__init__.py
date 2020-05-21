import json

from data.data_instance import ReadState
from general_util.logger import get_child_logger
from general_util.register import reader_dict as registered_readers
from reader.boolq_reader import BoolQYesNoReader
from reader.coqa_reader import CoQAYesNoReader
from reader.coqa_reader_top_k import CoQATopKReader
from reader.coqa_sentence_reader import CoQAYesNoSentenceReader
from reader.coqa_split import CoQASplitSentenceReader
from reader.fever_reader import FeverReader
from reader.msmarco_cb_reader import MSMARCOYesNoCBReader
from reader.msmarco_cb_topk_reader import MSMARCOYesNoCBReaderTopK
from reader.msmarco_reader import MSMARCOYesNoReader
from reader.multi_rc_multiple_reader import MultiRCMultipleReader
from reader.multi_rc_topk_reader import MultiRCTopKReader
from reader.race_multiple_reader import RACEMultipleReader
from reader.race_reader import RACEReader
from reader.squad_reader import SQuADReader
from reader.multi_rc_topk_reader_evidence_search import MultiRCTopKReader as MultiRCTopKReaderSearch
from reader.multi_rc_topk_reader_evidence_search_extended import MultiRCTopKReader as MultiRCTopKReaderSearchExtended

logger = get_child_logger(__name__)

try:
    from reader.coqa_reader_top_k_roberta import CoQATopKReader as CoQATopKReaderRoberta
    from reader.race_multiple_reader_roberta import RACEMultipleReader as RACEMultipleReaderRoberta
    from reader.multi_rc_topk_reader_roberta import MultiRCTopKReader as MultiRCTopKReaderRoberta
except ImportError:
    logger.warn("Couldn't load models with roBERTa")
    CoQATopKReaderRoberta = None
    RACEMultipleReaderRoberta = None
    MultiRCTopKReaderRoberta = None

reader_dict = {
    "coqa": CoQAYesNoReader,
    "boolq": BoolQYesNoReader,
    "ms-marco": MSMARCOYesNoReader,
    'cb-marco': MSMARCOYesNoCBReader,
    'cb-marco-top-k': MSMARCOYesNoCBReaderTopK,
    'fever': FeverReader,
    'race': RACEReader,
    'multiple-race': RACEMultipleReader,
    'multiple-race-roberta': RACEMultipleReaderRoberta,
    'multiple-multi-rc': MultiRCMultipleReader,
    'coqa-top-k': CoQATopKReader,
    'coqa-top-k-roberta': CoQATopKReaderRoberta,
    'topk-multi-rc': MultiRCTopKReader,
    'topk-multi-rc-roberta': MultiRCTopKReaderRoberta,
    # Sentence readers
    'coqa-sen': CoQAYesNoSentenceReader,
    'coqa-single': CoQASplitSentenceReader,
    # Pretrain readers
    'squad-pretrain': SQuADReader,
    # Evidence search readers
    'multi-rc-topk-search': MultiRCTopKReaderSearch,
    'multi-rc-topk-search-ex': MultiRCTopKReaderSearchExtended
}
reader_dict.update(registered_readers)


def initialize_reader(name: str, *arg, **kwargs):
    logger.info('Loading reader {} ...'.format(name))
    return reader_dict[name](*arg, **kwargs)


def prepare_read_params(args):
    if args.reader_name in ['coqa', 'quac', 'coqa-single', 'coqa-top-k', 'coqa-top-k-roberta']:
        read_params = {'dialog_turns': args.max_ctx}
    elif args.reader_name in ['cb-marco', 'ms-marco', 're-quac', 'snli', 'squad-pretrain', 'fever', 'boolq', 'race', 'multiple-race',
                              'multiple-multi-rc', 'topk-multi-rc', 'cb-marco-top-k', 'multiple-race-roberta', 'topk-multi-rc-roberta',
                              'multi-rc-topk-search', 'multi-rc-topk-search-ex']:
        read_params = {}
    elif args.reader_name in ['coqa-sen']:
        if args.do_negative_sampling:
            if args.read_extra_self:
                read_state = ReadState.SampleFromSelf
            else:
                read_state = ReadState.SampleFromExternal
        else:
            read_state = ReadState.NoNegative
        read_params = {'read_state': read_state,
                       'sample_ratio': args.sample_ratio,
                       'dialog_turns': args.max_ctx,
                       'extra_sen_file': args.extra_sen_file}
    else:
        raise RuntimeError(f'Wrong reader_name for {args.reader_name}')

    if args.reader_name in ['coqa-top-k', 'coqa', 'topk-multi-rc']:
        read_params['remove_evidence'] = args.remove_evidence
        read_params['remove_question'] = args.remove_question
        read_params['remove_passage'] = args.remove_passage
        read_params['remove_dict'] = args.remove_dict
    if args.reader_name in ['multi-rc-topk-search', 'multi-rc-topk-search-ex']:
        read_params['evidence_search_file'] = args.evidence_search_file

    return read_params


def read_from_squad(input_file: str):
    with open(input_file, 'r') as f:
        data = json.load(f)['data']

    contexts = data['contexts']
    output = []
    for context in contexts:
        output.extend(context)
    return output
