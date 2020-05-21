from bert_model.bert_cls_hierarchical import BertQAYesnoCLSHierarchical
from bert_model.bert_for_race import BertRACEPool, BertRACEHierarchical, BertRACEHierarchicalTwoView, \
    BertRACEHierarchicalMultiple, BertRACEHierarchicalTwoViewMultiple, BertRACEHierarchicalTopK, BertRACEHierarchicalTwoViewTopK
from bert_model.bert_hierarchical import BertQAYesnoHierarchical
from bert_model.bert_hierarchical import BertQAYesnoHierarchical, BertQAYesnoSimpleHierarchical, \
    BertQAYesnoHierarchicalTopKfp32
from bert_model.bert_hierarchical_for_sentence_pretrain import BertQAYesnoHierarchicalPretrain
from bert_model.bert_hierarchical_half import BertQAYesnoHierarchicalNegHalf
from bert_model.bert_hierarchical_hard import BertQAYesnoHierarchicalHard, BertQAYesnoHierarchicalReinforce, \
    BertQAYesnoHierarchicalHardFP16, BertQAYesnoHierarchicalReinforceFP16
from bert_model.bert_hierarchical_multiple import BertQAYesnoHierarchicalMultiple, BertQAYesnoHierarchicalTopK
from bert_model.bert_hierarchical_negative import BertQAYesnoHierarchicalNeg
from bert_model.bert_hierarchical_single import BertQAYesnoHierarchicalSingle
from bert_model.bert_hierarchical_single_rnn import BertQAYesnoHierarchicalSingleRNN
from bert_model.bert_hierarchical_twoview import BertQAYesnoHierarchicalTwoView, BertQAYesnoHierarchicalTwoViewTopK, \
    BertQAYesnoHierarchicalTwoViewTopKfp16
from bert_model.bert_mlp import BertQAYesNoMLP
from bert_model.bert_transformer_hierarchical import BertSplitPreTrainedModel, BertHierarchicalTransformer, \
    BertHierarchicalRNN, BertHierarchicalTransformer1, BertHierarchicalRNN1, BertHierarchicalRNN2
from bert_model.bert_hierarchical_hard_race import BertQAYesnoHierarchicalHardRACE, BertQAYesnoHierarchicalReinforceRACE
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)

try:
    from bert_model.bert_for_race_roberta import RobertaRACEHierarchicalTopK
    from bert_model.bert_hierarchical_roberta import RobertaQAYesnoHierarchicalTopKfp32, RobertaQAYesnoHierarchicalTopK
except ImportError:
    logger.warn("Couldn't load models with roBERTa")
    RobertaRACEHierarchicalTopK = None
    RobertaQAYesnoHierarchicalTopKfp32 = None
    RobertaQAYesnoHierarchicalTopK = None

model_dict = {
    "mlp": BertQAYesNoMLP,
    "hie": BertQAYesnoHierarchical,
    "twoview": BertQAYesnoHierarchicalTwoView,
    "twoview-topk": BertQAYesnoHierarchicalTwoViewTopK,
    "twoview-topk-fp16": BertQAYesnoHierarchicalTwoViewTopKfp16,
    "hie-neg": BertQAYesnoHierarchicalNeg,
    "hie-pretrain": BertQAYesnoHierarchicalPretrain,
    'cls-hie': BertQAYesnoCLSHierarchical,
    'hie-neg-half': BertQAYesnoHierarchicalNegHalf,
    'hie-single': BertQAYesnoHierarchicalSingle,
    'hie-single-rnn': BertQAYesnoHierarchicalSingleRNN,
    'transformer-hie': BertHierarchicalTransformer,
    'transformer-hie-1': BertHierarchicalTransformer1,
    'rnn-hie': BertHierarchicalRNN,
    'rnn-hie-1': BertHierarchicalRNN1,
    'rnn-hie-2': BertHierarchicalRNN2,
    'simple-hie': BertQAYesnoSimpleHierarchical,
    'pool-race': BertRACEPool,
    'hie-race': BertRACEHierarchical,
    'two-view-race': BertRACEHierarchicalTwoView,
    'multiple-hie-race': BertRACEHierarchicalMultiple,
    'multiple-two-view-race': BertRACEHierarchicalTwoViewMultiple,
    'topk-two-view-race': BertRACEHierarchicalTwoViewTopK,
    'topk-hie-race': BertRACEHierarchicalTopK,
    'topk-hie-race-roberta': RobertaRACEHierarchicalTopK,
    'hie-hard': BertQAYesnoHierarchicalHard,
    'hie-hard-16': BertQAYesnoHierarchicalHardFP16,
    'hie-reinforce': BertQAYesnoHierarchicalReinforce,
    'hie-reinforce-16': BertQAYesnoHierarchicalReinforceFP16,
    'hie-multiple': BertQAYesnoHierarchicalMultiple,
    'hie-topk': BertQAYesnoHierarchicalTopK,
    'hie-topk-32': BertQAYesnoHierarchicalTopKfp32,
    'hie-topk-roberta': RobertaQAYesnoHierarchicalTopK,
    'hie-race-hard': BertQAYesnoHierarchicalHardRACE,
    'hie-race-reinforce': BertQAYesnoHierarchicalReinforceRACE
}


def initialize_model(name, *arg, **kwargs):
    logger.info('Loading model {} ...'.format(name))
    return model_dict[name].from_pretrained(*arg, **kwargs)


def prepare_model_params(args):
    if args.bert_name in ['mlp', 'pool-race']:
        model_params = {}
    elif args.bert_name in ['hie', 'hie-pretrain', 'rnn-hie', 'rnn-hie-1', 'rnn-hie-2', 'simple-hie', 'multiple-hie-race',
                            'hie-multiple', 'topk-hie-race', 'hie-topk', 'hie-topk-32', 'topk-hie-race-roberta',
                            'hie-topk-roberta']:
        model_params = {'evidence_lambda': args.evidence_lambda}
    elif args.bert_name in ['hie-neg']:
        model_params = {'evidence_lambda': args.evidence_lambda,
                        'negative_lambda': args.negative_lambda,
                        'add_entropy': args.add_entropy}
    elif args.bert_name in ['cls-hie']:
        model_params = {'evidence_lambda': args.evidence_lambda,
                        'cls_sup': args.cls_sup,
                        'extra_yesno_lambda': args.extra_yesno_lambda}
    elif args.bert_name in ['hie-neg-half']:
        model_params = {'evidence_lambda': args.evidence_lambda,
                        'negative_lambda': args.negative_lambda,
                        'add_entropy': args.add_entropy,
                        'split_num': args.split_num,
                        'split_index': args.split_index}
    elif args.bert_name in ['hie-single', 'hie-single-rnn']:
        model_params = {'evidence_lambda': args.evidence_lambda,
                        'negative_lambda': args.negative_lambda,
                        'add_entropy': args.add_entropy}
    elif args.bert_name in ['twoview', 'multiple-two-view-race', 'twoview-topk', 'topk-two-view-race', 'twoview-topk-fp16']:
        model_params = {
            'evidence_lambda': args.evidence_lambda,
            'view_id': args.view_id
        }
    elif args.bert_name in ['transformer-hie', 'transformer-hie-1']:
        model_params = {
            'evidence_lambda': args.evidence_lambda,
            'tf_layers': args.tf_layers,
            'tf_inter_size': args.tf_inter_size
        }
    elif args.bert_name in ['hie-race']:
        model_params = {
            'evidence_lambda': args.evidence_lambda,
            'multi_evidence': args.multi_evidence
        }
    elif args.bert_name in ['two-view-race']:
        model_params = {
            'evidence_lambda': args.evidence_lambda,
            'multi_evidence': args.multi_evidence,
            'view_id': args.view_id
        }
    elif args.bert_name in ['hie-hard', 'hie-hard-16']:
        model_params = {
            'evidence_lambda': args.evidence_lambda,
            'use_gumbel': args.use_gumbel,
            'freeze_bert': args.freeze_bert
        }
    elif args.bert_name in ['hie-reinforce', 'hie-reinforce-16']:
        model_params = {
            'evidence_lambda': args.evidence_lambda,
            'sample_steps': args.sample_steps,
            'reward_func': args.reward_func,
            'freeze_bert': args.freeze_bert
        }
    elif args.bert_name in ['hie-race-hard']:
        model_params = {
            'evidence_lambda': args.evidence_lambda,
            'use_gumbel': args.use_gumbel,
            'freeze_bert': args.freeze_bert
        }
    elif args.bert_name in ['hie-race-reinforce']:
        model_params = {
            'evidence_lambda': args.evidence_lambda,
            'sample_steps': args.sample_steps,
            'reward_func': args.reward_func,
            'freeze_bert': args.freeze_bert
        }
    else:
        raise RuntimeError(f'Wrong bert_name for {args.bert_name}')
    if 'race' in args.bert_name:
        model_params['num_choices'] = args.num_choices
    if args.bert_name == 'twoview-topk':
        model_params['split_type'] = args.split_type
    if args.bert_name in ['hie-topk-32', 'hie-topk']:
        model_params['freeze_predictor'] = args.freeze_predictor
    return model_params
