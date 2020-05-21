import torch
from allennlp.nn.util import masked_softmax, masked_log_softmax
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch import nn
from torch.nn import functional as F

from bert_model import layers
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class BertQAYesnoHierarchicalSingleRNN(BertPreTrainedModel):
    """
    BertForQuestionAnsweringForYesNo

    Model Hierarchical Attention:
        - Use Hierarchical attention module to predict Non/Yes/No.
        - Add supervised to sentence attention.

    Sentence level model.
    """

    def __init__(self, config, evidence_lambda=0.8, negative_lambda=1.0, add_entropy: bool = False, fix_bert: bool = False):
        super(BertQAYesnoHierarchicalSingleRNN, self).__init__(config)
        logger.info(f'The model {self.__class__.__name__} is loading...')
        logger.info(f'The coefficient of evidence loss is {evidence_lambda}')
        logger.info(f'The coefficient of negative samples loss is {negative_lambda}')
        logger.info(f'Fix parameters of BERT: {fix_bert}')
        logger.info(f'Add entropy loss: {add_entropy}')
        # logger.info(f'Use bidirectional attention before summarizing vectors: {bi_attention}')

        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(config.hidden_dropout_prob)

        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.answer_choice = nn.Linear(config.hidden_size, 2)
        if fix_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.doc_sen_self_attn = layers.LinearSelfAttnAllennlp(config.hidden_size)
        self.que_self_attn = layers.LinearSelfAttn(config.hidden_size)

        self.doc_sen_encoder = layers.StackedBRNN(config.hidden_size, config.hidden_size // 2, num_layers=1)

        self.word_similarity = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)
        self.vector_similarity = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)

        # self.yesno_predictor = nn.Linear(config.hidden_size, 2)
        self.yesno_predictor = nn.Linear(config.hidden_size * 2, 3)
        self.evidence_lam = evidence_lambda
        self.negative_lam = negative_lambda
        self.add_entropy = add_entropy

        self.apply(self.init_bert_weights)

    def forward(self, ques_input_ids, ques_input_mask, pass_input_ids, pass_input_mask,
                answer_choice=None, sentence_ids=None, sentence_label=None):

        # Encoding question
        q_len = ques_input_ids.size(1)
        question, _ = self.bert(ques_input_ids, token_type_ids=None, attention_mask=ques_input_mask, output_all_encoded_layers=False)
        # Encoding passage
        batch, max_sen_num, p_len = pass_input_ids.size()
        pass_input_ids = pass_input_ids.reshape(batch * max_sen_num, p_len)
        pass_input_mask = pass_input_mask.reshape(batch * max_sen_num, p_len)
        passage, _ = self.bert(pass_input_ids, token_type_ids=None, attention_mask=pass_input_mask, output_all_encoded_layers=False)

        que_mask = (1 - ques_input_mask).byte()
        que_vec = layers.weighted_avg(question, self.que_self_attn(question, que_mask)).view(batch, 1, -1)

        doc = passage.reshape(batch, max_sen_num * p_len, -1)
        # [batch, max_sen, doc_len] -> [batch * max_sen, doc_len]
        word_sim = self.word_similarity(que_vec, doc).view(batch * max_sen_num, p_len)

        # doc_mask = 1 - pass_input_mask
        doc_mask = pass_input_ids  # 1 for true value and 0 for mask
        # [batch * max_sen, doc_len] -> [batch * max_sen, 1, doc_len] -> [batch * max_sen, 1, h]
        word_hidden = masked_softmax(word_sim, doc_mask, dim=1).unsqueeze(1).bmm(passage)
        word_hidden = word_hidden.view(batch, max_sen_num, -1)

        sentence_mask = pass_input_mask.reshape(batch, max_sen_num, p_len).sum(dim=-1).ge(1.0).float()

        # 1 - doc_mask: 0 for true value and 1 for mask
        doc_vecs = layers.weighted_avg(passage, self.doc_sen_self_attn(passage, 1 - doc_mask)).view(batch, max_sen_num, -1)

        doc_vecs = self.doc_sen_encoder(doc_vecs, 1 - sentence_mask)

        # [batch, 1, max_sen]
        sentence_sim = self.vector_similarity(que_vec, doc_vecs)
        # sentence_scores = masked_softmax(sentence_sim, 1 - sentence_mask)
        sentence_scores = masked_softmax(sentence_sim, sentence_mask)  # 1 for true value and 0 for mask
        sentence_hidden = sentence_scores.bmm(word_hidden).squeeze(1)

        yesno_logits = self.yesno_predictor(torch.cat([sentence_hidden, que_vec.squeeze(1)], dim=1))

        sentence_scores = sentence_scores.squeeze(1)
        max_sentence_score = sentence_scores.max(dim=-1)
        output_dict = {'yesno_logits': yesno_logits,
                       'sentence_logits': sentence_scores,
                       'max_weight': max_sentence_score[0],
                       'max_weight_index': max_sentence_score[1]}
        loss = 0
        if answer_choice is not None:
            choice_loss = F.cross_entropy(yesno_logits, answer_choice, ignore_index=-1)
            loss += choice_loss
        if sentence_ids is not None:
            log_sentence_sim = masked_log_softmax(sentence_sim.squeeze(1), sentence_mask, dim=-1)
            sentence_loss = self.evidence_lam * F.nll_loss(log_sentence_sim, sentence_ids, ignore_index=-1)
            loss += sentence_loss
            if self.add_entropy:
                no_evidence_mask = (sentence_ids != -1)
                entropy = layers.get_masked_entropy(sentence_scores, mask=no_evidence_mask)
                loss += self.evidence_lam * entropy
        if sentence_label is not None:
            # sentence_label: batch * List[k]
            # [batch, max_sen]
            # log_sentence_sim = masked_log_softmax(sentence_sim.squeeze(1), 1 - sentence_mask, dim=-1)
            sentence_prob = 1 - sentence_scores
            log_sentence_sim = - torch.log(sentence_prob + 1e-15)
            negative_loss = 0
            for b in range(batch):
                for sen_id, k in enumerate(sentence_label[b]):
                    negative_loss += k * log_sentence_sim[b][sen_id]
            negative_loss /= batch
            loss += self.negative_lam * negative_loss
        output_dict['loss'] = loss
        return output_dict
