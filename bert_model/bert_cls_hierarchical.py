import torch
from allennlp.nn.util import masked_softmax, masked_log_softmax
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch import nn
from torch.nn import functional as F

from bert_model import layers


class BertQAYesnoCLSHierarchical(BertPreTrainedModel):
    """
    BertForQuestionAnsweringForYesNo

    Model Hierarchical Attention:
        - Use hidden state of [CLS] to calculate attention with words in question and words in sentence in passage to
            obtain the vector representation.
        - Could add choice supervision to [CLS] to make it known global knowledge.
        - Add supervised to sentence attention.
    """

    def __init__(self, config, cls_sup: bool = False, evidence_lambda=0.8, extra_yesno_lambda=0.5):
        super(BertQAYesnoCLSHierarchical, self).__init__(config)
        print(f'The model {self.__class__.__name__} is loading...')
        print(f'The coefficient of evidence loss is {evidence_lambda}')
        print(f'Use cls extra supervision: {cls_sup}')
        print(f'The extra yesno loss lambda is {extra_yesno_lambda}')

        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(config.hidden_dropout_prob)

        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.answer_choice = nn.Linear(config.hidden_size, 2)
        self.doc_word_sum = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)
        self.que_word_sum = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)
        self.doc_sen_sum = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)

        self.cls_sup = cls_sup
        self.extra_yesno_lam = extra_yesno_lambda
        if cls_sup:
            self.extra_predictor = nn.Linear(config.hidden_size, 3)

        self.yesno_predictor = nn.Linear(config.hidden_size * 2, 3)
        self.evidence_lam = evidence_lambda

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, answer_choice=None,
                sentence_span_list=None, sentence_ids=None, sentence_label=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        # mask: 1 for masked value and 0 for true value
        # doc, que, doc_mask, que_mask = layers.split_doc_que(sequence_output, token_type_ids, attention_mask)
        doc_sen, que, doc_sen_mask, que_mask, sentence_mask, cls_h = \
            layers.split_doc_sen_que(sequence_output, token_type_ids, attention_mask, sentence_span_list, return_cls=True)

        batch, max_sen, doc_len = doc_sen_mask.size()
        que_mask = 1 - que_mask
        doc_sen_mask = 1 - doc_sen_mask
        sentence_mask = 1 - sentence_mask

        cls_h.unsqueeze_(1)
        que_vec = masked_softmax(self.que_word_sum(cls_h, que), que_mask).bmm(que)

        doc = doc_sen.reshape(batch, max_sen * doc_len, -1)
        doc_word_sim = self.doc_word_sum(cls_h, doc).view(batch * max_sen, doc_len)
        doc = doc_sen.reshape(batch * max_sen, doc_len, -1)
        doc_mask = doc_sen_mask.reshape(batch * max_sen, doc_len)
        doc_sen_h = masked_softmax(doc_word_sim, doc_mask, dim=-1).unsqueeze(1).bmm(doc).view(batch, max_sen, -1)

        sentence_sim = self.doc_sen_sum(que_vec, doc_sen_h)
        sentence_scores = masked_softmax(sentence_sim, sentence_mask)
        doc_vec = sentence_scores.bmm(doc_sen_h).squeeze(1)

        yesno_logits = self.yesno_predictor(torch.cat([doc_vec, que_vec.squeeze(1)], dim=1))

        output_dict = {'yesno_logits': yesno_logits,
                       'sentence_scores': sentence_scores}
        loss = 0
        if answer_choice is not None:
            loss += F.cross_entropy(yesno_logits, answer_choice, ignore_index=-1)
        if sentence_ids is not None:
            log_sentence_sim = masked_log_softmax(sentence_sim.squeeze(1), sentence_mask, dim=-1)
            sentence_loss = self.evidence_lam * F.nll_loss(log_sentence_sim, sentence_ids, ignore_index=-1)
            loss += sentence_loss
        if self.cls_sup and answer_choice is not None:
            extra_yesno_logits = self.extra_predictor(cls_h.squeeze(1))
            extra_choice_loss = self.extra_yesno_lam * F.cross_entropy(extra_yesno_logits, answer_choice, ignore_index=-1)
            loss += extra_choice_loss
        output_dict['loss'] = loss
        return output_dict
