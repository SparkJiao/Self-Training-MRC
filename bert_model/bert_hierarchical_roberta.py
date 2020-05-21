import os

import torch
from allennlp.nn.util import masked_softmax, masked_log_softmax
from transformers.modeling_roberta import BertPreTrainedModel, RobertaModel
from torch import nn
from torch.nn import functional as F

from bert_model import layers, rep_layers


class RobertaQAYesnoHierarchicalTopKfp32(BertPreTrainedModel):
    """
    BertForQuestionAnsweringForYesNo

    Model Hierarchical Attention:
        - Use Hierarchical attention module to predict Non/Yes/No.
        - Add supervised to sentence attention.
    """

    def __init__(self, config, evidence_lambda=0.8):
        super(RobertaQAYesnoHierarchicalTopKfp32, self).__init__(config)
        print(f'The model {self.__class__.__name__} is loading...')
        print(f'The coefficient of evidence loss is {evidence_lambda}')

        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(config.hidden_dropout_prob)

        # self.bert = BertModel(config)
        self.roberta = RobertaModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.answer_choice = nn.Linear(config.hidden_size, 2)

        self.doc_sen_self_attn = layers.LinearSelfAttnAllennlp(config.hidden_size)
        self.que_self_attn = layers.LinearSelfAttn(config.hidden_size)

        self.word_similarity = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)
        self.vector_similarity = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)

        # self.yesno_predictor = nn.Linear(config.hidden_size, 2)
        self.yesno_predictor = nn.Linear(config.hidden_size * 2, 3)
        self.evidence_lam = evidence_lambda

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, answer_choice=None,
                sentence_span_list=None, sentence_ids=None):

        sequence_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=None)[0]

        # mask: 1 for masked value and 0 for true value
        # doc, que, doc_mask, que_mask = layers.split_doc_que(sequence_output, token_type_ids, attention_mask)
        doc_sen, que, doc_sen_mask, que_mask, sentence_mask = \
            layers.split_doc_sen_que(sequence_output, token_type_ids, attention_mask, sentence_span_list)

        # check_sentence_id_class_num(sentence_mask, sentence_ids)

        batch, max_sen, doc_len = doc_sen_mask.size()

        que_vec = layers.weighted_avg(que, self.que_self_attn(que, que_mask)).view(batch, 1, -1)

        doc = doc_sen.reshape(batch, max_sen * doc_len, -1)
        # [batch, max_sen, doc_len] -> [batch * max_sen, doc_len]
        word_sim = self.word_similarity(que_vec, doc).view(batch * max_sen, doc_len)

        doc = doc_sen.reshape(batch * max_sen, doc_len, -1)
        doc_mask = doc_sen_mask.reshape(batch * max_sen, doc_len)
        # [batch * max_sen, doc_len] -> [batch * max_sen, 1, doc_len] -> [batch * max_sen, 1, h]
        word_hidden = masked_softmax(word_sim, 1 - doc_mask, dim=1).unsqueeze(1).bmm(doc)
        word_hidden = word_hidden.view(batch, max_sen, -1)

        doc_vecs = layers.weighted_avg(doc, self.doc_sen_self_attn(doc, doc_mask)).view(batch, max_sen, -1)

        # [batch, 1, h]
        # sentence_hidden = self.vector_similarity(que_vec, doc_vecs, x2_mask=sentence_mask, x3=word_hidden).squeeze(1)
        # [batch, 1, max_sen]
        sentence_sim = self.vector_similarity(que_vec, doc_vecs)
        sentence_hidden = masked_softmax(sentence_sim, 1 - sentence_mask).bmm(word_hidden).squeeze(1)

        yesno_logits = self.yesno_predictor(torch.cat([sentence_hidden, que_vec.squeeze(1)], dim=1))

        sentence_scores = masked_softmax(sentence_sim, 1 - sentence_mask, dim=-1).squeeze_(1)
        output_dict = {'yesno_logits': yesno_logits,
                       'sentence_logits': sentence_scores,
                       'max_weight_index': sentence_scores.max(dim=1)[1],
                       'max_weight': sentence_scores.max(dim=1)[0]}
        loss = 0
        if answer_choice is not None:
            choice_loss = F.cross_entropy(yesno_logits, answer_choice, ignore_index=-1)
            loss += choice_loss
            if sentence_ids is not None:
                sentence_loss = self.get_batch_evidence_loss(sentence_sim, 1 - sentence_mask, sentence_ids)
                loss += self.evidence_lam * sentence_loss
        output_dict['loss'] = loss
        output_dict['sentence_sim'] = sentence_sim.detach().cpu().float()
        output_dict['sentence_mask'] = (1 - sentence_mask).detach().cpu().float()
        return output_dict

    @staticmethod
    def get_evidence_loss(sentence_sim, sentence_mask, sentence_ids):
        """
        sentence_sim: [max_sentence_num] -> torch.Tensor
        sentence_mask: [max_sentence_num] -> torch.Tensor
        sentence_ids: [len] -> List[int]
        """
        if not sentence_ids:
            print('xxx')
            # Unlabeled training data
            return 0.
        # if type(sentence_ids) == int:
        # # Golden labels from dataset during evaluating
        # return 0.
        loss = 0.
        num_sentence_ids = len(sentence_ids) + 1e-15
        while sentence_ids:
            log_sentence_sim = rep_layers.masked_log_softmax(sentence_sim, sentence_mask, dim=-1)

            sim = log_sentence_sim[sentence_ids]
            assert sim.size(0) != 0, (sentence_ids, sim, log_sentence_sim)
            max_value, max_id = sim.max(dim=-1)
            max_id = sentence_ids[max_id]

            loss += max_value
            sentence_ids.remove(max_id)
            sentence_mask[max_id] = 0
        loss /= num_sentence_ids
        return loss

    def get_batch_evidence_loss(self, sentence_sim, sentence_mask, sentence_ids):
        """
        sentence_sim: [batch, 1, max_sentence_num] -> torch.Tensor
        sentence_mask: [batch, max_sentence_num] -> torch.Tensor
        sentence_ids: [len] -> List[int]
        """
        sentence_sim = sentence_sim.squeeze(1)
        # batch, max_sentence_num = sentence_mask.size()
        loss = [self.get_evidence_loss(x, y, z) for x, y, z in zip(sentence_sim, sentence_mask, sentence_ids) if z]
        if not loss:
            loss = 0.
        else:
            loss = -sum(loss) / len(loss)
        return loss


class RobertaQAYesnoHierarchicalTopK(BertPreTrainedModel):
    """
    BertForQuestionAnsweringForYesNo

    Model Hierarchical Attention:
        - Use Hierarchical attention module to predict Non/Yes/No.
        - Add supervised to sentence attention.
    """

    def __init__(self, config, evidence_lambda=0.8):
        super(RobertaQAYesnoHierarchicalTopK, self).__init__(config)
        print(f'The model {self.__class__.__name__} is loading...')
        print(f'The coefficient of evidence loss is {evidence_lambda}')

        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(config.hidden_dropout_prob)
        rep_layers.set_seq_dropout(True)
        rep_layers.set_my_dropout_prob(config.hidden_dropout_prob)

        self.roberta = RobertaModel(config)
        self.doc_sen_self_attn = rep_layers.LinearSelfAttention(config.hidden_size)
        self.que_self_attn = rep_layers.LinearSelfAttention(config.hidden_size)

        self.word_similarity = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)
        self.vector_similarity = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)

        # self.yesno_predictor = nn.Linear(config.hidden_size, 2)
        self.yesno_predictor = nn.Linear(config.hidden_size * 2, 3)
        self.evidence_lam = evidence_lambda

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, answer_choice=None,
                sentence_span_list=None, sentence_ids=None):
        sequence_output = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=None)[0]

        # mask: 1 for masked value and 0 for true value
        # doc, que, doc_mask, que_mask = layers.split_doc_que(sequence_output, token_type_ids, attention_mask)
        doc_sen, que, doc_sen_mask, que_mask, sentence_mask = \
            rep_layers.split_doc_sen_que_roberta(sequence_output, token_type_ids, attention_mask, sentence_span_list)

        # check_sentence_id_class_num(sentence_mask, sentence_ids)

        batch, max_sen, doc_len = doc_sen_mask.size()

        que_vec = self.que_self_attn(que, que_mask).view(batch, 1, -1)

        doc = doc_sen.reshape(batch, max_sen * doc_len, -1)
        # [batch, max_sen, doc_len] -> [batch * max_sen, doc_len]
        word_sim = self.word_similarity(que_vec, doc).view(batch * max_sen, doc_len)

        doc = doc_sen.reshape(batch * max_sen, doc_len, -1)
        doc_mask = doc_sen_mask.reshape(batch * max_sen, doc_len)
        word_hidden = rep_layers.masked_softmax(word_sim, doc_mask, dim=1).unsqueeze(1).bmm(doc)

        word_hidden = word_hidden.view(batch, max_sen, -1)

        doc_vecs = self.doc_sen_self_attn(doc, doc_mask).view(batch, max_sen, -1)

        # [batch, 1, h]
        # sentence_hidden = self.vector_similarity(que_vec, doc_vecs, x2_mask=sentence_mask, x3=word_hidden).squeeze(1)
        # [batch, 1, max_sen]
        sentence_sim = self.vector_similarity(que_vec, doc_vecs)
        sentence_alpha = rep_layers.masked_softmax(sentence_sim, sentence_mask)
        sentence_hidden = sentence_alpha.bmm(word_hidden).squeeze(1)

        yesno_logits = self.yesno_predictor(torch.cat([sentence_hidden, que_vec.squeeze(1)], dim=1))

        output_dict = {'yesno_logits': torch.softmax(yesno_logits, dim=-1).detach().cpu().float(),
                       'sentence_logits': sentence_alpha.squeeze(1),
                       'max_weight_index': sentence_alpha.squeeze(1).max(dim=1)[1],
                       'max_weight': sentence_alpha.squeeze(1).max(dim=1)[0]}

        loss = 0
        if answer_choice is not None:
            choice_loss = F.cross_entropy(yesno_logits, answer_choice, ignore_index=-1)
            loss += choice_loss
            if sentence_ids is not None:
                sentence_loss = self.get_batch_evidence_loss(sentence_sim, sentence_mask, sentence_ids)
                # print('Sentence loss: ')
                # print(sentence_loss)
                loss += self.evidence_lam * sentence_loss
        output_dict['loss'] = loss
        output_dict['sentence_sim'] = sentence_sim.detach().cpu().float()
        output_dict['sentence_mask'] = sentence_mask.detach().cpu().float()
        return output_dict

    @staticmethod
    def get_evidence_loss(sentence_sim, sentence_mask, sentence_ids):
        """
        sentence_sim: [max_sentence_num] -> torch.Tensor
        sentence_mask: [max_sentence_num] -> torch.Tensor
        sentence_ids: [len] -> List[int]
        """
        if not sentence_ids:
            print('xxx')
            # Unlabeled training data
            return 0.
        # if type(sentence_ids) == int:
        # # Golden labels from dataset during evaluating
        # return 0.
        loss = 0.
        num_sentence_ids = len(sentence_ids) + 1e-15
        while sentence_ids:
            log_sentence_sim = rep_layers.masked_log_softmax(sentence_sim, sentence_mask, dim=-1)

            sim = log_sentence_sim[sentence_ids]
            assert sim.size(0) != 0, (sentence_ids, sim, log_sentence_sim)
            max_value, max_id = sim.max(dim=-1)
            max_id = sentence_ids[max_id]

            loss += max_value
            sentence_ids.remove(max_id)
            sentence_mask[max_id] = 0
        loss /= num_sentence_ids
        return loss

    def get_batch_evidence_loss(self, sentence_sim, sentence_mask, sentence_ids):
        """
        sentence_sim: [batch, 1, max_sentence_num] -> torch.Tensor
        sentence_mask: [batch, max_sentence_num] -> torch.Tensor
        sentence_ids: [len] -> List[int]
        """
        sentence_sim = sentence_sim.squeeze(1)
        # batch, max_sentence_num = sentence_mask.size()
        loss = [self.get_evidence_loss(x, y, z) for x, y, z in zip(sentence_sim, sentence_mask, sentence_ids) if z]
        if not loss:
            loss = 0.
        else:
            loss = -sum(loss) / len(loss)
        return loss
