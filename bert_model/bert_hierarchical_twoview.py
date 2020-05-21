import os
import random

import torch
from allennlp.nn.util import masked_softmax, masked_log_softmax
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch import nn
from torch.nn import functional as F

from bert_model import layers, rep_layers


class BertQAYesnoHierarchicalTwoView(BertPreTrainedModel):
    """
    BertForQuestionAnsweringForYesNo

    Model Hierarchical Attention:
        - Use Hierarchical attention module to predict Non/Yes/No.
        - Add supervised to sentence attention.
    """

    def __init__(self, config, evidence_lambda=0.8, view_id=1):
        super(BertQAYesnoHierarchicalTwoView, self).__init__(config)
        print(f'The model {self.__class__.__name__} is loading...')
        print(f'The coefficient of evidence loss is {evidence_lambda}')
        print(f'The coefficient of view id is {view_id}')

        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(config.hidden_dropout_prob)

        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.answer_choice = nn.Linear(config.hidden_size, 2)
        config.hidden_size = int(config.hidden_size / 2)

        self.doc_sen_self_attn = layers.LinearSelfAttnAllennlp(config.hidden_size)
        self.que_self_attn = layers.LinearSelfAttn(config.hidden_size)

        self.word_similarity = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)
        self.vector_similarity = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)

        # self.yesno_predictor = nn.Linear(config.hidden_size, 2)
        self.yesno_predictor = nn.Linear(config.hidden_size * 2, 3)
        self.evidence_lam = evidence_lambda
        self.view_id = view_id

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, answer_choice=None,
                sentence_span_list=None, sentence_ids=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        origin_hidden_size = int(sequence_output.size(2) / 2)
        sequence_output = sequence_output[:, :, self.view_id * origin_hidden_size:(self.view_id + 1) * origin_hidden_size]

        # mask: 1 for masked value and 0 for true value
        # doc, que, doc_mask, que_mask = layers.split_doc_que(sequence_output, token_type_ids, attention_mask)
        doc_sen, que, doc_sen_mask, que_mask, sentence_mask = \
            layers.split_doc_sen_que(sequence_output, token_type_ids, attention_mask, sentence_span_list)

        # check_sentence_id_class_num(sentence_mask, sentence_ids)

        batch, max_sen, doc_len = doc_sen_mask.size()
        # que_len = que_mask.size(1)

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
        # Test performance of only evidence sentences
        # if not self.training:
        #     max_index = masked_softmax(sentence_sim, 1 - sentence_mask).max(dim=-1, keepdim=True)[1]
        #     one_hot_vec = torch.zeros_like(sentence_sim).scatter_(-1, max_index, 1.0)
        #     sentence_hidden = one_hot_vec.bmm(word_hidden).squeeze(1)
        # else:
        # Test performance of only max k evidence sentences
        # if not self.training:
        #     k_max_mask = rep_layers.get_k_max_mask(sentence_sim * (1 - sentence_mask.unsqueeze(1)).to(sentence_sim.dtype), dim=-1, k=2)
        #     sentence_hidden = masked_softmax(sentence_sim, k_max_mask).bmm(word_hidden).squeeze(1)
        # else:
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
            log_sentence_sim = masked_log_softmax(sentence_sim.squeeze(1), 1 - sentence_mask, dim=-1)
            sentence_loss = self.evidence_lam * F.nll_loss(log_sentence_sim, sentence_ids, ignore_index=-1)
            loss += sentence_loss
        output_dict['loss'] = loss
        return output_dict


class BertQAYesnoHierarchicalTwoViewTopK(BertPreTrainedModel):
    """
    BertForQuestionAnsweringForYesNo

    Model Hierarchical Attention:
        - Use Hierarchical attention module to predict Non/Yes/No.
        - Add supervised to sentence attention.
    """

    def __init__(self, config, evidence_lambda=0.8, view_id=1, split_type=0):
        super(BertQAYesnoHierarchicalTwoViewTopK, self).__init__(config)
        print(f'The model {self.__class__.__name__} is loading...')
        print(f'The coefficient of evidence loss is {evidence_lambda}')
        print(f'The coefficient of view id is {view_id}')

        total_dim = 768
        rank_list = list(range(total_dim))
        self.split_type = split_type
        if split_type == 0:
            self.view_ranks = [rank_list[:int(.5 * total_dim)], rank_list[int(.5 * total_dim):]]
        elif split_type == 1:
            self.view_ranks = [rank_list[::2], rank_list[1::2]]
        elif split_type == 2:
            random.seed(19970417)
            random.shuffle(rank_list)
            self.view_ranks = [rank_list[:int(.5 * total_dim)], rank_list[int(.5 * total_dim):]]
        elif split_type == 3:
            random.seed(20190914)
            random.shuffle(rank_list)
            self.view_ranks = [rank_list[:int(.5 * total_dim)], rank_list[int(.5 * total_dim):]]
        else:
            raise ValueError("split type should be 0/1/2/3, but found %d" % (split_type))
        print(self.view_ranks[view_id][:20])

        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(config.hidden_dropout_prob)

        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.answer_choice = nn.Linear(config.hidden_size, 2)
        config.hidden_size = int(config.hidden_size / 2)

        self.doc_sen_self_attn = layers.LinearSelfAttnAllennlp(config.hidden_size)
        self.que_self_attn = layers.LinearSelfAttn(config.hidden_size)

        self.word_similarity = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)
        self.vector_similarity = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)

        # self.yesno_predictor = nn.Linear(config.hidden_size, 2)
        self.yesno_predictor = nn.Linear(config.hidden_size * 2, 3)
        self.evidence_lam = evidence_lambda
        self.view_id = view_id

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, answer_choice=None,
                sentence_span_list=None, sentence_ids=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        origin_hidden_size = int(sequence_output.size(2) / 2)
        sequence_output = sequence_output[:, :, self.view_ranks[self.view_id]]

        # mask: 1 for masked value and 0 for true value
        # doc, que, doc_mask, que_mask = layers.split_doc_que(sequence_output, token_type_ids, attention_mask)
        doc_sen, que, doc_sen_mask, que_mask, sentence_mask = \
            layers.split_doc_sen_que(sequence_output, token_type_ids, attention_mask, sentence_span_list)

        # check_sentence_id_class_num(sentence_mask, sentence_ids)

        batch, max_sen, doc_len = doc_sen_mask.size()
        # que_len = que_mask.size(1)

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
        output_dict['sentence_sim'] = sentence_sim
        output_dict['sentence_mask'] = 1 - sentence_mask
        return output_dict

    @staticmethod
    def get_evidence_loss(sentence_sim, sentence_mask, sentence_ids):
        assert sentence_ids != -1
        if not sentence_ids:
            # unlabeled training data
            return 0.
        # if type(sentence_ids) == int:
        #     # golden labels from dataset during evaluating
        #     sentence_ids = [sentence_ids]
        loss = 0.
        num_sentence_ids = len(sentence_ids) + 1e-15
        while sentence_ids:
            log_sentence_sim = rep_layers.masked_log_softmax(sentence_sim, sentence_mask, dim=-1)

            sim = log_sentence_sim[sentence_ids]
            max_value, max_id = sim.max(dim=-1)
            max_id = sentence_ids[max_id]

            loss += max_value
            sentence_ids.remove(max_id)
            sentence_mask[max_id] = 0
        loss /= num_sentence_ids
        return loss

    def get_batch_evidence_loss(self, sentence_sim, sentence_mask, sentence_ids):
        sentence_sim = sentence_sim.squeeze(1)
        # sentence_mask = 1 - sentence_mask
        batch_loss = [self.get_evidence_loss(x, y, z) for x, y, z in zip(sentence_sim, sentence_mask, sentence_ids) if z]
        if not batch_loss:
            loss = 0.
        else:
            loss = -sum(batch_loss) / len(batch_loss)
        return loss


class BertQAYesnoHierarchicalTwoViewTopKfp16(BertPreTrainedModel):
    """
    BertForQuestionAnsweringForYesNo

    Model Hierarchical Attention:
        - Use Hierarchical attention module to predict Non/Yes/No.
        - Add supervised to sentence attention.
    """

    def __init__(self, config, evidence_lambda=0.8, view_id=1):
        super(BertQAYesnoHierarchicalTwoViewTopKfp16, self).__init__(config)
        print(f'The model {self.__class__.__name__} is loading...')
        print(f'The coefficient of evidence loss is {evidence_lambda}')
        print(f'The coefficient of view id is {view_id}')

        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(config.hidden_dropout_prob)
        rep_layers.set_seq_dropout(True)
        rep_layers.set_my_dropout_prob(config.hidden_dropout_prob)

        self.bert = BertModel(config)
        config.hidden_size = int(config.hidden_size / 2)
        self.doc_sen_self_attn = rep_layers.LinearSelfAttention(config.hidden_size)
        self.que_self_attn = rep_layers.LinearSelfAttention(config.hidden_size)

        self.word_similarity = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)
        self.vector_similarity = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)

        # self.yesno_predictor = nn.Linear(config.hidden_size, 2)
        self.yesno_predictor = nn.Linear(config.hidden_size * 2, 3)
        self.evidence_lam = evidence_lambda
        self.view_id = view_id

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, answer_choice=None,
                sentence_span_list=None, sentence_ids=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        origin_hidden_size = int(sequence_output.size(2) / 2)
        sequence_output = sequence_output[:, :, self.view_id * origin_hidden_size:(self.view_id + 1) * origin_hidden_size]

        # mask: 1 for masked value and 0 for true value
        # doc, que, doc_mask, que_mask = layers.split_doc_que(sequence_output, token_type_ids, attention_mask)
        doc_sen, que, doc_sen_mask, que_mask, sentence_mask = \
            rep_layers.split_doc_sen_que(sequence_output, token_type_ids, attention_mask, sentence_span_list)

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

        output_dict = {'yesno_logits': torch.softmax(yesno_logits, dim=-1).detach().cpu().float()}

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
