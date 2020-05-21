import numpy as np
import torch
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch import nn
from torch.nn import functional

from general_util.logger import get_child_logger
from . import layers
from . import rep_layers

logger = get_child_logger(__name__)


class BertRACEPool(BertPreTrainedModel):
    def __init__(self, config, num_choices=4):
        super(BertRACEPool, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sentence_span_list=None,
                sentence_labels=None, sentence_ids=None, sentence_prob=None, max_sentences: int = 0):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        _, pooled_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)
        output_dict = {
            'choice_logits': reshaped_logits.float()
        }

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            output_dict['loss'] = loss

        return output_dict


class BertRACEHierarchical(BertPreTrainedModel):
    def __init__(self, config, evidence_lambda=0.8, num_choices=4, multi_evidence: bool = False):
        super(BertRACEHierarchical, self).__init__(config)
        logger.info(f'The model {self.__class__.__name__} is loading...')
        logger.info(f'Currently the number of choices is {num_choices}')
        logger.info(f'The coefficient of evidence loss is {evidence_lambda}')

        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(config.hidden_dropout_prob)
        rep_layers.set_seq_dropout(True)
        rep_layers.set_my_dropout_prob(config.hidden_dropout_prob)

        self.bert = BertModel(config)
        self.doc_sen_self_attn = rep_layers.LinearSelfAttention(config.hidden_size)
        self.que_self_attn = rep_layers.LinearSelfAttention(config.hidden_size)

        self.word_similarity = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)
        self.vector_similarity = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)

        # self.yesno_predictor = nn.Linear(config.hidden_size, 2)
        self.classifier = nn.Linear(config.hidden_size * 2, 1)
        self.evidence_lam = evidence_lambda
        self.num_choices = num_choices
        self.multi_evidence = multi_evidence

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sentence_span_list=None,
                sentence_labels=None, sentence_ids=None, sentence_prob=None, max_sentences: int = 0):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        seq_output, pool_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)

        # mask: 1 for masked value and 0 for true value
        # doc, que, doc_mask, que_mask = layers.split_doc_que(sequence_output, token_type_ids, attention_mask)
        doc_sen, que, doc_sen_mask, que_mask, sentence_mask = \
            rep_layers.split_doc_sen_que(seq_output, flat_token_type_ids, flat_attention_mask, sentence_span_list,
                                         max_sentences=max_sentences)
        # doc_sen_mask = 1 - doc_sen_mask
        # que_mask = 1 - que_mask
        # sentence_mask = 1 - sentence_mask
        # assert doc_sen.sum() != torch.nan
        batch, max_sen, doc_len = doc_sen_mask.size()

        que_vec = self.que_self_attn(que, que_mask).view(batch, 1, -1)

        doc = doc_sen.reshape(batch, max_sen * doc_len, -1)
        word_sim = self.word_similarity(que_vec, doc).view(batch * max_sen, doc_len)
        doc = doc_sen.reshape(batch * max_sen, doc_len, -1)
        doc_mask = doc_sen_mask.reshape(batch * max_sen, doc_len)
        word_hidden = rep_layers.masked_softmax(word_sim, doc_mask, dim=1).unsqueeze(1).bmm(doc)

        word_hidden = word_hidden.view(batch, max_sen, -1)

        doc_vecs = self.doc_sen_self_attn(doc, doc_mask).view(batch, max_sen, -1)

        sentence_sim = self.vector_similarity(que_vec, doc_vecs)
        sentence_alpha = rep_layers.masked_softmax(sentence_sim, sentence_mask)
        sentence_hidden = sentence_alpha.bmm(word_hidden).squeeze(1)
        choice_logits = self.classifier(torch.cat([sentence_hidden, que_vec.squeeze(1)], dim=1)).reshape(-1, self.num_choices)

        if self.training:
            output_dict = {}
        else:
            output_dict = {
                'choice_logits': torch.softmax(choice_logits, dim=-1).detach().cpu().float(),
                'sentence_logits': sentence_alpha.reshape(choice_logits.size(0), self.num_choices, max_sen).detach().cpu().float()
            }

        if labels is not None:
            choice_loss = functional.cross_entropy(choice_logits, labels)
            loss = choice_loss
            if self.multi_evidence and sentence_prob is not None:
                sentence_prob = sentence_prob.reshape(batch, -1)
                true_prob_mask = (sentence_prob > 0).to(dtype=sentence_sim.dtype) * sentence_mask
                kl_sentence_loss = functional.kl_div(sentence_alpha * true_prob_mask, sentence_prob * true_prob_mask, reduction='sum')
                loss += self.evidence_lam * kl_sentence_loss / choice_logits.size(0)
            elif not self.multi_evidence and sentence_ids is not None:
                log_masked_sentence_prob = rep_layers.masked_log_softmax(sentence_sim.squeeze(1), sentence_mask)
                sentence_loss = functional.nll_loss(log_masked_sentence_prob, sentence_ids.view(batch), reduction='sum',
                                                    ignore_index=-1)
                loss += self.evidence_lam * sentence_loss / choice_logits.size(0)
            output_dict['loss'] = loss
        return output_dict


class BertRACEHierarchicalTwoView(BertPreTrainedModel):
    def __init__(self, config, evidence_lambda=0.8, num_choices=4, multi_evidence: bool = False, view_id=1):
        super(BertRACEHierarchicalTwoView, self).__init__(config)
        logger.info(f'The model {self.__class__.__name__} is loading...')
        logger.info(f'Currently the number of choices is {num_choices}')
        logger.info(f'The coefficient of evidence loss is {evidence_lambda}')
        logger.info(f'The view id of current model is {view_id}')

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
        self.classifier = nn.Linear(config.hidden_size * 2, 1)
        self.evidence_lam = evidence_lambda
        self.num_choices = num_choices
        self.multi_evidence = multi_evidence
        self.view_id = view_id

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sentence_span_list=None,
                sentence_labels=None, sentence_ids=None, sentence_prob=None, max_sentences=0):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        seq_output, _ = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
        origin_hidden_size = int(seq_output.size(2) / 2)
        seq_output = seq_output[:, :, self.view_id * origin_hidden_size: (self.view_id + 1) * origin_hidden_size]

        # mask: 1 for masked value and 0 for true value
        # doc, que, doc_mask, que_mask = layers.split_doc_que(sequence_output, token_type_ids, attention_mask)
        doc_sen, que, doc_sen_mask, que_mask, sentence_mask = \
            rep_layers.split_doc_sen_que(seq_output, flat_token_type_ids, flat_attention_mask, sentence_span_list,
                                         max_sentences=max_sentences)
        # doc_sen_mask = 1 - doc_sen_mask
        # que_mask = 1 - que_mask
        # sentence_mask = 1 - sentence_mask
        # assert doc_sen.sum() != torch.nan
        batch, max_sen, doc_len = doc_sen_mask.size()
        assert max_sen == max_sentences

        que_vec = self.que_self_attn(que, que_mask).view(batch, 1, -1)

        doc = doc_sen.reshape(batch, max_sen * doc_len, -1)
        word_sim = self.word_similarity(que_vec, doc).view(batch * max_sen, doc_len)
        doc = doc_sen.reshape(batch * max_sen, doc_len, -1)
        doc_mask = doc_sen_mask.reshape(batch * max_sen, doc_len)
        word_hidden = rep_layers.masked_softmax(word_sim, doc_mask, dim=1).unsqueeze(1).bmm(doc)

        word_hidden = word_hidden.view(batch, max_sen, -1)

        doc_vecs = self.doc_sen_self_attn(doc, doc_mask).view(batch, max_sen, -1)

        sentence_sim = self.vector_similarity(que_vec, doc_vecs)
        sentence_alpha = rep_layers.masked_softmax(sentence_sim, sentence_mask)
        sentence_hidden = sentence_alpha.bmm(word_hidden).squeeze(1)
        choice_logits = self.classifier(torch.cat([sentence_hidden, que_vec.squeeze(1)], dim=1)).reshape(-1, self.num_choices)

        if self.training:
            output_dict = {}
        else:
            output_dict = {
                'choice_logits': torch.softmax(choice_logits, dim=-1).detach().cpu().float(),
                'sentence_logits': sentence_alpha.reshape(choice_logits.size(0), self.num_choices, max_sen).detach().cpu().float()
            }

        if labels is not None:
            choice_loss = functional.cross_entropy(choice_logits, labels)
            loss = choice_loss
            if self.multi_evidence and sentence_prob is not None:
                sentence_prob = sentence_prob.reshape(batch, -1).to(sentence_sim.dtype)
                true_prob_mask = ((sentence_prob > 0).to(dtype=sentence_sim.dtype)) * sentence_mask.to(dtype=sentence_sim.dtype)
                kl_sentence_loss = functional.kl_div(sentence_alpha * true_prob_mask, sentence_prob * true_prob_mask, reduction='sum')
                output_dict['sentence_loss'] = kl_sentence_loss.item()
                loss += self.evidence_lam * kl_sentence_loss / choice_logits.size(0)
            elif not self.multi_evidence and sentence_ids is not None:
                # logger.info(f'sentence_ids total number: {(sentence_ids != -1).sum().item()}')
                # logger.info('sentence_mask.sum() = ', sentence_mask.sum())
                assert sentence_mask.sum() != 0, sentence_mask.sum()
                # assert all(x < sentence_mask.sum() for x in sentence_ids.view(batch).detach().tolist())
                assertion = (sentence_ids.view(batch) >= sentence_mask.sum(dim=-1)).sum()
                
                log_masked_sentence_prob = rep_layers.masked_log_softmax(sentence_sim.squeeze(1), sentence_mask)
                sentence_loss = functional.nll_loss(log_masked_sentence_prob, sentence_ids.view(batch), reduction='sum',
                                                    ignore_index=-1)
                # sentence_loss = functional.cross_entropy(sentence_sim.squeeze(1), sentence_ids.view(batch), reduction='sum',
                #                                          ignore_index=-1)
                # logger.info(f'sentence loss: {sentence_loss.item()}')
                loss += self.evidence_lam * sentence_loss / choice_logits.size(0)
                output_dict['sentence_loss'] = sentence_loss.item()
            output_dict['loss'] = loss
        return output_dict


class BertRACEHierarchicalMultiple(BertPreTrainedModel):
    def __init__(self, config, evidence_lambda=0.8, num_choices=4):
        super(BertRACEHierarchicalMultiple, self).__init__(config)
        logger.info(f'The model {self.__class__.__name__} is loading...')
        logger.info(f'Currently the number of choices is {num_choices}')
        logger.info(f'The coefficient of evidence loss is {evidence_lambda}')

        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(config.hidden_dropout_prob)
        rep_layers.set_seq_dropout(True)
        rep_layers.set_my_dropout_prob(config.hidden_dropout_prob)

        self.bert = BertModel(config)
        self.doc_sen_self_attn = rep_layers.LinearSelfAttention(config.hidden_size)
        self.que_self_attn = rep_layers.LinearSelfAttention(config.hidden_size)

        self.word_similarity = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)
        self.vector_similarity = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)

        # self.yesno_predictor = nn.Linear(config.hidden_size, 2)
        self.classifier = nn.Linear(config.hidden_size * 2, 1)
        self.evidence_lam = evidence_lambda
        self.num_choices = num_choices

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sentence_span_list=None,
                sentence_ids=None, max_sentences: int = 0):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        seq_output, pool_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)

        # mask: 1 for masked value and 0 for true value
        # doc, que, doc_mask, que_mask = layers.split_doc_que(sequence_output, token_type_ids, attention_mask)
        doc_sen, que, doc_sen_mask, que_mask, sentence_mask = \
            rep_layers.split_doc_sen_que(seq_output, flat_token_type_ids, flat_attention_mask, sentence_span_list,
                                         max_sentences=max_sentences)
        # doc_sen_mask = 1 - doc_sen_mask
        # que_mask = 1 - que_mask
        # sentence_mask = 1 - sentence_mask
        # assert doc_sen.sum() != torch.nan
        batch, max_sen, doc_len = doc_sen_mask.size()

        que_vec = self.que_self_attn(que, que_mask).view(batch, 1, -1)

        doc = doc_sen.reshape(batch, max_sen * doc_len, -1)
        word_sim = self.word_similarity(que_vec, doc).view(batch * max_sen, doc_len)
        doc = doc_sen.reshape(batch * max_sen, doc_len, -1)
        doc_mask = doc_sen_mask.reshape(batch * max_sen, doc_len)
        word_hidden = rep_layers.masked_softmax(word_sim, doc_mask, dim=1).unsqueeze(1).bmm(doc)

        word_hidden = word_hidden.view(batch, max_sen, -1)

        doc_vecs = self.doc_sen_self_attn(doc, doc_mask).view(batch, max_sen, -1)

        sentence_sim = self.vector_similarity(que_vec, doc_vecs)
        sentence_alpha = rep_layers.masked_softmax(sentence_sim, sentence_mask)
        sentence_hidden = sentence_alpha.bmm(word_hidden).squeeze(1)
        choice_logits = self.classifier(torch.cat([sentence_hidden, que_vec.squeeze(1)], dim=1)).reshape(-1, self.num_choices)

        if self.training:
            output_dict = {}
        else:
            output_dict = {
                'choice_logits': torch.softmax(choice_logits, dim=-1).detach().cpu().float(),
                'sentence_logits': sentence_sim.reshape(choice_logits.size(0), self.num_choices, max_sen).detach().cpu().float(),
                'sentence_mask': sentence_mask.reshape(choice_logits.size(0), self.num_choices, max_sen).detach().cpu().float()
            }

        if labels is not None:
            choice_loss = functional.cross_entropy(choice_logits, labels)
            loss = choice_loss
            if sentence_ids is not None:
                sentence_loss = self.get_batch_evidence_loss(sentence_sim, sentence_mask, sentence_ids)
                # print(sentence_loss)
                loss += self.evidence_lam * sentence_loss
                # output_dict['sentence_loss'] = sentence_loss
            output_dict['loss'] = loss
        return output_dict

    @staticmethod
    def get_evidence_loss(sentence_sim, sentence_mask, sentence_ids):
        """
        sentence_sim: [max_sentence_num] -> torch.Tensor
        sentence_mask: [max_sentence_num] -> torch.Tensor
        sentence_ids: [len] -> List[int]
        """
        if not sentence_ids:
            # Unlabeled training data
            return 0.
        # if type(sentence_ids) == int:
            # # Golden labels from dataset during evaluating
            # return 0.
        loss = 0.
        num_sentence_ids = len(sentence_ids) + 1e-15
        for id_index, sentence_id in enumerate(sentence_ids):
            log_sentence_sim = rep_layers.masked_log_softmax(sentence_sim, sentence_mask, dim=-1)
            if sentence_id == 0:
                # max_id = 0
                max_value = log_sentence_sim[0]
            else:
                sim = log_sentence_sim[sentence_ids[id_index:-1]]
                max_value, _ = sim.max(dim=-1)
            loss += max_value
            # FIXME:
            #  这里有一个BUG，当此选择了剩余的`sentence_ids`中log prob最大的那个值作为计算熵后，应该把该位置的mask置0，
            #  而不是当前剩余`sentence_ids`的第一个位置置0。即正确的写法应该是每次从剩余的`sentence_ids`中选择概率最大的那个计算loss并把它
            #  的mask和值都去掉，现在的写法是每次不管怎样都去掉剩余列表里的第一个值
            sentence_mask[sentence_ids[id_index]] = 0
        loss /= num_sentence_ids
        return loss

    def get_batch_evidence_loss(self, sentence_sim, sentence_mask, sentence_ids):
        """
        sentence_sim: [batch, 1, max_sentence_num] -> torch.Tensor
        sentence_mask: [batch, max_sentence_num] -> torch.Tensor
        sentence_ids: [len] -> List[int]
        """
        sentence_sim = sentence_sim.squeeze(1)
        batch, max_sentence_num = sentence_mask.size()
        sentence_mask = torch.cat([sentence_mask.new_ones(batch, 1), sentence_mask], dim=1)
        sentence_sim = torch.cat([sentence_sim.new_zeros(batch, 1), sentence_sim], dim=1)
        loss = -sum([self.get_evidence_loss(x, y, z) for x, y, z in zip(sentence_sim, sentence_mask, sentence_ids)])
        loss /= batch
        return loss


class BertRACEHierarchicalTwoViewMultiple(BertPreTrainedModel):
    def __init__(self, config, evidence_lambda=0.8, num_choices=4, view_id=1):
        super(BertRACEHierarchicalTwoViewMultiple, self).__init__(config)
        logger.info(f'The model {self.__class__.__name__} is loading...')
        logger.info(f'Currently the number of choices is {num_choices}')
        logger.info(f'The coefficient of evidence loss is {evidence_lambda}')
        logger.info(f'The view id of current model is {view_id}')

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
        self.classifier = nn.Linear(config.hidden_size * 2, 1)
        self.evidence_lam = evidence_lambda
        self.num_choices = num_choices
        self.view_id = view_id

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sentence_span_list=None, sentence_ids=None,
                max_sentences=0):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        seq_output, _ = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
        origin_hidden_size = int(seq_output.size(2) / 2)
        seq_output = seq_output[:, :, self.view_id * origin_hidden_size: (self.view_id + 1) * origin_hidden_size]

        # mask: 0 for masked value and 1 for true value
        # doc, que, doc_mask, que_mask = layers.split_doc_que(sequence_output, token_type_ids, attention_mask)
        doc_sen, que, doc_sen_mask, que_mask, sentence_mask = \
            rep_layers.split_doc_sen_que(seq_output, flat_token_type_ids, flat_attention_mask, sentence_span_list,
                                         max_sentences=max_sentences)
        # doc_sen_mask = 1 - doc_sen_mask
        # que_mask = 1 - que_mask
        # sentence_mask = 1 - sentence_mask
        # assert doc_sen.sum() != torch.nan
        batch, max_sen, doc_len = doc_sen_mask.size()
        assert max_sen == max_sentences

        que_vec = self.que_self_attn(que, que_mask).view(batch, 1, -1)

        doc = doc_sen.reshape(batch, max_sen * doc_len, -1)
        word_sim = self.word_similarity(que_vec, doc).view(batch * max_sen, doc_len)
        doc = doc_sen.reshape(batch * max_sen, doc_len, -1)
        doc_mask = doc_sen_mask.reshape(batch * max_sen, doc_len)
        word_hidden = rep_layers.masked_softmax(word_sim, doc_mask, dim=1).unsqueeze(1).bmm(doc)

        word_hidden = word_hidden.view(batch, max_sen, -1)

        doc_vecs = self.doc_sen_self_attn(doc, doc_mask).view(batch, max_sen, -1)

        sentence_sim = self.vector_similarity(que_vec, doc_vecs)
        sentence_alpha = rep_layers.masked_softmax(sentence_sim, sentence_mask)
        sentence_hidden = sentence_alpha.bmm(word_hidden).squeeze(1)
        choice_logits = self.classifier(torch.cat([sentence_hidden, que_vec.squeeze(1)], dim=1)).reshape(-1, self.num_choices)

        if self.training:
            output_dict = {}
        else:
            output_dict = {
                'choice_logits': torch.softmax(choice_logits, dim=-1).detach().cpu().float(),
                # 'sentence_logits': sentence_alpha.reshape(choice_logits.size(0), self.num_choices, max_sen).detach().cpu().float()
                'sentence_logits': sentence_sim.reshape(choice_logits.size(0), self.num_choices, max_sen).detach().cpu().float(),
                'sentence_mask': sentence_mask.reshape(choice_logits.size(0), self.num_choices, max_sen).detach().cpu().float()
            }

        if labels is not None:
            choice_loss = functional.cross_entropy(choice_logits, labels)
            loss = choice_loss
            if sentence_ids is not None:
                sentence_loss = self.get_batch_evidence_loss(sentence_sim, sentence_mask, sentence_ids)
                loss += self.evidence_lam * sentence_loss
                # output_dict['sentence_loss'] = sentence_loss
            output_dict['loss'] = loss
        return output_dict

    @staticmethod
    def get_evidence_loss(sentence_sim, sentence_mask, sentence_ids):
        """
        sentence_sim: [max_sentence_num] -> torch.Tensor
        sentence_mask: [max_sentence_num] -> torch.Tensor
        sentence_ids: [len] -> List[int]
        """
        if not sentence_ids:
            # Unlabeled training data
            return 0.
        # if type(sentence_ids) == int:
            # # Golden labels from dataset during evaluating
            # return 0.
        loss = 0.
        num_sentence_ids = len(sentence_ids) + 1e-15
        for id_index, sentence_id in enumerate(sentence_ids):
            log_sentence_sim = rep_layers.masked_log_softmax(sentence_sim, sentence_mask, dim=-1)
            if sentence_id == 0:
                # max_id = 0
                max_value = log_sentence_sim[0]
            else:
                sim = log_sentence_sim[sentence_ids[id_index:-1]]
                max_value, _ = sim.max(dim=-1)
            loss += max_value
            sentence_mask[sentence_ids[id_index]] = 0
        loss /= num_sentence_ids
        return loss

    def get_batch_evidence_loss(self, sentence_sim, sentence_mask, sentence_ids):
        """
        sentence_sim: [batch, 1, max_sentence_num] -> torch.Tensor
        sentence_mask: [batch, max_sentence_num] -> torch.Tensor
        sentence_ids: [len] -> List[int]
        """
        sentence_sim = sentence_sim.squeeze(1)
        batch, max_sentence_num = sentence_mask.size()
        sentence_mask = torch.cat([sentence_mask.new_ones(batch, 1), sentence_mask], dim=1)
        sentence_sim = torch.cat([sentence_sim.new_zeros(batch, 1), sentence_sim], dim=1)
        loss = -sum([self.get_evidence_loss(x, y, z) for x, y, z in zip(sentence_sim, sentence_mask, sentence_ids)])
        loss /= batch
        return loss


class BertRACEHierarchicalTopK(BertPreTrainedModel):
    def __init__(self, config, evidence_lambda=0.8, num_choices=4):
        super(BertRACEHierarchicalTopK, self).__init__(config)
        logger.info(f'The model {self.__class__.__name__} is loading...')
        logger.info(f'Currently the number of choices is {num_choices}')
        logger.info(f'The coefficient of evidence loss is {evidence_lambda}')

        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(config.hidden_dropout_prob)
        rep_layers.set_seq_dropout(True)
        rep_layers.set_my_dropout_prob(config.hidden_dropout_prob)

        self.bert = BertModel(config)
        self.doc_sen_self_attn = rep_layers.LinearSelfAttention(config.hidden_size)
        self.que_self_attn = rep_layers.LinearSelfAttention(config.hidden_size)

        self.word_similarity = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)
        self.vector_similarity = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)

        # self.yesno_predictor = nn.Linear(config.hidden_size, 2)
        self.classifier = nn.Linear(config.hidden_size * 2, 1)
        self.evidence_lam = evidence_lambda
        self.num_choices = num_choices

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sentence_span_list=None,
                sentence_ids=None, max_sentences: int = 0):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        seq_output, pool_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)

        # mask: 1 for masked value and 0 for true value
        # doc, que, doc_mask, que_mask = layers.split_doc_que(sequence_output, token_type_ids, attention_mask)
        doc_sen, que, doc_sen_mask, que_mask, sentence_mask = \
            rep_layers.split_doc_sen_que(seq_output, flat_token_type_ids, flat_attention_mask, sentence_span_list,
                                         max_sentences=max_sentences)
        # doc_sen_mask = 1 - doc_sen_mask
        # que_mask = 1 - que_mask
        # sentence_mask = 1 - sentence_mask
        # assert doc_sen.sum() != torch.nan
        batch, max_sen, doc_len = doc_sen_mask.size()

        que_vec = self.que_self_attn(que, que_mask).view(batch, 1, -1)

        doc = doc_sen.reshape(batch, max_sen * doc_len, -1)
        word_sim = self.word_similarity(que_vec, doc).view(batch * max_sen, doc_len)
        doc = doc_sen.reshape(batch * max_sen, doc_len, -1)
        doc_mask = doc_sen_mask.reshape(batch * max_sen, doc_len)
        word_hidden = rep_layers.masked_softmax(word_sim, doc_mask, dim=1).unsqueeze(1).bmm(doc)

        word_hidden = word_hidden.view(batch, max_sen, -1)

        doc_vecs = self.doc_sen_self_attn(doc, doc_mask).view(batch, max_sen, -1)

        sentence_sim = self.vector_similarity(que_vec, doc_vecs)
        sentence_alpha = rep_layers.masked_softmax(sentence_sim, sentence_mask)
        sentence_hidden = sentence_alpha.bmm(word_hidden).squeeze(1)
        choice_logits = self.classifier(torch.cat([sentence_hidden, que_vec.squeeze(1)], dim=1)).reshape(-1, self.num_choices)

        if self.training:
            output_dict = {}
        else:
            output_dict = {
                'choice_logits': torch.softmax(choice_logits, dim=-1).detach().cpu().float(),
                'sentence_logits': sentence_sim.reshape(choice_logits.size(0), self.num_choices, max_sen).detach().cpu().float(),
                'sentence_mask': sentence_mask.reshape(choice_logits.size(0), self.num_choices, max_sen).detach().cpu().float()
            }

        if labels is not None:
            choice_loss = functional.cross_entropy(choice_logits, labels)
            loss = choice_loss
            if sentence_ids is not None:
                sentence_loss = self.get_batch_evidence_loss(sentence_sim, sentence_mask, sentence_ids)
                # print(sentence_loss)
                loss += self.evidence_lam * sentence_loss
                # output_dict['sentence_loss'] = sentence_loss
            output_dict['loss'] = loss
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
        batch, max_sentence_num = sentence_mask.size()
        loss = [self.get_evidence_loss(x, y, z) for x, y, z in zip(sentence_sim, sentence_mask, sentence_ids) if z]
        if not loss:
            loss = 0.
        else:
            loss = -sum(loss) / len(loss)
        return loss


class BertRACEHierarchicalTwoViewTopK(BertPreTrainedModel):
    def __init__(self, config, evidence_lambda=0.8, num_choices=4, view_id=1):
        super(BertRACEHierarchicalTwoViewTopK, self).__init__(config)
        logger.info(f'The model {self.__class__.__name__} is loading...')
        logger.info(f'Currently the number of choices is {num_choices}')
        logger.info(f'The coefficient of evidence loss is {evidence_lambda}')
        logger.info(f'The view id of current model is {view_id}')

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
        self.classifier = nn.Linear(config.hidden_size * 2, 1)
        self.evidence_lam = evidence_lambda
        self.num_choices = num_choices
        self.view_id = view_id

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sentence_span_list=None, sentence_ids=None,
                max_sentences=0):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        seq_output, _ = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
        origin_hidden_size = int(seq_output.size(2) / 2)
        seq_output = seq_output[:, :, self.view_id * origin_hidden_size: (self.view_id + 1) * origin_hidden_size]

        # mask: 0 for masked value and 1 for true value
        # doc, que, doc_mask, que_mask = layers.split_doc_que(sequence_output, token_type_ids, attention_mask)
        doc_sen, que, doc_sen_mask, que_mask, sentence_mask = \
            rep_layers.split_doc_sen_que(seq_output, flat_token_type_ids, flat_attention_mask, sentence_span_list,
                                         max_sentences=max_sentences)
        # doc_sen_mask = 1 - doc_sen_mask
        # que_mask = 1 - que_mask
        # sentence_mask = 1 - sentence_mask
        # assert doc_sen.sum() != torch.nan
        batch, max_sen, doc_len = doc_sen_mask.size()
        # assert max_sen == max_sentences

        que_vec = self.que_self_attn(que, que_mask).view(batch, 1, -1)

        doc = doc_sen.reshape(batch, max_sen * doc_len, -1)
        word_sim = self.word_similarity(que_vec, doc).view(batch * max_sen, doc_len)
        doc = doc_sen.reshape(batch * max_sen, doc_len, -1)
        doc_mask = doc_sen_mask.reshape(batch * max_sen, doc_len)
        word_hidden = rep_layers.masked_softmax(word_sim, doc_mask, dim=1).unsqueeze(1).bmm(doc)

        word_hidden = word_hidden.view(batch, max_sen, -1)

        doc_vecs = self.doc_sen_self_attn(doc, doc_mask).view(batch, max_sen, -1)

        sentence_sim = self.vector_similarity(que_vec, doc_vecs)
        sentence_alpha = rep_layers.masked_softmax(sentence_sim, sentence_mask)
        sentence_hidden = sentence_alpha.bmm(word_hidden).squeeze(1)
        choice_logits = self.classifier(torch.cat([sentence_hidden, que_vec.squeeze(1)], dim=1)).reshape(-1, self.num_choices)

        if self.training:
            output_dict = {}
        else:
            output_dict = {
                'choice_logits': torch.softmax(choice_logits, dim=-1).detach().cpu().float(),
                # 'sentence_logits': sentence_alpha.reshape(choice_logits.size(0), self.num_choices, max_sen).detach().cpu().float()
                'sentence_logits': sentence_sim.reshape(choice_logits.size(0), self.num_choices, max_sen).detach().cpu().float(),
                'sentence_mask': sentence_mask.reshape(choice_logits.size(0), self.num_choices, max_sen).detach().cpu().float()
            }

        if labels is not None:
            choice_loss = functional.cross_entropy(choice_logits, labels)
            loss = choice_loss
            if sentence_ids is not None:
                sentence_loss = self.get_batch_evidence_loss(sentence_sim, sentence_mask, sentence_ids)
                loss += self.evidence_lam * sentence_loss
                # output_dict['sentence_loss'] = sentence_loss
            output_dict['loss'] = loss
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
        batch, max_sentence_num = sentence_mask.size()
        loss = [self.get_evidence_loss(x, y, z) for x, y, z in zip(sentence_sim, sentence_mask, sentence_ids) if z]
        if not loss:
            loss = 0.
        else:
            loss = -sum(loss) / len(loss)
        return loss
