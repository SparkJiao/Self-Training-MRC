import logging

import torch
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertLayer, BertEncoder
from torch.nn import functional
from torch import nn

from . import rep_layers as layers

logger = logging.getLogger(__name__)


class BertSplitPreTrainedModel(BertPreTrainedModel):
    def __init__(self, config, my_dropout_p):
        super(BertSplitPreTrainedModel, self).__init__(config)
        logger.info(f'Model {__class__.__name__} is loading...')
        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(my_dropout_p)
        self.bert = BertModel(config)
        self.query_self_attn = layers.MultiHeadPooling(config.hidden_size, 6)
        self.value_self_attn = layers.MultiHeadPooling(config.hidden_size, 6)
        self.sentence_input = layers.BertSentInput(config)
        self.sentence_encoder = BertLayer(config)
        self.attention_score = layers.AttentionScore(config.hidden_size, 256)

    def forward(self, input_ids, attention_mask, token_type_ids, sentence_span_list, label):
        batch, max_seq_length = input_ids.size()
        seq_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        doc, que, doc_mask, que_mask, sentence_mask = layers.split_doc_sen_que(seq_output, token_type_ids, attention_mask,
                                                                               sentence_span_list)
        _, max_sentence_num, max_sent_len, _ = doc.size()
        doc = doc.reshape(batch * max_sentence_num, max_sent_len, -1)
        doc_mask = doc_mask.reshape(batch * max_sentence_num, max_sent_len)
        que = self.query_self_attn(que, que_mask).unsqueeze(1)
        doc = self.value_self_attn(doc, doc_mask).reshape(batch, max_sentence_num, -1)
        doc_input = self.sentence_input(doc, self.bert.embeddings.position_embeddings)
        doc = self.sentence_encoder(doc_input, layers.extended_bert_attention_mask(sentence_mask, dtype=doc_input.dtype))

        attention_score = self.attention_score(que, doc).squeeze(1)
        # masked_attention_score = layers.masked_log_softmax(attention_score, sentence_mask, dim=-1).squeeze(1)

        # loss = functional.nll_loss(masked_attention_score, label.reshape(batch), ignore_index=-1)
        loss = functional.cross_entropy(attention_score, label.reshape(batch), ignore_index=-1)

        output = {
            'scores': attention_score.float(),
            'loss': loss
        }
        return output


class BertHierarchicalTransformer(BertPreTrainedModel):
    def __init__(self, config, evidence_lambda: float = 0.8, my_dropout_p: float = 0.2, tf_layers: int = 1,
                 tf_inter_size: int = 3072):
        super(BertHierarchicalTransformer, self).__init__(config)
        logger.info(f'Model {__class__.__name__} is loading...')
        logger.info(f'Model parameters:')
        logger.info(f'Evidence lambda: {evidence_lambda}')
        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(my_dropout_p)
        self.bert = BertModel(config)
        self.query_self_attn = layers.MultiHeadPooling(config.hidden_size, 6)
        self.value_self_attn = layers.MultiHeadPooling(config.hidden_size, 6)
        # self.sentence_input = layers.BertSentInput(config)
        config.num_hidden_layers = tf_layers
        config.intermediate_size = tf_inter_size
        self.sentence_encoder = BertEncoder(config)
        self.attention_score = layers.AttentionScore(config.hidden_size, 256)

        # Output layer
        self.evidence_lambda = evidence_lambda
        self.predictor = nn.Linear(config.hidden_size * 2, 3)

    def forward(self, input_ids, attention_mask, token_type_ids, sentence_span_list, answer_choice=None,
                sentence_ids=None):
        batch, max_seq_length = input_ids.size()
        seq_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        doc, que, doc_mask, que_mask, sentence_mask = layers.split_doc_sen_que(seq_output, token_type_ids, attention_mask,
                                                                               sentence_span_list)
        _, max_sentence_num, max_sent_len, _ = doc.size()
        doc = doc.reshape(batch * max_sentence_num, max_sent_len, -1)
        doc_mask = doc_mask.reshape(batch * max_sentence_num, max_sent_len)
        que = self.query_self_attn(que, que_mask).unsqueeze(1)
        doc = self.value_self_attn(doc, doc_mask).reshape(batch, max_sentence_num, -1)
        # doc_input = self.sentence_input(doc, self.bert.embeddings.position_embeddings)
        # doc = self.sentence_encoder(doc_input, layers.extended_bert_attention_mask(sentence_mask, dtype=doc_input.dtype))[-1]
        doc = self.sentence_encoder(doc, layers.extended_bert_attention_mask(sentence_mask, dtype=doc.dtype))[-1]

        attention_score = self.attention_score(que, doc)
        attended_doc = layers.masked_softmax(attention_score, sentence_mask, dim=-1).bmm(doc).squeeze(1)
        attention_score = attention_score.squeeze(1)
        que = que.squeeze(1)
        choice_logits = self.predictor(torch.cat([que, attended_doc], dim=1))

        output_dict = {'yesno_logits': choice_logits,
                       'sentence_logits': attention_score,
                       'max_weight_index': attention_score.max(dim=1)[1],
                       'max_weight': attention_score.max(dim=1)[0]}
        loss = 0
        if answer_choice is not None:
            choice_loss = functional.cross_entropy(choice_logits, answer_choice, ignore_index=-1)
            loss += choice_loss
        if sentence_ids is not None:
            sentence_loss = self.evidence_lambda * functional.cross_entropy(attention_score, sentence_ids, ignore_index=-1)
            loss += sentence_loss
        output_dict['loss'] = loss
        return output_dict


class BertHierarchicalTransformer1(BertPreTrainedModel):
    def __init__(self, config, evidence_lambda: float = 0.8, my_dropout_p: float = 0.2, tf_layers: int = 1,
                 tf_inter_size: int = 3072):
        super(BertHierarchicalTransformer1, self).__init__(config)
        logger.info(f'Model {__class__.__name__} is loading...')
        logger.info(f'Model parameters:')
        logger.info(f'Evidence lambda: {evidence_lambda}')
        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(my_dropout_p)
        self.bert = BertModel(config)
        self.query_self_attn = layers.MultiHeadPooling1(config.hidden_size, 6)
        self.value_self_attn = layers.MultiHeadPooling1(config.hidden_size, 6)
        # self.sentence_input = layers.BertSentInput(config)
        config.num_hidden_layers = tf_layers
        config.intermediate_size = tf_inter_size
        self.sentence_encoder = BertEncoder(config)
        self.attention_score = layers.AttentionScore(config.hidden_size, 256)

        # Output layer
        self.evidence_lambda = evidence_lambda
        self.predictor = nn.Linear(config.hidden_size * 2, 3)

    def forward(self, input_ids, attention_mask, token_type_ids, sentence_span_list, answer_choice=None,
                sentence_ids=None):
        batch, max_seq_length = input_ids.size()
        seq_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        doc, que, doc_mask, que_mask, sentence_mask = layers.split_doc_sen_que(seq_output, token_type_ids, attention_mask,
                                                                               sentence_span_list)
        _, max_sentence_num, max_sent_len, _ = doc.size()
        doc = doc.reshape(batch * max_sentence_num, max_sent_len, -1)
        doc_mask = doc_mask.reshape(batch * max_sentence_num, max_sent_len)
        que = self.query_self_attn(que, que_mask).unsqueeze(1)
        doc = self.value_self_attn(doc, doc_mask).reshape(batch, max_sentence_num, -1)
        # doc_input = self.sentence_input(doc, self.bert.embeddings.position_embeddings)
        # doc = self.sentence_encoder(doc_input, layers.extended_bert_attention_mask(sentence_mask, dtype=doc_input.dtype))[-1]
        doc = self.sentence_encoder(doc, layers.extended_bert_attention_mask(sentence_mask, dtype=doc.dtype))[-1]

        attention_score = self.attention_score(que, doc)
        attended_doc = layers.masked_softmax(attention_score, sentence_mask, dim=-1).bmm(doc).squeeze(1)
        attention_score = attention_score.squeeze(1)
        que = que.squeeze(1)
        choice_logits = self.predictor(torch.cat([que, attended_doc], dim=1))

        output_dict = {'yesno_logits': choice_logits,
                       'sentence_logits': attention_score,
                       'max_weight_index': attention_score.max(dim=1)[1],
                       'max_weight': attention_score.max(dim=1)[0]}
        loss = 0
        if answer_choice is not None:
            choice_loss = functional.cross_entropy(choice_logits, answer_choice, ignore_index=-1)
            loss += choice_loss
        if sentence_ids is not None:
            sentence_loss = self.evidence_lambda * functional.cross_entropy(attention_score, sentence_ids, ignore_index=-1)
            loss += sentence_loss
        output_dict['loss'] = loss
        return output_dict


class BertHierarchicalRNN(BertPreTrainedModel):
    def __init__(self, config, evidence_lambda: float = 0.8, my_dropout_p: float = 0.2):
        super(BertHierarchicalRNN, self).__init__(config)
        logger.info(f'Model {__class__.__name__} is loading...')
        logger.info(f'Model parameters:')
        logger.info(f'Evidence lambda: {evidence_lambda}')
        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(my_dropout_p)
        self.bert = BertModel(config)
        self.query_self_attn = layers.MultiHeadPooling(config.hidden_size, 6)
        self.value_self_attn = layers.MultiHeadPooling(config.hidden_size, 6)
        self.sentence_encoder = layers.ConcatRNN(config.hidden_size, config.hidden_size // 2,
                                                 num_layers=1, bidirectional=True, rnn_type='lstm')
        self.attention_score = layers.AttentionScore(config.hidden_size, 256)

        # Output layer
        self.evidence_lambda = evidence_lambda
        self.predictor = nn.Linear(config.hidden_size * 2, 3)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, attention_mask, token_type_ids, sentence_span_list, answer_choice=None,
                sentence_ids=None):
        batch, max_seq_length = input_ids.size()
        seq_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        doc, que, doc_mask, que_mask, sentence_mask = layers.split_doc_sen_que(seq_output, token_type_ids, attention_mask,
                                                                               sentence_span_list)
        _, max_sentence_num, max_sent_len, _ = doc.size()
        doc = doc.reshape(batch * max_sentence_num, max_sent_len, -1)
        doc_mask = doc_mask.reshape(batch * max_sentence_num, max_sent_len)
        que = self.query_self_attn(que, que_mask).unsqueeze(1)
        doc = self.value_self_attn(doc, doc_mask).reshape(batch, max_sentence_num, -1)
        doc = self.sentence_encoder(doc, sentence_mask)

        attention_score = self.attention_score(que, doc)
        attended_doc = layers.masked_softmax(attention_score, sentence_mask, dim=-1).bmm(doc).squeeze(1)
        attention_score = attention_score.squeeze(1)
        que = que.squeeze(1)
        choice_logits = self.predictor(torch.cat([que, attended_doc], dim=1))

        output_dict = {'yesno_logits': choice_logits,
                       'sentence_logits': attention_score,
                       'max_weight_index': attention_score.max(dim=1)[1],
                       'max_weight': attention_score.max(dim=1)[0]}
        loss = 0
        if answer_choice is not None:
            choice_loss = functional.cross_entropy(choice_logits, answer_choice, ignore_index=-1)
            loss += choice_loss
        if sentence_ids is not None:
            sentence_loss = self.evidence_lambda * functional.cross_entropy(attention_score, sentence_ids, ignore_index=-1)
            loss += sentence_loss
        output_dict['loss'] = loss
        return output_dict


class BertHierarchicalRNN1(BertPreTrainedModel):
    def __init__(self, config, evidence_lambda: float = 0.8, my_dropout_p: float = 0.2):
        super(BertHierarchicalRNN1, self).__init__(config)
        logger.info(f'Model {__class__.__name__} is loading...')
        logger.info(f'Model parameters:')
        logger.info(f'Evidence lambda: {evidence_lambda}')
        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(my_dropout_p)
        self.bert = BertModel(config)
        self.query_self_attn = layers.MultiHeadPooling1(config.hidden_size, 6)
        self.value_self_attn = layers.MultiHeadPooling1(config.hidden_size, 6)
        self.sentence_encoder = layers.ConcatRNN(config.hidden_size, config.hidden_size // 2,
                                                 num_layers=1, bidirectional=True, rnn_type='lstm')
        self.attention_score = layers.AttentionScore(config.hidden_size, 256)

        # Output layer
        self.evidence_lambda = evidence_lambda
        self.predictor = nn.Linear(config.hidden_size * 2, 3)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, attention_mask, token_type_ids, sentence_span_list, answer_choice=None,
                sentence_ids=None):
        batch, max_seq_length = input_ids.size()
        seq_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        doc, que, doc_mask, que_mask, sentence_mask = layers.split_doc_sen_que(seq_output, token_type_ids, attention_mask,
                                                                               sentence_span_list)
        _, max_sentence_num, max_sent_len, _ = doc.size()
        doc = doc.reshape(batch * max_sentence_num, max_sent_len, -1)
        doc_mask = doc_mask.reshape(batch * max_sentence_num, max_sent_len)
        que = self.query_self_attn(que, que_mask).unsqueeze(1)
        doc = self.value_self_attn(doc, doc_mask).reshape(batch, max_sentence_num, -1)
        doc = self.sentence_encoder(doc, sentence_mask)

        attention_score = self.attention_score(que, doc)
        attended_doc = layers.masked_softmax(attention_score, sentence_mask, dim=-1).bmm(doc).squeeze(1)
        attention_score = attention_score.squeeze(1)
        que = que.squeeze(1)
        choice_logits = self.predictor(torch.cat([que, attended_doc], dim=1))

        output_dict = {'yesno_logits': choice_logits,
                       'sentence_logits': attention_score,
                       'max_weight_index': attention_score.max(dim=1)[1],
                       'max_weight': attention_score.max(dim=1)[0]}
        loss = 0
        if answer_choice is not None:
            choice_loss = functional.cross_entropy(choice_logits, answer_choice, ignore_index=-1)
            loss += choice_loss
        if sentence_ids is not None:
            sentence_loss = self.evidence_lambda * functional.cross_entropy(attention_score, sentence_ids, ignore_index=-1)
            loss += sentence_loss
        output_dict['loss'] = loss
        return output_dict


class BertHierarchicalRNN2(BertPreTrainedModel):
    def __init__(self, config, evidence_lambda: float = 0.8, my_dropout_p: float = 0.2):
        super(BertHierarchicalRNN2, self).__init__(config)
        logger.info(f'Model {__class__.__name__} is loading...')
        logger.info(f'Model parameters:')
        logger.info(f'Evidence lambda: {evidence_lambda}')
        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(my_dropout_p)
        self.bert = BertModel(config)
        self.query_self_attn = layers.MultiHeadPooling2(config.hidden_size, 6)
        self.value_self_attn = layers.MultiHeadPooling2(config.hidden_size, 6)
        self.sentence_encoder = layers.ConcatRNN(config.hidden_size, config.hidden_size // 2,
                                                 num_layers=1, bidirectional=True, rnn_type='lstm')
        self.attention_score = layers.AttentionScore(config.hidden_size, 256)

        # Output layer
        self.evidence_lambda = evidence_lambda
        self.predictor = nn.Linear(config.hidden_size * 2, 3)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, attention_mask, token_type_ids, sentence_span_list, answer_choice=None,
                sentence_ids=None):
        batch, max_seq_length = input_ids.size()
        seq_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        doc, que, doc_mask, que_mask, sentence_mask = layers.split_doc_sen_que(seq_output, token_type_ids, attention_mask,
                                                                               sentence_span_list)
        _, max_sentence_num, max_sent_len, _ = doc.size()
        doc = doc.reshape(batch * max_sentence_num, max_sent_len, -1)
        doc_mask = doc_mask.reshape(batch * max_sentence_num, max_sent_len)
        que = self.query_self_attn(que, que_mask).unsqueeze(1)
        doc = self.value_self_attn(doc, doc_mask).reshape(batch, max_sentence_num, -1)
        doc = self.sentence_encoder(doc, sentence_mask)

        attention_score = self.attention_score(que, doc)
        attended_doc = layers.masked_softmax(attention_score, sentence_mask, dim=-1).bmm(doc).squeeze(1)
        attention_score = attention_score.squeeze(1)
        que = que.squeeze(1)
        choice_logits = self.predictor(torch.cat([que, attended_doc], dim=1))

        output_dict = {'yesno_logits': choice_logits,
                       'sentence_logits': attention_score,
                       'max_weight_index': attention_score.max(dim=1)[1],
                       'max_weight': attention_score.max(dim=1)[0]}
        loss = 0
        if answer_choice is not None:
            choice_loss = functional.cross_entropy(choice_logits, answer_choice, ignore_index=-1)
            loss += choice_loss
        if sentence_ids is not None:
            sentence_loss = self.evidence_lambda * functional.cross_entropy(attention_score, sentence_ids, ignore_index=-1)
            loss += sentence_loss
        output_dict['loss'] = loss
        return output_dict
