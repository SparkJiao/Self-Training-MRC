import logging

from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertLayer
from torch import nn
from torch.nn import functional

from . import layers

logger = logging.getLogger(__name__)


class BertSentencePreTrainedModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSentencePreTrainedModel, self).__init__(config)
        logger.info(f'Model {__class__.__name__} is loading...')
        # layers.set_seq_dropout(True)
        # layers.set_my_dropout_prob(my_dropout_p)
        self.bert = BertModel(config)
        self.bert_sent_input = layers.BertSentInput(config)
        config.intermediate_size = 1024
        config.num_attention_heads = 6
        self.bert_layer = BertLayer(config)
        self.sent_label_predictor = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, input_mask, sentence_mask, sentence_labels):
        batch, max_sen_num, max_seq_len = input_ids.size()
        input_ids = input_ids.reshape(batch * max_sen_num, max_seq_len)
        input_mask = input_mask.reshape(batch * max_sen_num, max_seq_len)
        seq_output, _ = self.bert(input_ids, attention_mask=input_mask, output_all_encoded_layers=False)
        seq_output = seq_output.reshape(batch, max_sen_num, max_seq_len, -1)
        sent_hidden = seq_output[:, :, 0]
        sent_input = self.bert_sent_input(sent_hidden, self.bert.embeddings.position_embeddings)
        sent_output = self.bert_layer(sent_input, layers.extended_bert_attention_mask(sentence_mask, next(self.parameters()).dtype))
        sent_label = self.sent_label_predictor(sent_output)

        loss = functional.cross_entropy(sent_label.view(batch * max_sen_num, 2), sentence_labels.reshape(batch * max_sen_num),
                                        ignore_index=-1, reduction='sum') / sent_label.size(0)

        output = {
            'scores': sent_label,
            'loss': loss
        }
        return output


class BertSentencePreTrainedModel2(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSentencePreTrainedModel2, self).__init__(config)
        logger.info(f'Model {__class__.__name__} is loading...')
        # layers.set_seq_dropout(True)
        # layers.set_my_dropout_prob(my_dropout_p)
        self.bert = BertModel(config)
        self.bert_sent_input = layers.BertSentInput(config)
        config.intermediate_size = 1024
        config.num_attention_heads = 6
        self.bert_layer = BertLayer(config)
        self.sent_label_predictor = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, input_mask, sentence_mask, sentence_labels):
        batch, max_sen_num, max_seq_len = input_ids.size()
        input_ids = input_ids.reshape(batch * max_sen_num, max_seq_len)
        input_mask = input_mask.reshape(batch * max_sen_num, max_seq_len)
        seq_output, _ = self.bert(input_ids, attention_mask=input_mask, output_all_encoded_layers=False)
        seq_output = seq_output.reshape(batch, max_sen_num, max_seq_len, -1)
        sent_hidden = seq_output[:, :, 0]
        sent_input = self.bert_sent_input(sent_hidden, self.bert.embeddings.position_embeddings)
        sent_output = self.bert_layer(sent_input, layers.extended_bert_attention_mask(sentence_mask, next(self.parameters()).dtype))
        sent_label = self.sent_label_predictor(sent_output)

        loss = functional.cross_entropy(sent_label.view(batch, max_sen_num), sentence_labels.reshape(batch),
                                        ignore_index=-1)

        output = {
            'scores': sent_label,
            'loss': loss
        }
        return output
