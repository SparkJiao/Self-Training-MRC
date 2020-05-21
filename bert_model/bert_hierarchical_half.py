import torch
from allennlp.nn.util import masked_softmax, masked_log_softmax
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch import nn
from torch.nn import functional as F

from bert_model import layers


class BertQAYesnoHierarchicalNegHalf(BertPreTrainedModel):
    """
    BertForQuestionAnsweringForYesNo

    Model Hierarchical Attention:
        - Use Hierarchical attention module to predict Non/Yes/No.
        - Add supervised to sentence attention.

    This model is just to test that, if we use only the half of hidden vectors of bert, how will the final model perform.
    """

    def __init__(self, config, evidence_lambda=0.8, negative_lambda=1.0, add_entropy: bool = False,
                 split_num: int = 3, split_index: int = 0):
        super(BertQAYesnoHierarchicalNegHalf, self).__init__(config)
        print(f'The model {self.__class__.__name__} is loading...')
        print(f'The coefficient of evidence loss is {evidence_lambda}')
        print(f'The coefficient of negative samples loss is {negative_lambda}')
        print(f'Add entropy loss: {add_entropy}')

        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(config.hidden_dropout_prob)

        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.answer_choice = nn.Linear(config.hidden_size, 2)
        self.split_num = split_num
        self.split_size = config.hidden_size // self.split_num
        self.split_index = split_index
        self.split_sizes = [((i - 1) * self.split_size, i * self.split_size) for i in range(1, split_num)] + [
            ((split_num - 1) * self.split_size, config.hidden_size)]
        print(f'Split BERT output into {self.split_num}.')
        print(f'Current model use the {self.split_index}th hidden state.')
        print(f'Read hidden state in {self.split_sizes[self.split_index]}.')

        self.doc_sen_self_attn = layers.LinearSelfAttnAllennlp(self.split_size)
        self.que_self_attn = layers.LinearSelfAttn(self.split_size)

        self.word_similarity = layers.AttentionScore(self.split_size, 250, do_similarity=False)
        self.vector_similarity = layers.AttentionScore(self.split_size, 250, do_similarity=False)

        # self.yesno_predictor = nn.Linear(config.hidden_size, 2)
        self.yesno_predictor = nn.Linear(self.split_size * 2, 3)
        self.evidence_lam = evidence_lambda
        self.negative_lam = negative_lambda
        self.add_entropy = add_entropy

        self.apply(self.init_bert_weights)

    def select_separate_hidden(self, hidden_state):
        hidden_start, hidden_end = self.split_sizes[self.split_index]
        return hidden_state[:, :, hidden_start: hidden_end]

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, answer_choice=None,
                sentence_span_list=None, sentence_ids=None, sentence_label=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.select_separate_hidden(sequence_output)

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

        # [batch, 1, max_sen]
        sentence_sim = self.vector_similarity(que_vec, doc_vecs)
        sentence_scores = masked_softmax(sentence_sim, 1 - sentence_mask)
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
            log_sentence_sim = masked_log_softmax(sentence_sim.squeeze(1), 1 - sentence_mask, dim=-1)
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
