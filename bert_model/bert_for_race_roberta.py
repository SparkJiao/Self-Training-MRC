import torch
from torch import nn
from torch.nn import functional
from transformers.modeling_roberta import RobertaModel, BertPreTrainedModel

from general_util.logger import get_child_logger
from . import layers
from . import rep_layers

logger = get_child_logger(__name__)


class RobertaRACEHierarchicalTopK(BertPreTrainedModel):
    def __init__(self, config, evidence_lambda=0.8, num_choices=4):
        super(RobertaRACEHierarchicalTopK, self).__init__(config)
        logger.info(f'The model {self.__class__.__name__} is loading...')
        logger.info(f'Currently the number of choices is {num_choices}')
        logger.info(f'The coefficient of evidence loss is {evidence_lambda}')

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
        self.classifier = nn.Linear(config.hidden_size * 2, 1)
        self.evidence_lam = evidence_lambda
        self.num_choices = num_choices

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sentence_span_list=None,
                sentence_ids=None, max_sentences: int = 0):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # Roberta doesn't use token_type_ids anymore. the token_type_ids is only used for split sentence.
        outputs = self.roberta(flat_input_ids, attention_mask=flat_attention_mask, token_type_ids=None)
        seq_output = outputs[0]

        # mask: 1 for masked value and 0 for true value
        # doc, que, doc_mask, que_mask = layers.split_doc_que(sequence_output, token_type_ids, attention_mask)

        doc_sen, que, doc_sen_mask, que_mask, sentence_mask = \
            rep_layers.split_doc_sen_que_roberta(seq_output, flat_token_type_ids, flat_attention_mask, sentence_span_list,
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
