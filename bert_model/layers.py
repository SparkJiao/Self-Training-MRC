import torch
import torch.nn.functional as F
from allennlp.nn import util
from torch import nn
from torch.nn.parameter import Parameter
from .rep_layers import masked_softmax

from pytorch_pretrained_bert.modeling import BertLayerNorm

""" Dropout Module"""


# Before using dropout, call set_seq_dropout and set_my_dropout_prob first in the __init__() of your own model
def set_seq_dropout(option):  # option = True or False
    global do_seq_dropout
    do_seq_dropout = option


def set_my_dropout_prob(p):  # p between 0 to 1
    global my_dropout_p
    my_dropout_p = p


def seq_dropout(x, p=0, training=False):
    """
    x: batch * len * input_size
    """
    if training is False or p == 0:
        return x
    dropout_mask = 1.0 / (1 - p) * torch.bernoulli((1 - p) * (x.new_zeros(x.size(0), x.size(2)) + 1))
    return dropout_mask.unsqueeze(1).expand_as(x) * x


def dropout(x, p=0, training=False):
    """
    x: (batch * len * input_size) or (any other shape)
    """
    if do_seq_dropout and len(x.size()) == 3:  # if x is (batch * len * input_size)
        return seq_dropout(x, p=p, training=training)
    else:
        return F.dropout(x, p=p, training=training)


""" Sub Module """


class StackedBRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, rnn_type=nn.LSTM, concat_layers=False, do_residual=False, add_feat=0,
                 dialog_flow=False, bidir=True):
        super(StackedBRNN, self).__init__()
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.do_residual = do_residual
        self.dialog_flow = dialog_flow
        self.hidden_size = hidden_size

        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            if i != 0:
                if not bidir:
                    input_size = hidden_size
                else:
                    input_size = 2 * hidden_size
                if i == 1:
                    input_size += add_feat
            # input_size = input_size if i == 0 else (2 * hidden_size + add_feat if i == 1 else 2 * hidden_size)
            if self.dialog_flow:
                input_size += 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size, num_layers=1, bidirectional=bidir))

    def forward(self, x, x_mask=None, return_list=False, additional_x=None, previous_hiddens=None):
        # return_list: return a list for layers of hidden vectors
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)
        if additional_x is not None:
            additional_x = additional_x.transpose(0, 1)

        # Encode all layers
        hiddens = [x]
        for i in range(self.num_layers):
            rnn_input = hiddens[-1]
            if i == 1 and additional_x is not None:
                rnn_input = torch.cat((rnn_input, additional_x), 2)
            # Apply dropout to input
            if my_dropout_p > 0:
                rnn_input = dropout(rnn_input, p=my_dropout_p, training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            if self.do_residual and i > 0:
                rnn_output = rnn_output + hiddens[-1]
            hiddens.append(rnn_output)

        # Transpose back
        hiddens = [h.transpose(0, 1) for h in hiddens]

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(hiddens[1:], 2)
        else:
            output = hiddens[-1]

        if return_list:
            return output, hiddens[1:]
        else:
            return output


# For summarizing a set of vectors into a single vector
class LinearSelfAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSelfAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x = dropout(x, p=my_dropout_p, training=self.training)

        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=1)
        return alpha


class LinearSelfAttnAllennlp(nn.Module):
    """
    This module use allennlp.nn.utils.masked_softmax to avoid NAN while all values are masked.
    The input mask is 1 for masked value and 0 for true value.
    """

    def __init__(self, input_size):
        super(LinearSelfAttnAllennlp, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        x = dropout(x, p=my_dropout_p, training=self.training)

        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        # alpha = util.masked_softmax(scores, 1 - x_mask, dim=1)
        alpha = masked_softmax(scores, 1 - x_mask, dim=1)
        return alpha


class AttentionScore(nn.Module):
    """
    correlation_func = 1, sij = x1^Tx2
    correlation_func = 2, sij = (Wx1)D(Wx2)
    correlation_func = 3, sij = Relu(Wx1)DRelu(Wx2)
    correlation_func = 4, sij = x1^TWx2
    correlation_func = 5, sij = Relu(Wx1)DRelu(Wx2)
    correlation_func = 6: sij = Relu(W1x1)^TRelu(W2x2)
    """

    def __init__(self, input_size, hidden_size, do_similarity=False, correlation_func=3):
        super(AttentionScore, self).__init__()
        self.correlation_func = correlation_func
        self.hidden_size = hidden_size

        if correlation_func == 2 or correlation_func == 3:
            self.linear = nn.Linear(input_size, hidden_size, bias=False)
            if do_similarity:
                self.diagonal = Parameter(torch.ones(1, 1, 1) / (hidden_size ** 0.5), requires_grad=False)
            else:
                self.diagonal = Parameter(torch.ones(1, 1, hidden_size), requires_grad=True)

        if correlation_func == 4:
            self.linear = nn.Linear(input_size, input_size, bias=False)

        if correlation_func == 5:
            self.linear = nn.Linear(input_size, hidden_size, bias=False)
        if correlation_func == 6:
            self.linear1 = nn.Linear(input_size, hidden_size, bias=False)
            self.linear2 = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, x1, x2):
        '''
        Input:
        x1: batch x word_num1 x dim
        x2: batch x word_num2 x dim
        Output:
        scores: batch x word_num1 x word_num2
        '''
        x1 = dropout(x1, p=my_dropout_p, training=self.training)
        x2 = dropout(x2, p=my_dropout_p, training=self.training)

        x1_rep = x1
        x2_rep = x2
        batch = x1_rep.size(0)
        word_num1 = x1_rep.size(1)
        word_num2 = x2_rep.size(1)
        dim = x1_rep.size(2)
        if self.correlation_func == 2 or self.correlation_func == 3:
            x1_rep = self.linear(x1_rep.contiguous().view(-1, dim)).view(batch, word_num1, self.hidden_size)  # Wx1
            x2_rep = self.linear(x2_rep.contiguous().view(-1, dim)).view(batch, word_num2, self.hidden_size)  # Wx2
            if self.correlation_func == 3:
                x1_rep = F.relu(x1_rep)
                x2_rep = F.relu(x2_rep)
            x1_rep = x1_rep * self.diagonal.expand_as(x1_rep)
            # x1_rep is (Wx1)D or Relu(Wx1)D
            # x1_rep: batch x word_num1 x dim (corr=1) or hidden_size (corr=2,3)

        if self.correlation_func == 4:
            x2_rep = self.linear(x2_rep.contiguous().view(-1, dim)).view(batch, word_num2, dim)  # Wx2

        if self.correlation_func == 5:
            x1_rep = self.linear(x1_rep.contiguous().view(-1, dim)).view(batch, word_num1, self.hidden_size)  # Wx1
            x2_rep = self.linear(x2_rep.contiguous().view(-1, dim)).view(batch, word_num2, self.hidden_size)  # Wx2
            x1_rep = F.relu(x1_rep)
            x2_rep = F.relu(x2_rep)

        if self.correlation_func == 6:
            x1_rep = self.linear1(x1_rep.contiguous().view(-1, dim)).view(batch, word_num1, self.hidden_size)
            x2_rep = self.linear2(x2_rep.contiguous().view(-1, dim)).view(batch, word_num2, self.hidden_size)
            x1_rep = F.relu(x1_rep)
            x2_rep = F.relu(x2_rep)

        scores = x1_rep.bmm(x2_rep.transpose(1, 2))
        return scores


class GetAttentionHiddens(nn.Module):
    def __init__(self, input_size, attention_hidden_size, similarity_attention=False, correlation_func=3):
        super(GetAttentionHiddens, self).__init__()
        self.scoring = AttentionScore(input_size, attention_hidden_size, do_similarity=similarity_attention,
                                      correlation_func=correlation_func)

    def forward(self, x1, x2, x2_mask=None, x3=None, scores=None, return_scores=False, drop_diagonal=False, extra_scores=None):
        """
        Using x1, x2 to calculate attention score, but x1 will take back info from x3.
        If x3 is not specified, x1 will attend on x2.

        x1: batch * len1 * x1_input_size
        x2: batch * len2 * x2_input_size
        x2_mask: batch * len2

        x3: batch * len2 * x3_input_size (or None)
        """
        if x3 is None:
            x3 = x2

        if scores is None:
            scores = self.scoring(x1, x2)

        if extra_scores is not None:
            scores += extra_scores

        # Mask padding
        if x2_mask is not None:
            # x2_mask = x2_mask.unsqueeze(1).expand_as(scores)
            x2_mask = x2_mask.unsqueeze(1)
            scores.data.masked_fill_(x2_mask.data, -float('inf'))
        if drop_diagonal:
            assert (scores.size(1) == scores.size(2))
            diag_mask = torch.diag(scores.data.new(scores.size(1)).zero_() + 1).byte().unsqueeze(0).expand_as(scores)
            scores.data.masked_fill_(diag_mask, -float('inf'))

        # Normalize with softmax
        alpha = F.softmax(scores, dim=2)

        # Take weighted average
        matched_seq = alpha.bmm(x3)
        if return_scores:
            return matched_seq, scores
        else:
            return matched_seq


class FusionLayer(nn.Module):
    def __init__(self, input_size, activation_func=nn.Tanh):
        super(FusionLayer, self).__init__()
        self.func_f = nn.Sequential(
            nn.Linear(input_size * 4, input_size),
            activation_func()
        )
        self.func_g = nn.Sequential(
            nn.Linear(input_size * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        z = torch.cat([x, y, x - y, x * y], dim=2)
        z_d = dropout(z, p=my_dropout_p, training=self.training)
        f = self.func_f(z_d)
        g = self.func_g(z_d)
        return f * g + (1 - g) * x


class LocalBiAttention(nn.Module):
    def __init__(self, input_size, attention_hidden_size, do_similarity: bool = False, do_fuse: bool = True):
        super(LocalBiAttention, self).__init__()
        self.attn_score = AttentionScore(input_size, attention_hidden_size, do_similarity=do_similarity)
        self.doc_self_attn = AttentionScore(input_size, attention_hidden_size, do_similarity=do_similarity)
        self.que_self_attn = AttentionScore(input_size, attention_hidden_size, do_similarity=do_similarity)
        self.do_fuse = do_fuse
        if do_fuse:
            self.a_fuse = nn.Sequential(nn.Linear(input_size * 2, attention_hidden_size), nn.ReLU(inplace=True))
            self.b_fuse = nn.Sequential(nn.Linear(input_size * 2, attention_hidden_size), nn.ReLU(inplace=True))

    def forward(self, doc, que, doc_mask, que_mask, distance=1, drop_diagonal=True, ex_doc=None, ex_que=None):
        attn_score = self.attn_score(doc, que)
        doc_self = self.doc_self_attn(doc, doc)
        que_self = self.que_self_attn(que, que)

        batch, doc_len, que_len = attn_score.size()
        device = attn_score.device

        doc_local_mask = self.generate_local_mask(device, batch, doc_len, doc_len, distance=distance, drop_diagonal=drop_diagonal)
        doc_final_mask = (1 - doc_local_mask) * (1 - doc_mask.unsqueeze(1))

        que_local_mask = self.generate_local_mask(device, batch, que_len, que_len, distance=distance, drop_diagonal=drop_diagonal)
        que_final_mask = (1 - que_local_mask) * (1 - que_mask.unsqueeze(1))

        doc_local = util.masked_softmax(doc_self, doc_final_mask).bmm(doc)
        que_local = util.masked_softmax(que_self, que_final_mask).bmm(que)

        que_score_pooling = attn_score.max(dim=1, keepdim=True)[0]
        que_vec = util.masked_softmax(que_score_pooling, que_mask, dim=2).bmm(que_local).squeeze(1)
        doc_score_pooling = attn_score.transpose(1, 2).max(dim=1, keepdim=True)[0]
        doc_vec = util.masked_softmax(doc_score_pooling, doc_mask, dim=2).bmm(doc_local).squeeze(1)

        if self.do_fuse and ex_doc is not None and ex_que is not None:
            doc_h = self.a_fuse(torch.cat([ex_doc, doc_vec], dim=1))
            que_h = self.b_fuse(torch.cat([ex_que, que_vec], dim=1))
            return doc_h, que_h
        else:
            return doc_vec, que_vec

    @staticmethod
    def generate_local_mask(device, batch_size, len1, len2, distance: int, drop_diagonal: bool = False):
        mask = torch.ones(len1, len2, device=device).byte()
        l_mask = mask.tril(distance)
        u_mask = mask.triu(distance)

        final_mask = (1 - l_mask * u_mask).byte()

        if drop_diagonal:
            min_len = min(len1, len2)
            for i in range(min_len):
                final_mask[i][i] = 1

        return final_mask.unsqueeze(0).expand(batch_size, -1, -1)


class BertSentInput(nn.Module):
    def __init__(self, config):
        super(BertSentInput, self).__init__()
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_state, position_embedding):
        batch, seq_length, _ = hidden_state.size()
        position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_state.device)
        position_ids = position_ids.unsqueeze(0).expand(batch, -1)
        position_embeddings = position_embedding(position_ids)

        sent_input = hidden_state + position_embeddings
        sent_input = self.LayerNorm(sent_input)
        sent_input = self.dropout(sent_input)
        return sent_input


""" Functions """


def combine_doc_que(doc, que, doc_mask, que_mask, max_seq_length):
    batch = doc.size(0)

    # 1 for true value and 0 for masked value.
    doc_value_mask = 1 - doc_mask
    que_value_mask = 1 - que_mask

    combined_mask = doc_mask.new_ones(batch, max_seq_length)

    if len(doc.size()) == 3:
        combined_seq = doc.new_zeros(batch, max_seq_length, doc.size(-1))
    else:
        combined_seq = doc.new_zeros(batch, max_seq_length)

    for b in range(batch):
        que_selected = que[b].masked_select(que_value_mask[b])
        que_len = que_selected.size(0)
        combined_seq[b, 1:(1 + que_len)] = que_selected

        doc_selected = doc[b].masked_select(doc_value_mask[b])
        doc_len = doc_selected.size(0)
        combined_seq[b, (2 + que_len):(2 + que_len + doc_len)] = doc_selected

        combined_mask[b, (2 + que_len):(2 + que_len + doc_len)] = doc_mask.new_zeros(doc_len)

    return combined_seq, combined_mask.byte()


def split_doc_que(hidden_state, token_type_ids, attention_mask):
    batch, seq_len, hidden_size = hidden_state.size()

    que_hidden_list = []
    doc_hidden_list = []
    max_doc_len = 0
    max_que_len = 0
    for b in range(batch):
        unmasked_len = torch.sum(attention_mask[b]).item()
        # doc + [SEP]
        doc_len = torch.sum(token_type_ids[b]).item()
        # [CLS] + query + [SEP]
        que_len = unmasked_len - doc_len

        # TODO: the two lines are wrong but won't cause calculation error.
        # max_doc_len = max(max_doc_len, doc_len - 1)
        # max_que_len = max(max_que_len, que_len - 2)
        max_doc_len = max(max_doc_len, doc_len)
        max_que_len = max(max_que_len, que_len)

        que_hidden_list.append(hidden_state[b, 1:(que_len - 1)])
        doc_hidden_list.append(hidden_state[b, que_len:(que_len + doc_len - 1)])

    doc_rep = hidden_state.new_zeros(batch, max_doc_len, hidden_size)
    que_rep = hidden_state.new_zeros(batch, max_que_len, hidden_size)
    doc_mask = attention_mask.new_ones(batch, max_doc_len)
    que_mask = attention_mask.new_ones(batch, max_que_len)

    for b, (doc_h, que_h) in enumerate(zip(doc_hidden_list, que_hidden_list)):
        doc_rep[b, :doc_h.size(0)] = doc_h
        que_rep[b, :que_h.size(0)] = que_h
        doc_mask[b, :doc_h.size(0)] = attention_mask.new_zeros(doc_h.size(0))
        que_mask[b, :que_h.size(0)] = attention_mask.new_zeros(que_h.size(0))

    return doc_rep, que_rep, doc_mask.byte(), que_mask.byte()


def split_doc_sen_que(hidden_state, token_type_ids, attention_mask, sentence_span_list, return_byte_mask: bool = True,
                      return_cls: bool = False):
    batch, seq_len, hidden_size = hidden_state.size()
    cls_h = hidden_state[:, 0]

    que_hidden_list = []
    doc_hidden_list = []
    max_que_len = 0
    max_doc_len = 0
    max_sentences = 0
    for b in range(batch):
        unmasked_len = torch.sum(attention_mask[b]).item()
        # doc + [SEP]
        doc_len = torch.sum(token_type_ids[b]).item()
        # [CLS] + query + [SEP]
        que_len = unmasked_len - doc_len

        # TODO: the following line is wrong but won't cause calculation error.
        # max_que_len = max(max_que_len, que_len - 2)
        que_hidden_list.append(hidden_state[b, 1:(que_len - 1)])
        max_que_len = max(max_que_len, que_len)

        # doc_hidden = hidden_state[b, que_len:(que_len + doc_len - 1)]
        doc_sentence_list = []
        for sen_start, sen_end in sentence_span_list[b]:
            assert sen_start >= que_len
            assert sen_end < seq_len
            sentence = hidden_state[b, sen_start: (sen_end + 1)]
            max_doc_len = max(max_doc_len, sen_end - sen_start + 1)
            doc_sentence_list.append(sentence)
        max_sentences = max(max_sentences, len(doc_sentence_list))
        doc_hidden_list.append(doc_sentence_list)

    doc_rep = hidden_state.new_zeros(batch, max_sentences, max_doc_len, hidden_size)
    doc_mask = attention_mask.new_ones(batch, max_sentences, max_doc_len)
    que_rep = hidden_state.new_zeros(batch, max_que_len, hidden_size)
    que_mask = attention_mask.new_ones(batch, max_que_len)

    sentence_mask = attention_mask.new_ones(batch, max_sentences)

    for b in range(batch):
        sentence_num = len(doc_hidden_list[b])
        sentence_mask[b, :sentence_num] = attention_mask.new_zeros(sentence_num)
        for s in range(sentence_num):
            cur_sen_len = doc_hidden_list[b][s].size(0)
            doc_rep[b, s, :cur_sen_len] = doc_hidden_list[b][s]
            doc_mask[b, s, :cur_sen_len] = attention_mask.new_zeros(cur_sen_len)

    for b in range(batch):
        cur_que_len = que_hidden_list[b].size(0)
        que_rep[b, :cur_que_len] = que_hidden_list[b]
        que_mask[b, :cur_que_len] = attention_mask.new_zeros(cur_que_len)

    if return_byte_mask:
        output = [doc_rep, que_rep, doc_mask.byte(), que_mask.byte(), sentence_mask.byte()]
    else:
        output = [doc_rep, que_rep, doc_mask, que_mask, sentence_mask]
    if return_cls:
        output = tuple(output + [cls_h])
    else:
        output = tuple(output)
    return output


def split_doc_parts_que(hidden_state, token_type_ids, attention_mask, sentence_span_list, return_byte_mask: bool = True):
    batch, seq_len, hidden_size = hidden_state.size()

    que_hidden_list = []
    doc_sent_hidden_list = []
    doc_full_hidden_list = []
    # doc_hidden_list = []
    max_que_len = 0
    max_full_doc_len = 0
    max_sent_doc_len = 0
    # max_doc_len = 0
    max_sentences = 0
    for b in range(batch):
        unmasked_len = torch.sum(attention_mask[b]).item()
        # doc + [SEP]
        doc_len = torch.sum(token_type_ids[b]).item()
        # [CLS] + query + [SEP]
        que_len = unmasked_len - doc_len

        que_hidden_list.append(hidden_state[b, 1:(que_len - 1)])
        max_que_len = max(max_que_len, que_len - 2)

        doc_full_hidden_list.append(hidden_state[b, que_len: (que_len + doc_len - 1)])
        max_full_doc_len = max(max_full_doc_len, doc_len - 1)

        doc_sentence_list = []
        for sen_start, sen_end in sentence_span_list[b]:
            assert sen_start >= que_len and sen_end < seq_len
            sentence = hidden_state[b, sen_start: (sen_end + 1)]
            max_sent_doc_len = max(max_sent_doc_len, sen_end - sen_start + 1)
            doc_sentence_list.append(sentence)
        max_sentences = max(max_sentences, len(doc_sentence_list))
        doc_sent_hidden_list.append(doc_sentence_list)

    # doc_rep = hidden_state.new_zeros(batch, max_sentences, max_doc_len, hidden_size)
    # doc_mask = attention_mask.new_ones(batch, max_sentences, max_doc_len)
    doc_sent_rep = hidden_state.new_zeros(batch, max_sentences, max_sent_doc_len, hidden_size)
    doc_sent_mask = attention_mask.new_ones(batch, max_sentences, max_sent_doc_len)

    doc_full_rep = hidden_state.new_zeros(batch, max_full_doc_len, hidden_size)
    doc_full_mask = hidden_state.new_ones(batch, max_full_doc_len, hidden_size)

    que_rep = hidden_state.new_zeros(batch, max_que_len, hidden_size)
    que_mask = attention_mask.new_ones(batch, max_que_len)

    sentence_mask = attention_mask.new_ones(batch, max_sentences)

    for b in range(batch):
        sentence_num = len(doc_sent_hidden_list[b])
        sentence_mask[b, :sentence_num] = attention_mask.new_zeros(sentence_num)
        for s in range(sentence_num):
            cur_sen_len = doc_sent_hidden_list[b][s].size(0)
            doc_sent_rep[b, s, :cur_sen_len] = doc_sent_hidden_list[b][s]
            doc_sent_mask[b, s, :cur_sen_len] = attention_mask.new_zeros(cur_sen_len)

    for b in range(batch):
        cur_que_len = que_hidden_list[b].size(0)
        que_rep[b, :cur_que_len] = que_hidden_list[b]
        que_mask[b, :cur_que_len] = attention_mask.new_zeros(cur_que_len)

        cur_doc_len = doc_full_hidden_list[b].size(0)
        doc_full_rep[b, :cur_doc_len] = doc_full_hidden_list[b]
        doc_full_mask[b, :cur_doc_len] = attention_mask.new_zeros(cur_doc_len)

    if return_byte_mask:
        return doc_full_rep, doc_sent_rep, que_rep, doc_full_mask.byte(), doc_sent_mask.byte(), que_mask.byte(), sentence_mask.byte()
    else:
        return doc_full_rep, doc_sent_rep, que_rep, doc_full_mask, doc_sent_mask, que_mask, sentence_mask


def generate_unanswerable_label(answer_choice):
    """0 for unanswerable and 1 for answerable"""
    answerable_mask = ((answer_choice == 2) | (answer_choice == 3))
    label = answer_choice.masked_fill(answerable_mask, 1)
    return label


def generate_un_ab_label(answer_choice, ignore_index):
    """
    generate the label for unanswerable/abstract answers
    0 for unanswerable, 1 for abstract.
    """
    yesno_mask = ((answer_choice == 1) | (answer_choice == 2))
    label = answer_choice.masked_fill(yesno_mask, 1)
    extract_mask = (answer_choice == 3)
    label.masked_fill_(extract_mask, ignore_index)
    return label


def generate_yesno_label(answer_choice, ignore_index):
    """0 for yes and 1 for no"""
    not_yesno_mask = ((answer_choice == 0) | (answer_choice == 3))
    label = answer_choice.masked_fill(not_yesno_mask, ignore_index)

    yes_mask = (answer_choice == 1)
    label.masked_fill_(yes_mask, 0)
    no_mask = (answer_choice == 2)
    label.masked_fill_(no_mask, 1)
    return label


# bmm: batch matrix multiplication
# unsqueeze: add singleton dimension
# squeeze: remove singleton dimension
def weighted_avg(x, weights):  # used in lego_reader.py
    """ x = batch * len * d
        weights = batch * len
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)


def fp16_masked_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        # mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # To limit numerical errors from large vector elements outside the mask, we zero these out.
        result = torch.nn.functional.softmax(vector * mask, dim=dim)
        result = result * mask
        result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
    return result


def generate_local_mask(similarity_matrix, distance: int, drop_diagonal: bool = False):
    b, len1, len2 = similarity_matrix.size()

    mask = similarity_matrix.new_ones(len1, len2)
    l_mask = mask.tril(distance)
    u_mask = mask.triu(distance)

    final_mask = (1 - l_mask * u_mask).byte()

    if drop_diagonal:
        min_len = min(len1, len2)
        for i in range(min_len):
            final_mask[i][i] = 1

    return final_mask.unsqueeze(0).expand_as(similarity_matrix)


def local_prop_attention(que, doc, doc_mask, get_bi_attn_score, get_self_attn_score, drop_self: bool, distance: int):
    bi_attn = get_bi_attn_score(que, doc)

    self_attn = get_self_attn_score(doc, doc)
    local_mask = generate_local_mask(self_attn, distance=distance, drop_diagonal=drop_self)
    local_hidden = util.masked_softmax(self_attn, 1 - local_mask, dim=2).bmm(doc)

    bi_attn = get_bi_attn_score(que, doc)
    bi_attn_pooling = torch.max(bi_attn, dim=1, keepdim=True)[0]
    bi_hidden = util.masked_softmax(bi_attn_pooling, 1 - doc_mask, dim=2).bmm(local_hidden).squeeze(1)

    return bi_hidden


def get_masked_entropy(p: torch.Tensor, mask: torch.Tensor, reduction='mean'):
    """ Calculate entropy ot normalized tensor
    :param p: normalized tensor, [batch, seq_len]
    :param mask: 0 for true value and 1 for masked value, [batch]
    :param reduction:
    :return: sum(-p_i * log p_i)

    TODO: Shall we here divide the entropy with batch size instead of masked batch size?
        Because the simple cross entropy for sentence ids uses batch size too
    The 1e-15 could be replaced with 1e-45
    """
    mask = (1 - mask).unsqueeze(1).float()
    entropy = p * (-torch.log(p + 1e-15)) * mask
    # num = mask.sum()
    if reduction == 'mean':
        return entropy.sum() / p.size(0)
    return entropy.sum()


def extended_bert_attention_mask(attention_mask, dtype):
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask
