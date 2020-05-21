"""
In this file, all the mask means "1" for true vale while "0" for masked value, which is different from layers.py
"""
import torch
import warnings
from allennlp.modules import Seq2SeqEncoder
from pytorch_pretrained_bert.modeling import BertLayerNorm
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

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


class RNNDropout(nn.Module):
    """
    A thin wrapper of Allennlp's dropout.
    """

    def __init__(self, p, batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.batch_first = batch_first

    def forward(self, inputs):

        if not self.training:
            return inputs
        if self.batch_first:
            mask = inputs.new_ones(inputs.size(0), 1, inputs.size(2), requires_grad=False)
        else:
            mask = inputs.new_ones(1, inputs.size(1), inputs.size(2), requires_grad=False)
        return self.dropout(mask) * inputs


@Seq2SeqEncoder.register("concat_rnn")
class ConcatRNN(Seq2SeqEncoder):

    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True, rnn_type="GRU", stateful=False, batch_first=True):
        super().__init__(stateful=stateful)
        self.input_dim = input_size
        self.output_dim = num_layers * (hidden_size * 2 if bidirectional else hidden_size)
        self.bidirectional = bidirectional
        rnn_cls = Seq2SeqEncoder.by_name(rnn_type.lower())
        self.rnn_list = torch.nn.ModuleList(
            [rnn_cls(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional)])

        for _ in range(num_layers - 1):
            self.rnn_list.append(
                rnn_cls(input_size=hidden_size * 2, hidden_size=hidden_size, bidirectional=bidirectional))

        # self.dropout = RNNDropout(dropout, batch_first=batch_first)

    def forward(self, inputs, mask, hidden=None):
        outputs_list = []

        for layer in self.rnn_list:
            # outputs = layer(self.dropout(inputs), mask, hidden)
            outputs = layer(dropout(inputs, p=my_dropout_p, training=self.training), mask, hidden)
            outputs_list.append(outputs)
            inputs = outputs

        return torch.cat(outputs_list, -1)

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim

    def is_bidirectional(self) -> bool:
        return self.bidirectional


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


class MultiHeadPooling(nn.Module):
    def __init__(self, input_dim, num_attention_heads, do_transform: bool = False):
        super(MultiHeadPooling, self).__init__()
        self.attention_dim_per_head = input_dim // num_attention_heads
        self.all_heads_attention_dim = self.attention_dim_per_head * num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.query = nn.Linear(input_dim, num_attention_heads)
        self.value = nn.Linear(input_dim, self.all_heads_attention_dim)

        self.do_transform = do_transform
        if self.do_transform:
            self.transform = nn.Linear(self.attention_dim_per_head, self.attention_dim_per_head)

        self.layer_norm = BertLayerNorm(self.attention_dim_per_head, eps=1e-12)

    def shape(self, x, dim):
        batch_size = x.size(0)
        return x.reshape(batch_size, -1, self.num_attention_heads, dim).transpose(1, 2)

    def un_shape(self, x, dim):
        batch = x.size(0)
        return x.transpose(1, 2).reshape(batch, -1, self.num_attention_heads * dim)

    def forward(self, x, x_mask):
        # Version 0(Has bug)
        # batch, seq_length, _ = x.size()
        # x_mask = x_mask.unsqueeze(1).expand(-1, self.num_attention_heads, seq_length).reshape(batch * self.num_attention_heads, -1)
        # query = self.query(x).reshape(batch * self.num_attention_heads, seq_length).unsqueeze(1)
        # value = self.value(x).reshape(batch * self.num_attention_heads, seq_length, -1)
        # product = masked_softmax(query, x_mask).bmm(value)
        # transformed_product = self.transform(product)
        # transformed_product = dropout(transformed_product, p=my_dropout_p, training=self.training)
        # transformed_product = self.layer_norm(transformed_product)
        # return transformed_product.reshape(batch, -1)

        # Version 1
        batch, seq_len, _ = x.size()
        query = self.query(x)
        query = self.shape(query, 1).squeeze(-1)
        # assert query.size() == (batch, self.num_attention_heads, seq_len)
        value = self.value(x)
        value = self.shape(value, self.attention_dim_per_head)
        # assert value.size() == (batch, self.num_attention_heads, seq_len, self.attention_dim_per_head)

        x_mask = x_mask.unsqueeze(1).expand_as(query)
        alpha = masked_softmax(query, x_mask)
        dropout_alpha = F.dropout(alpha, p=my_dropout_p, training=self.training)
        y = dropout_alpha.unsqueeze(2).matmul(value)  # [b, k, 1, m] * [b, k, m, h] -> [b, k, 1, h]
        # assert y.size() == (batch, self.num_attention_heads, 1, self.attention_dim_per_head)
        y = self.un_shape(y, self.attention_dim_per_head).squeeze(1)
        # assert y.size() == (batch, self.all_heads_attention_dim)
        if self.do_transform:
            y = self.transform(y.reshape(-1, self.attention_dim_per_head))
        else:
            y = y.reshape(-1, self.attention_dim_per_head)
        y = self.layer_norm(y).reshape(batch, self.all_heads_attention_dim)
        # return self.layer_norm(y.reshape(-1, self.attention_dim_per_head)).reshape(batch, self.all_heads_attention_dim)
        return y


class MultiHeadPooling1(nn.Module):
    """
    Remove the transform layer and LayerNorm layer
    """

    def __init__(self, input_dim, num_attention_heads):
        super(MultiHeadPooling1, self).__init__()
        self.attention_dim_per_head = input_dim // num_attention_heads
        self.all_heads_attention_dim = self.attention_dim_per_head * num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.query = nn.Linear(input_dim, num_attention_heads)
        self.value = nn.Linear(input_dim, self.all_heads_attention_dim)

    def shape(self, x, dim):
        batch_size = x.size(0)
        return x.reshape(batch_size, -1, self.num_attention_heads, dim).transpose(1, 2)

    def un_shape(self, x, dim):
        batch = x.size(0)
        return x.transpose(1, 2).reshape(batch, -1, self.num_attention_heads * dim)

    def forward(self, x, x_mask):
        batch, seq_len, _ = x.size()
        query = self.query(x)
        query = self.shape(query, 1).squeeze(-1)
        # assert query.size() == (batch, self.num_attention_heads, seq_len)
        value = self.value(x)
        value = self.shape(value, self.attention_dim_per_head)
        # assert value.size() == (batch, self.num_attention_heads, seq_len, self.attention_dim_per_head)

        x_mask = x_mask.unsqueeze(1).expand_as(query)
        alpha = masked_softmax(query, x_mask)
        dropout_alpha = F.dropout(alpha, p=my_dropout_p, training=self.training)
        y = dropout_alpha.unsqueeze(2).matmul(value)  # [b, k, 1, m] * [b, k, m, h] -> [b, k, 1, h]
        # assert y.size() == (batch, self.num_attention_heads, 1, self.attention_dim_per_head)
        y = self.un_shape(y, self.attention_dim_per_head).squeeze(1)
        # assert y.size() == (batch, self.all_heads_attention_dim)
        return y


class MultiHeadPooling2(nn.Module):
    """
    Change the way of dropout
    """

    def __init__(self, input_dim, num_attention_heads):
        super(MultiHeadPooling2, self).__init__()
        self.attention_dim_per_head = input_dim // num_attention_heads
        self.all_heads_attention_dim = self.attention_dim_per_head * num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.query = nn.Linear(input_dim, num_attention_heads)
        self.value = nn.Linear(input_dim, self.all_heads_attention_dim)

    def shape(self, x, dim):
        batch_size = x.size(0)
        return x.reshape(batch_size, -1, self.num_attention_heads, dim).transpose(1, 2)

    def un_shape(self, x, dim):
        batch = x.size(0)
        return x.transpose(1, 2).reshape(batch, -1, self.num_attention_heads * dim)

    def forward(self, x, x_mask):
        batch, seq_len, _ = x.size()
        x = dropout(x, p=my_dropout_p, training=self.training)
        query = self.query(x)
        query = self.shape(query, 1).squeeze(-1)
        # assert query.size() == (batch, self.num_attention_heads, seq_len)
        value = self.value(x)
        value = self.shape(value, self.attention_dim_per_head)
        # assert value.size() == (batch, self.num_attention_heads, seq_len, self.attention_dim_per_head)

        x_mask = x_mask.unsqueeze(1).expand_as(query)
        alpha = masked_softmax(query, x_mask)
        # dropout_alpha = F.dropout(alpha, p=my_dropout_p, training=self.training)
        y = alpha.unsqueeze(2).matmul(value)  # [b, k, 1, m] * [b, k, m, h] -> [b, k, 1, h]
        # assert y.size() == (batch, self.num_attention_heads, 1, self.attention_dim_per_head)
        y = self.un_shape(y, self.attention_dim_per_head).squeeze(1)
        # assert y.size() == (batch, self.all_heads_attention_dim)
        return y


class LinearSelfAttention(nn.Module):
    def __init__(self, input_size):
        super(LinearSelfAttention, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        # Version 0
        x_flat = dropout(x, p=my_dropout_p, training=self.training).view(-1, x.size(-1))
        # Version 1
        # x_flat = dropout(x.reshape(-1, x.size(-1)), p=my_dropout_p, training=self.training)
        scores = self.linear(x_flat).reshape(x.size(0), 1, -1)
        y = masked_softmax(scores, x_mask).bmm(x).squeeze(1)

        # Version 2
        # x_flat = x.reshape(-1, x.size(-1))
        # scores = self.linear(x_flat).reshape(x.size(0), 1, -1)
        # y = masked_softmax(scores, x_mask).bmm(x).squeeze(1)
        # y = dropout(y, p=my_dropout_p, training=self.training)
        return y


def extended_bert_attention_mask(attention_mask, dtype):
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


def masked_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.

    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    In the case that the input vector is completely masked, this function returns an array
    of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of a model
    that uses categorical cross-entropy loss.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.to(dtype=vector.dtype)
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # To limit numerical errors from large vector elements outside the mask, we zero these out.
        result = torch.nn.functional.softmax(vector * mask, dim=dim)
        result = result * mask
        result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
    return result


def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.

    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.

    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.

        # Note: (1e-45).log() will cause `inf` while using fp16, so will first use float to compute zero mask and convert it back,
        mask = (mask.float() + 1e-45)
        vector = vector + (mask + 1e-45).log().to(dtype=vector.dtype)
    return torch.nn.functional.log_softmax(vector, dim=dim)


def split_doc_sen_que(hidden_state, token_type_ids, attention_mask, sentence_span_list, return_cls: bool = False,
                      max_sentences: int = 0):
    batch, seq_len, hidden_size = hidden_state.size()
    cls_h = hidden_state[:, 0]

    que_hidden_list = []
    doc_hidden_list = []
    max_que_len = 0
    max_doc_len = 0
    # Use the setting of global max_sentences.
    # max_sentences = 0
    for b in range(batch):
        unmasked_len = torch.sum(attention_mask[b]).item()
        # doc + [SEP]
        doc_len = torch.sum(token_type_ids[b]).item()
        # [CLS] + query + [SEP]
        que_len = unmasked_len - doc_len

        que_hidden_list.append(hidden_state[b, 1:(que_len - 1)])
        max_que_len = max(max_que_len, que_len - 2)

        # doc_hidden = hidden_state[b, que_len:(que_len + doc_len - 1)]
        doc_sentence_list = []
        for (sen_start, sen_end) in sentence_span_list[b]:
            assert sen_start >= que_len, (sen_start, que_len)
            assert sen_end < seq_len
            sentence = hidden_state[b, sen_start: (sen_end + 1)]
            max_doc_len = max(max_doc_len, sen_end - sen_start + 1)
            doc_sentence_list.append(sentence)
        max_sentences = max(max_sentences, len(doc_sentence_list))
        doc_hidden_list.append(doc_sentence_list)

    doc_rep = hidden_state.new_zeros(batch, max_sentences, max_doc_len, hidden_size)
    doc_mask = attention_mask.new_zeros(batch, max_sentences, max_doc_len)
    que_rep = hidden_state.new_zeros(batch, max_que_len, hidden_size)
    que_mask = attention_mask.new_zeros(batch, max_que_len)

    sentence_mask = attention_mask.new_zeros(batch, max_sentences)

    for b in range(batch):
        sentence_num = len(doc_hidden_list[b])
        sentence_mask[b, :sentence_num] = attention_mask.new_ones(sentence_num)
        for s in range(sentence_num):
            cur_sen_len = doc_hidden_list[b][s].size(0)
            doc_rep[b, s, :cur_sen_len] = doc_hidden_list[b][s]
            doc_mask[b, s, :cur_sen_len] = attention_mask.new_ones(cur_sen_len)
        # Avoid all masked.
        for s in range(sentence_num, max_sentences):
            doc_mask[b, s, 0] = 1

    for b in range(batch):
        cur_que_len = que_hidden_list[b].size(0)
        que_rep[b, :cur_que_len] = que_hidden_list[b]
        que_mask[b, :cur_que_len] = attention_mask.new_ones(cur_que_len)

    output = [doc_rep, que_rep, doc_mask, que_mask, sentence_mask]
    if return_cls:
        output = tuple(output + [cls_h])
    else:
        output = tuple(output)
    return output


def split_doc_sen_que_roberta(hidden_state, token_type_ids, attention_mask, sentence_span_list, return_cls: bool = False,
                              max_sentences: int = 0):
    batch, seq_len, hidden_size = hidden_state.size()
    cls_h = hidden_state[:, 0]

    que_hidden_list = []
    doc_hidden_list = []
    max_que_len = 0
    max_doc_len = 0
    # Use the setting of global max_sentences.
    # max_sentences = 0
    for b in range(batch):
        unmasked_len = torch.sum(attention_mask[b]).item()
        # doc + [SEP]
        doc_len = torch.sum(token_type_ids[b]).item()
        # [CLS] + query + [SEP] + [SEP]
        que_len = unmasked_len - doc_len

        que_hidden_list.append(hidden_state[b, 1:(que_len - 2)])
        max_que_len = max(max_que_len, que_len - 3)

        # doc_hidden = hidden_state[b, que_len:(que_len + doc_len - 1)]
        doc_sentence_list = []
        for (sen_start, sen_end) in sentence_span_list[b]:
            assert sen_start >= que_len
            assert sen_end < seq_len
            sentence = hidden_state[b, sen_start: (sen_end + 1)]
            max_doc_len = max(max_doc_len, sen_end - sen_start + 1)
            doc_sentence_list.append(sentence)
        max_sentences = max(max_sentences, len(doc_sentence_list))
        doc_hidden_list.append(doc_sentence_list)

    doc_rep = hidden_state.new_zeros(batch, max_sentences, max_doc_len, hidden_size)
    doc_mask = attention_mask.new_zeros(batch, max_sentences, max_doc_len)
    que_rep = hidden_state.new_zeros(batch, max_que_len, hidden_size)
    que_mask = attention_mask.new_zeros(batch, max_que_len)

    sentence_mask = attention_mask.new_zeros(batch, max_sentences)

    for b in range(batch):
        sentence_num = len(doc_hidden_list[b])
        sentence_mask[b, :sentence_num] = attention_mask.new_ones(sentence_num)
        for s in range(sentence_num):
            cur_sen_len = doc_hidden_list[b][s].size(0)
            doc_rep[b, s, :cur_sen_len] = doc_hidden_list[b][s]
            doc_mask[b, s, :cur_sen_len] = attention_mask.new_ones(cur_sen_len)
        # Avoid all masked.
        for s in range(sentence_num, max_sentences):
            doc_mask[b, s, 0] = 1

    for b in range(batch):
        cur_que_len = que_hidden_list[b].size(0)
        que_rep[b, :cur_que_len] = que_hidden_list[b]
        que_mask[b, :cur_que_len] = attention_mask.new_ones(cur_que_len)

    output = [doc_rep, que_rep, doc_mask, que_mask, sentence_mask]
    if return_cls:
        output = tuple(output + [cls_h])
    else:
        output = tuple(output)
    return output


def gumbel_softmax(logits, mask=None, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """

    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    # y_soft = gumbels.softmax(dim)
    y_soft = masked_softmax(gumbels, mask, dim=dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def get_max_index_one_hot(tensor: torch.Tensor, dim=-1):
    index = tensor.max(dim=dim, keepdim=True)[1]
    one_hot = torch.zeros_like(tensor).scatter_(dim, index, 1.0)
    return one_hot


def get_k_max_mask(tensor: torch.Tensor, dim=-1, k=1):
    tmp = tensor
    final_vec = torch.zeros_like(tensor)
    for i in range(k):
        mask = get_max_index_one_hot(tmp, dim)
        final_vec += mask
        tmp = tmp.masked_fill(mask.byte(), float('-inf'))
    return final_vec
