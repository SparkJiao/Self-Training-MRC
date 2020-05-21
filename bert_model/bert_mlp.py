from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch import nn
from torch.nn import functional as F


class BertQAYesNoMLP(BertPreTrainedModel):
    """
    BertForQuestionAnsweringForYesNo

    Model baseline:
        - Use the hidden state of [CLS] to predict the answer in Yes/No.
            - Version 1: Just use the pooled output of bert, which is passed through tanh activation and dropout.
                - only_yesno_output0
            - Version 2: Use the [CLS]'s hidden output to directly predict. (waiting to test)
                - only_yesno_output1
            - Version 3: Add dropout to [CLS]
                - only_yesno_output2

            - Version 3 is better.

        - Update:
            - Overfitting, considering increasing layers of mlp.
    """

    def __init__(self, config):
        super(BertQAYesNoMLP, self).__init__(config)
        print(f'The model {self.__class__.__name__} is loading...')
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.answer_choice = nn.Linear(config.hidden_size, 3)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, answer_choice=None,
                sentence_span_list=None, sentence_ids=None):
        sequence_output, pool_out = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # choice_logits = self.answer_choice(self.dropout(pool_out))  # Version 1
        # choice_logits = self.answer_choice(sequence_output[:, 0])  # Version 2
        choice_logits = self.answer_choice(self.dropout(sequence_output[:, 0]))  # Version 3

        output_dict = {
            'yesno_logits': choice_logits.float()
        }

        if answer_choice is not None:
            loss = F.cross_entropy(choice_logits, answer_choice)
            output_dict['loss'] = loss
        return output_dict
