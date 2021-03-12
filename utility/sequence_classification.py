import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss, MSELoss
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForSequenceClassification, BertPreTrainedModel, BertModel, Trainer, TrainingArguments

class MySequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)      

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None, 
        output_attentions=None,
        output_hidden_states=None,
    ):
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        pooled_output = outputs[1] # hidden state
        # output = torch.reshape(outputs[0][check], (-1, 3 * 768))

        output = self.dropout(pooled_output)
        # output = self.dropout(output)
        logits = self.classifier(pooled_output)
        # logits = self.classifier(output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None: # if labels is None, then loss won't be returned
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)