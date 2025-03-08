import torch.nn as nn

class MoLFormerWithRegressionHead(nn.Module):
    # TODO: your code goes here
    def __init__(self, model):
        super(MoLFormerWithRegressionHead, self).__init__()
        self.model = model
        self.regression_head = nn.Linear(model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        # sequence_output = outputs[0]
        # cls_token = sequence_output[:, 0, :]
        outputs_head = self.regression_head(cls_token)
        return outputs_head
    

class HiddenModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None, task_ids=None, **kwargs):
        model_output = self.model(input_ids, attention_mask)
        return model_output