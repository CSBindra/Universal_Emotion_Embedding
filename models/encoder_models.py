import torch
from transformers import AutoModel

from data.data_config import datasets_name


class TextEncoder(torch.nn.Module):
    def __init__(self, model_name="FacebookAI/roberta-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0]


class RobertaForMultilabelRegression(torch.nn.Module):
    def __init__(self, model_name='FacebookAI/roberta-base', num_labels=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.regressor = torch.nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        predictions = self.regressor(pooled_output)

        loss = None
        if labels is not None:
            loss_function = torch.nn.MSELoss()
            loss = loss_function(predictions, labels)

        return {"loss": loss, "logits": predictions}


class RobertaForClassification(torch.nn.Module):
    def __init__(self, model_name='FacebookAI/roberta-base', num_labels=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size, bias=True),
            torch.nn.Dropout(p=0.1, inplace=False),
            torch.nn.Linear(self.bert.config.hidden_size, num_labels, bias=True)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        predictions = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(predictions, labels)
        return {"loss": loss, "logits": predictions}


def load_bert_model(model_path, label_columns, dataset_type=datasets_name[0], freeze_encoder=False,
                    pre_trained_encoder=False):
    if pre_trained_encoder:
        if dataset_type == datasets_name[0]:
            model = RobertaForMultilabelRegression(num_labels=len(label_columns))
        else:
            model = RobertaForClassification(num_labels=len(label_columns))
        encoder_model = TextEncoder()
        encoder_model.load_state_dict(torch.load(model_path))
        model.bert = encoder_model.encoder
    else:
        if dataset_type == datasets_name[0]:
            model = RobertaForMultilabelRegression(num_labels=len(label_columns))
        else:
            model = RobertaForClassification(num_labels=len(label_columns))

    if freeze_encoder:
        for param in model.bert.parameters():
            param.requires_grad = False

        for param in model.regressor.parameters():
            param.requires_grad = True

    model.train()
    return model
