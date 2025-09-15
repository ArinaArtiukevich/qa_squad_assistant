from torch import nn
from transformers import AutoModel


class QAModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.qa_outputs = nn.Linear(hidden_size, 2)
        self.no_answer_classifier = nn.Linear(hidden_size, 1)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            start_positions=None,
            end_positions=None,
            is_impossible=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        pooled_output = self.dropout(outputs.pooler_output)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        no_answer_logit = self.no_answer_classifier(pooled_output).squeeze(-1).contiguous()

        outputs = {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "no_answer_logit": no_answer_logit,
        }

        if start_positions is not None and end_positions is not None and is_impossible is not None:
            ce_loss = nn.CrossEntropyLoss()
            start_loss = ce_loss(start_logits, start_positions)
            end_loss = ce_loss(end_logits, end_positions)

            bce_loss = nn.BCEWithLogitsLoss()
            no_answer_loss = bce_loss(no_answer_logit, is_impossible.float())

            total_loss = (start_loss + end_loss + no_answer_loss) / 3.0
            outputs["loss"] = total_loss

        return outputs