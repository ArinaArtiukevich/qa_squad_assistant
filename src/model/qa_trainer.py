from transformers import Trainer


class QATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_names = ["start_positions", "end_positions", "is_impossible"]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.get("loss", None)
        return (loss, outputs) if return_outputs else loss