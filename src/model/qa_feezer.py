from transformers import TrainerCallback

from src.model.qa_utils import freeze_embeddings, unfreeze_layer


class GradualUnfreezingCallback(TrainerCallback):
    def __init__(self, model, unfreeze_start=1):
        self.model = model
        self.current_epoch = 0
        self.unfreeze_start = unfreeze_start

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch += 1

        if self.current_epoch == 1:
            freeze_embeddings(self.model)

        elif self.current_epoch >= self.unfreeze_start:
            layer_to_unfreeze = self.current_epoch - self.unfreeze_start
            if layer_to_unfreeze < len(self.model.bert.encoder.layer):
                unfreeze_layer(self.model, layer_to_unfreeze)
                print(f"Unfroze layer {layer_to_unfreeze}")
