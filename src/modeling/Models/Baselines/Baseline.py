import pytorch_lightning as L

class BaselineModel(L.LightningModule):
	# pylint: disable-next=unused-argument
    def __init__(self, hyperparameters=None, input_channel_map=None, output_label_map=None):
        super().__init__()

        self._model = None

    def get_model(self):
        return self._model
