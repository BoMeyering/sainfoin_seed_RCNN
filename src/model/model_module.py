import sys
from lightning import LightningModule
sys.path.append('./src/model')

from model import create_model


class FRCNNModel(LightningModule):
    """LightningModule for FasterRCNN model
    """

    def __init__(
        self,
        n_classes: int,
        n_obj_det: int, 
        img_size: int,
        learning_rate: float,
        momentum: float,
        gamma: float
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_obj_det = n_obj_det
        self.model = create_model(n_classes=self.n_classes, n_obj_det=self.n_obj_det)
        self.img_size = img_size
        self.lr = learning_rate
        
        self.map = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        self.id2label = {1: "can", 2: "carton", 3: "milk_bottle", 4: "water_bottle"}

    def forward(self, images, targets):
        return self.model(images, targets)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
    

    def training_step(self, batch, batch_idx):
        images, annotations, _ = batch

        losses = self.model(images, annotations)

        self.log(
            "train_loss",
            losses["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_class_loss",
            losses["class_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_box_loss",
            losses["box_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return losses["loss"]