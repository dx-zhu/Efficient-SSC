from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
import numpy as np
import os
from datetime import datetime
import time


class SSCLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        interval: str,
        dataset: str,
        class_num: int,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net", "loss"])

        self.net = net

        # loss function
        self.criterion = loss

        self.data_set = dataset

        # metric objects for calculating and averaging acc,miou across batches
        self.train_mAcc = MulticlassAccuracy(class_num)
        self.train_mIoU = MulticlassJaccardIndex(class_num)
        self.val_mAcc = MulticlassAccuracy(class_num)
        self.val_mIoU = MulticlassJaccardIndex(class_num)
        self.test_mAcc = MulticlassAccuracy(class_num, average='none')
        self.test_mIoU = MulticlassJaccardIndex(class_num, average='none')

        # for averaging loss across batches
        self.train_cd = MeanMetric()
        self.train_seg = MeanMetric()
        self.val_cd = MeanMetric()
        self.val_seg = MeanMetric()
        self.test_cd = MeanMetric()
        self.test_seg = MeanMetric()

        self.val_cd_best = MinMetric()
        
        self.cd_class = []
        
        self.save_data = False
        # self.save_data = True
        self.metricvalue = []
        self.output_dir = f""

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        self.val_cd_best.reset()

    def model_step(self, batch: Any):
        x, y, _ = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y, self.current_epoch)
        sum_loss, last_cd, last_seg, cd_c = (
            loss["sum_loss"],
            loss["last_cd"],
            loss["last_seg"],
            loss["cd_c"]
        )
        preds, gt_seg = loss["pred_seg"], loss["gt_seg"]

        return (sum_loss, last_cd, last_seg, cd_c), preds, gt_seg
    
    def model_step_save(self, batch: Any):
        x, y, p = batch
        logits = self.forward(x)

        os.makedirs(self.output_dir, exist_ok=True)

        for idx, tensor in enumerate(logits):
            batch_size = tensor.shape[0]
            for batch_idx in range(batch_size):
                point_cloud = tensor[batch_idx].cpu().numpy()
                xyz = point_cloud[:, :3]
                class_scores = point_cloud[:, 3:]
                class_indices = np.argmax(class_scores, axis=1)
                point_cloud = np.concatenate((xyz, class_indices[:, None]), axis=1)
                file_path = f"{self.output_dir}/{p[batch_idx]}_{idx}.npy"
                np.save(file_path, point_cloud)

        loss = self.criterion(logits, y, self.current_epoch)
        sum_loss, last_cd, last_seg, cd_c = (
            loss["sum_loss"],
            loss["last_cd"],
            loss["last_seg"],
            loss["cd_c"]
        )
        preds, gt_seg = loss["pred_seg"], loss["gt_seg"]

        return (sum_loss, last_cd, last_seg, cd_c), preds, gt_seg

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        loss, last_cd, last_seg = loss[0], loss[1], loss[2]
        preds = preds if self.data_set == "ssc_pc" else preds[:, :, 1:]
        preds = preds.transpose(1, 2)
        # update and log metrics
        self.train_cd(last_cd)
        self.train_seg(last_seg)
        targets = targets if self.data_set == "ssc_pc" else targets - 1
        _ = self.train_mAcc(preds, targets)
        _ = self.train_mIoU(preds, targets)
        self.log("train/cd", self.train_cd, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/seg", self.train_seg, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/mAcc", self.train_mAcc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/mIoU", self.train_mIoU, on_step=False, on_epoch=True, prog_bar=True
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        loss, last_cd, last_seg = loss[0], loss[1], loss[2]
        preds = preds if self.data_set == "ssc_pc" else preds[:, :, 1:]
        preds = preds.transpose(1, 2)
        # update and log metrics
        self.val_cd(last_cd)
        self.val_seg(last_seg)
        targets = targets if self.data_set == "ssc_pc" else targets - 1
        _ = self.val_mAcc(preds, targets)
        _ = self.val_mIoU(preds, targets)
        self.log("val/cd", self.val_cd, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/seg", self.val_seg, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mAcc", self.val_mAcc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mIoU", self.val_mIoU, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        val_cd = self.val_cd.compute()
        self.val_cd_best(val_cd)  # update best so far val acc
        val_best = self.val_cd_best.compute()
        if val_cd == val_best:
            # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
            # otherwise metric would be reset by lightning after each epoch
            self.log("val_best/cd", self.val_cd_best.compute(), prog_bar=True)
            self.log("val_best/seg", self.val_seg.compute(), prog_bar=True)
            self.log("val_best/mAcc", self.val_mAcc.compute(), prog_bar=True)
            self.log("val_best/mIoU", self.val_mIoU.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        if self.save_data:
            loss, preds, targets = self.model_step_save(batch)
        else:
            loss, preds, targets = self.model_step(batch)
        loss, last_cd, last_seg, cd_c = loss[0], loss[1], loss[2], loss[3]
        self.cd_class.append(cd_c)
        preds = preds if self.data_set == "ssc_pc" else preds[:, :, 1:]
        preds = preds.transpose(1, 2)
        # update and log metrics
        self.test_cd(last_cd)
        self.test_seg(last_seg)
        targets = targets if self.data_set == "ssc_pc" else targets - 1
        _ = self.test_mAcc(preds, targets)
        _ = self.test_mIoU(preds, targets)
        # acc = accuracy_score(targets, preds)
        # miou = jaccard_score(targets, preds, average="macro")
        self.log("test/cd", self.test_cd, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/seg", self.test_seg, on_step=False, on_epoch=True, prog_bar=True)
        # self.log(
        #     "test/mAcc", self.test_mAcc, on_step=False, on_epoch=True, prog_bar=True
        # )
        # self.log(
        #     "test/mIoU", self.test_mIoU, on_step=False, on_epoch=True, prog_bar=True
        # )

        self.metricvalue.append([batch[2][0], 
                                 torch.mean(self.test_mAcc.compute()).item(), 
                                 torch.argmax(self.test_mAcc.compute()).item(),
                                 torch.mean(self.test_mIoU.compute()).item(), 
                                 torch.argmax(self.test_mIoU.compute()).item()])

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        if self.save_data:
            file_path = f"{self.output_dir}/000.npy"
            np.save(file_path, self.metricvalue)
        self.log(
            "test/mAcc", torch.mean(self.test_mAcc.compute()).item(), on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test/mIoU", torch.mean(self.test_mIoU.compute()).item(), on_step=False, on_epoch=True, prog_bar=True
        )
        
        data = np.array(self.cd_class)
        data[data == 0] = np.nan
        means = np.nanmean(data, axis=0)
        if self.data_set == 'nyucad_pc':
            self.log("test/cd_ceiling", means[1].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_floor", means[2].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_wall", means[3].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_window", means[4].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_chair", means[5].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_bed", means[6].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_sofa", means[7].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_table", means[8].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_tvs", means[9].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_furn", means[10].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_objects", means[11].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_ceiling", self.test_mAcc.compute()[0].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_floor", self.test_mAcc.compute()[1].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_wall", self.test_mAcc.compute()[2].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_window", self.test_mAcc.compute()[3].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_chair", self.test_mAcc.compute()[4].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_bed", self.test_mAcc.compute()[5].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_sofa", self.test_mAcc.compute()[6].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_table", self.test_mAcc.compute()[7].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_tvs", self.test_mAcc.compute()[8].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_furn", self.test_mAcc.compute()[9].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_objects", self.test_mAcc.compute()[10].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_ceiling", self.test_mIoU.compute()[0].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_floor", self.test_mIoU.compute()[1].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_wall", self.test_mIoU.compute()[2].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_window", self.test_mIoU.compute()[3].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_chair", self.test_mIoU.compute()[4].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_bed", self.test_mIoU.compute()[5].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_sofa", self.test_mIoU.compute()[6].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_table", self.test_mIoU.compute()[7].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_tvs", self.test_mIoU.compute()[8].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_furn", self.test_mIoU.compute()[9].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_objects", self.test_mIoU.compute()[10].item(), on_step=False, on_epoch=True, prog_bar=True)
        else:
            self.log("test/cd_bathtub", means[0].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_bed", means[1].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_bookshelf", means[2].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_cabinet", means[3].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_celling", means[4].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_chair", means[5].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_desk", means[6].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_door", means[7].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_floor", means[8].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_other", means[9].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_sink", means[10].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_sofa", means[11].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_table", means[12].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_toilet", means[13].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_tv", means[14].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/cd_wall", means[15].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_bathtub", self.test_mAcc.compute()[0].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_bed", self.test_mAcc.compute()[1].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_bookshelf", self.test_mAcc.compute()[2].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_cabinet", self.test_mAcc.compute()[3].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_celling", self.test_mAcc.compute()[4].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_chair", self.test_mAcc.compute()[5].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_desk", self.test_mAcc.compute()[6].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_door", self.test_mAcc.compute()[7].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_floor", self.test_mAcc.compute()[8].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_other", self.test_mAcc.compute()[9].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_sink", self.test_mAcc.compute()[10].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_sofa", self.test_mAcc.compute()[11].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_table", self.test_mAcc.compute()[12].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_toilet", self.test_mAcc.compute()[13].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_tv", self.test_mAcc.compute()[14].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mAcc_wall", self.test_mAcc.compute()[15].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_bathtub", self.test_mIoU.compute()[0].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_bed", self.test_mIoU.compute()[1].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_bookshelf", self.test_mIoU.compute()[2].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_cabinet", self.test_mIoU.compute()[3].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_celling", self.test_mIoU.compute()[4].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_chair", self.test_mIoU.compute()[5].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_desk", self.test_mIoU.compute()[6].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_door", self.test_mIoU.compute()[7].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_floor", self.test_mIoU.compute()[8].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_other", self.test_mIoU.compute()[9].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_sink", self.test_mIoU.compute()[10].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_sofa", self.test_mIoU.compute()[11].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_table", self.test_mIoU.compute()[12].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_toilet", self.test_mIoU.compute()[13].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_tv", self.test_mIoU.compute()[14].item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mIoU_wall", self.test_mIoU.compute()[15].item(), on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": self.hparams.interval,
                    # "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
