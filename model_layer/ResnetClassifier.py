import sys
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from configuration.model_config import Model_Config
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, update_bn
import main_pipe
from model_layer import TripletLoss
from configuration import config
from pytorch_lightning import LightningModule, seed_everything, Trainer

from model_layer.ArcFace import ArcFace


def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=2)
    model.conv1 = nn.Conv2d(5, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    modules = list(model.children())[:-1]
    model = nn.Sequential(*modules)

    return model

class LitResnet(LightningModule):

    def __init__(self, lr=0.05,alpha=0.1):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()
        self.fc1 = nn.Linear(512,2)
        # self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        # features1 = self.model(x)
        # features2 = self.fc1(features1)
        features = self.model(x)
        logits = self.fc1(features)
        return F.log_softmax(logits, dim=1), features

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits,features = self.forward(x)
        # output = F.softmax(logits, dim=1)
        loss = F.cross_entropy(logits, y)
        # preds = torch.argmax(output, dim=1)
        # acc = accuracy(preds, y)
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, training_step_outputs):
        pass

    def test_epoch_end(self, outputs):
        pass

    def evaluate(self, batch, stage=None):
        x, y, plate = batch
        logits,features = self.forward(x)
        output = F.softmax(logits, dim=1)
        loss = F.cross_entropy(output, y)
        preds = torch.argmax(output, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Val",
        loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Val",
        acc, self.current_epoch)

        if stage=='test':
            pass

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=5e-7)
        # steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            'scheduler': ReduceLROnPlateau(optimizer,factor=0.3,patience=8),
            'monitor': 'val_loss',
            'interval': 'step',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}


class LitResnetwithTripletLoss(LitResnet):

    def __init__(self, lr=0.05,alpha=0.1, loss_type='triplet_plate'):
        super(LitResnetwithTripletLoss, self).__init__(lr)
        # self.alpha = 0.9
        # self.loss_type = loss_type
        self.save_hyperparameters()

    def forward(self, x):

        features = self.model(x)
        # features2 = self.fc1(features1.squeeze())
        # logits = self.fc2(features2.squeeze())
        logits = self.fc1(features.squeeze())
        return logits, features

    def training_step(self, batch, batch_idx):
        x, y, plates = batch
        logits, features = self.forward(x)
        # feature_layer = model._modules.get('fc')
        # output = F.softmax(logits, dim=1)
        out = F.softmax(logits)
        ce_loss = F.cross_entropy(out, y)
        # trip_loss=0
        # mask = triplet_loss.get_valid_triplets_mask(plates)
        # if self.hparams.loss_type=='triplet_plate':

            # loss = self.hparams.alpha * ce_loss + (1 - self.hparams.alpha) * plate_loss_out

        # if self.hparams.loss_type == 'contractive_plate':
        #     plate_loss_out, _ = TripletLoss.batch_all_triplet_loss(plates, features.squeeze(), 1.0, squared=False,loss_type=self.hparams.loss_type)
            # loss = self.hparams.alpha * ce_loss + (1 - self.hparams.alpha) * plate_loss_out


        # elif self.hparams.loss_type=='arcface_plate':
        #     plate_loss = ArcFace()
        #     plate_loss_out = plate_loss(features.squeeze(), plates)

        # elif self.hparams.loss_type == 'arcface':
        #     loss_func = ArcFace()
        #     plate_loss_out = loss_func(features.squeeze(), y)
        #     self.hparams.alpha = 1
        if self.hparams.loss_type == 'contractive_plate' or self.hparams.loss_type == 'triplet_plate':
            plate_loss_out, _ = TripletLoss.batch_all_triplet_loss(plates, features.squeeze(), 1.0, squared=False,
                                                                   loss_type=self.hparams.loss_type)
        elif self.hparams.loss_type =='softmax':
            self.hparams.alpha = 0
            # plate_loss_out,_ = TripletLoss.batch_all_triplet_loss(plates, features.squeeze(), 1.0, squared=False)
            plate_loss_out=0
        else:
            raise ValueError(self.hparams.loss_type + ' not supported')

        loss = ce_loss + self.hparams.alpha * plate_loss_out
        # preds = torch.argmax(output, dim=1)
        # acc = accuracy(preds, y)
        self.log('ce_loss', ce_loss, on_epoch=True,on_step=False,prog_bar=True)
        self.log('plate_loss', plate_loss_out,on_epoch=True,on_step=False,prog_bar=True)
        self.log('loss', plate_loss_out, on_epoch=True, on_step=False, prog_bar=True)
        # self.logger.experiment.add_scalar("ce_loss/Train",
        # ce_loss,self.current_epoch)
        # self.logger.experiment.add_scalar("plate_loss/Train",
        # plate_loss_out,self.current_epoch)
        # self.logger.experiment.add_scalar("overall_loss/Train",
        #                                   loss, self.current_epoch)

        return loss