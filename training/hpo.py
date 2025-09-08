#!/usr/bin/env python
# coding: utf-8

from lightning.pytorch.callbacks import ModelCheckpoint
import torch.nn.init as init
import torch
import os
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from constants import *
import hickle as hkl
import numpy.random as npr

from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.tune.integration.pytorch_lightning import (
    TuneReportCheckpointCallback)
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray import air
from ray.train.torch import TorchTrainer


import statistics
import time
import csv

time.sleep(npr.rand()*5) # Sleep random (up to 5 secs) to avoid race condition for ray tune

torch.manual_seed(42)
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"#"cuda:2"

# DEVICE = "cuda"


ERR = 1e-6

def r2_score(y_pred:torch.Tensor, y_true:torch.Tensor) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    
    r2 = 1 - ss_res / ss_tot
    
    return r2.item()




n_column_features = 21#23
n_scalar_features = 14#16#25 #14 for losing lat/lon info
n_features = n_column_features*60+n_scalar_features
print(n_features)




large_scale_forcing_mask = np.zeros(n_features, dtype=bool)
large_scale_forcing_mask[6*60:12*60] = True
n_column_features = 15
n_features = n_column_features*60+n_scalar_features
print(n_features)




n_column_targets = 5
n_scalar_targets = 2#8
n_targets = n_column_targets*60+n_scalar_targets
print(n_targets)




def get_traindataloader(batch_size):
    # train_dataset = TensorDataset(*torch.load("/scratch/b/b309215/LEAP/ClimSim_high-res/expandcnv_postprocessed/train_nols_nolatlon_normed_1e6sample.pt"))
    train_dataset = TensorDataset(*torch.load("/scratch/b/b309215/LEAP/ClimSim_high-res/expandcnv_postprocessed/train_nols_nolatlon_normed_3e6sample.pt"))
    # train_dataset = TensorDataset(*torch.load("/scratch/b/b309215/LEAP/ClimSim_high-res/expandcnv_postprocessed/train_nols_nolatlon_normed_1e4sample.pt"))
    # train_dataset = torch.utils.data.Subset(train_dataset, range(10000))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader

def get_valdataloader(batch_size):
    # val_dataset = TensorDataset(*torch.load("/scratch/b/b309215/LEAP/ClimSim_high-res/expandcnv_postprocessed/val_nols_nolatlon_normed_1e6sample.pt"))
    val_dataset = TensorDataset(*torch.load("/scratch/b/b309215/LEAP/ClimSim_high-res/expandcnv_postprocessed/val_nols_nolatlon_normed_1.5e6sample.pt"))
    # val_dataset = TensorDataset(*torch.load("/scratch/b/b309215/LEAP/ClimSim_high-res/expandcnv_postprocessed/val_nols_nolatlon_normed_1e4sample.pt"))
    # val_dataset = torch.utils.data.Subset(val_dataset, range(10000))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return val_loader
    
def get_testdataloader(batch_size):
    test_dataset = TensorDataset(*torch.load("/scratch/b/b309215/LEAP/ClimSim_high-res/expandcnv_postprocessed/test_nols_nolatlon_normed_1.5e6sample.pt"))
    # test_dataset = TensorDataset(*torch.load("/scratch/b/b309215/LEAP/ClimSim_high-res/expandcnv_postprocessed/test_nols_nolatlon_normed_1e4sample.pt"))
    # test_dataset = torch.utils.data.Subset(test_dataset, range(1000))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # test_dataset = torch.utils.data.Subset(val_dataset, range(16))
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    return test_loader




Y_MEAN = torch.from_numpy(np.load('/scratch/b/b309215/LEAP/ClimSim_high-res/expandcnv_postprocessed/train_target_normed_my.npy'))
Y_STD = torch.from_numpy(np.load('/scratch/b/b309215/LEAP/ClimSim_high-res/expandcnv_postprocessed/train_target_normed_sy.npy'))



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




zeroout_index = list(range(60,75)) + list(range(120,135)) + list(range(180,195)) + list(range(240,255))#Indices to just have a 0 output
zeroout_index = torch.tensor(list(zeroout_index))

class FFNN_LSTM_6_AVG(nn.Module):
    def __init__(self, feature_target_lengths, zeroout_index, nlev, config):
        super(FFNN_LSTM_6_AVG, self).__init__()
        self.nlev = nlev

        # # 13.6M model
        # self.encode_dim = 300
        # self.hidden_dim = 280
        # self.iter_dim = 800
        # self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,6,batch_first=True,dropout=0.01,bidirectional=True)

        # 8M model
        self.encode_dim = config['encode_dim']#300
        self.hidden_dim = config['hidden_dim']#280
        self.iter_dim = config['iter_dim']#800
        self.lstm_layers = config['lstm_layers']#800
        self.dropout_rate = config['dropout_rate']
        
        self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,self.lstm_layers,batch_first=True,dropout=self.dropout_rate,bidirectional=True)
        
        # # 1.2M model
        # self.encode_dim = 200
        # self.hidden_dim = 100
        # self.iter_dim = 300
        # self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,3,batch_first=True,dropout=0.01,bidirectional=True)

        self.n_column_features, self.n_scalar_features, self.n_column_targets, self.n_scalar_targets = \
            feature_target_lengths
        self.zeroout_index = zeroout_index
        self.input_size = self.n_column_features*self.nlev + self.n_scalar_features
        self.output_size = self.n_column_targets*self.nlev + self.n_scalar_targets
        
        self.Linear_1 = nn.Linear(self.n_column_features+self.n_scalar_features, self.encode_dim)
        self.Linear_2 = nn.Linear(6*self.hidden_dim+self.encode_dim, self.iter_dim)
        self.Linear_3 = nn.Linear(self.iter_dim, self.n_column_targets)
        self.Linear_4_0 = nn.Linear(self.iter_dim, self.iter_dim*2)

        self.Linear_4 = nn.Linear(self.iter_dim*2, self.n_scalar_targets)
        
        self.weight = nn.Parameter(torch.zeros(1,self.output_size))
        self.bias = nn.Parameter(torch.zeros(1,self.output_size))
        torch.nn.init.xavier_uniform(self.weight)
        torch.nn.init.xavier_uniform(self.bias)
        # self.bias = nn.Linear(len(seq_y_list)*self.nlev+len(num_y_list),1)
        # self.weight = nn.Linear(len(seq_y_list)*self.nlev+len(num_y_list),1)
        
        self.avg_pool_1 = nn.AvgPool1d(kernel_size=3,stride=1,padding=1)
        
    def forward(self, x):
#         x_seq = x[:,0:60*len(seq_fea_list)]
#         x_seq = x_seq.reshape((-1,len(seq_fea_list),60))
#         x_seq = torch.transpose(x_seq, 1, 2)
        
#         x_num = x[:,self.nlev*len(seq_fea_list):x.shape[1]]
#         x_num_repeat = x_num.reshape((-1,1,len(num_fea_list)))
#         x_num_repeat = x_num_repeat.repeat((1,self.nlev,1))
        
        # x_seq = F.elu(self.Linear_1(torch.concat((x_seq,x_num_repeat),dim=-1)/5))
        # dims: b->batch, v->variable, h->height, e->encode
        # print(x.shape) # (b,h,v)
        x_seq = F.elu(self.Linear_1(x)) # (b,h,e)
        
        # self.LSTM_1.flatten_parameters()
        x_seq_1,_ = self.LSTM_1(x_seq/5)
        
        x_seq_1_mean = torch.mean(x_seq_1,dim=1,keepdim=True)
        x_seq_1_mean = x_seq_1_mean.repeat((1,self.nlev,1))

        x_seq_1_avg_pool = self.avg_pool_1(torch.transpose(x_seq_1, 1, 2))
        x_seq_1_avg_pool = torch.transpose(x_seq_1_avg_pool,1, 2)
        
        x_seq_1 = F.elu(self.Linear_2(torch.cat((x_seq_1,x_seq_1_mean,x_seq,x_seq_1_avg_pool),dim=-1)/5))
        
        x_seq_out = self.Linear_3(x_seq_1)
        x_seq_out = torch.transpose(x_seq_out, 1, 2)
        x_seq_out = x_seq_out.reshape((-1,self.nlev*self.n_column_targets)) # (b,seq_out)
        
        x_num_out = F.elu(self.Linear_4_0(torch.mean(x_seq_1,dim=1)))
        x_num_out = self.Linear_4(x_num_out) # (b,num_out)

        # print(x_seq_out.shape, x_num_out.shape)
        # output = self.weight.weight*(torch.concat((x_seq_out,x_num_out),dim=-1))/3+self.bias.weight/3
        output = self.weight*(torch.concat((x_seq_out,x_num_out),dim=-1))/3+self.bias/3
        # output = torch.concat((x_seq_out,x_num_out),dim=-1)
        
        output[:,self.zeroout_index] = output[:,self.zeroout_index]*0.0
        
        return output

class HighResLeapModel(LightningModule):
    def __init__(self, feature_target_lengths, zeroout_index, config, nlev=60, use_confidence=False):
        super().__init__()
        self.nlev = nlev
        self.n_column_features, self.n_scalar_features, self.n_column_targets, self.n_scalar_targets = \
            feature_target_lengths
        self.input_size = self.n_column_features*self.nlev + self.n_scalar_features
        self.output_size = self.n_column_targets*self.nlev + self.n_scalar_targets
        self.zeroout_index = zeroout_index
        self.zeroout_mask = torch.zeros(n_column_targets*60+n_scalar_targets, dtype=bool)
        self.zeroout_mask[zeroout_index] = 1
        
        # self.val_score = []
        # self.masked_val_score = []
        # self.cpu_times = []
        # self.gpu_times = []
        
        self.lr = config['learning_rate']
        self.wd = config['weight_decay']
        self.scheduler = config['scheduler']
        self.batch_size = config['batch_size']

        self.use_confidence = use_confidence
        # For confidence loss:
        if self.use_confidence:
            self.n_column_targets *= 2
            self.n_scalar_targets *= 2

        feature_target_lengths = \
            self.n_column_features, self.n_scalar_features, self.n_column_targets, self.n_scalar_targets
        self.network = FFNN_LSTM_6_AVG(feature_target_lengths, zeroout_index, nlev, config)
        
        if self.use_confidence:
            self.criterion = nn.HuberLoss(delta=1., reduction='none')
        else:
            self.criterion = nn.HuberLoss(delta=1.)
            
        # self.n_model_params = count_parameters(self)
        # self.log("n_model_params", self.n_model_params)

    def forward(self, x):
        output = self.network(x)
        if self.use_confidence:
            output[:,self.output_size:] = F.relu(output[:,self.output_size:])
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        # print(x.size(), y.size())
        # y = y.view(y.size(1)*y.size(0), y.size(2))
        # x = x.view(x.size(1)*x.size(0), x.size(2), x.size(3))
        y_pred = self(x)
        # loss = self.criterion(y_pred[:, JOINT_MASK==1], y[:, JOINT_MASK==1])
#             predloss = self.criterion(yhat[:, self.zeroout_mask==0], y[:, self.zeroout_mask==0])
#             predloss = torch.mean(predloss, dim=0)
#             confloss = self.criterion(yloss[:, self.zeroout_mask==0], predloss)
#             loss = torch.mean(predloss+confloss)
        if self.use_confidence:
            # confidence loss
            yhat = y_pred[:,:(self.n_column_targets*self.nlev+self.n_scalar_targets)//2]
            ylhat = y_pred[:,(self.n_column_targets*self.nlev+self.n_scalar_targets)//2:]
            
            yhat = yhat[:,~self.zeroout_mask]
            ylhat = ylhat[:,~self.zeroout_mask]
            y = y[:,~self.zeroout_mask]
            
            yhatloss = self.criterion(yhat, y)
            # yhatloss = torch.mean(yhatloss, dim=0)
            confloss = self.criterion(ylhat, yhatloss)
            loss = torch.mean(yhatloss+confloss)
            # From 1st place:
            # def loss_fn(x, preds):
            #     confidence = preds[:, col_num_y:]
            #     preds = preds[:, :col_num_y]
            #     loss = tf.math.abs(x-preds)
            #     loss = loss*loss_mask
            #     loss_2 = tf.math.abs(loss-confidence)
            #     loss_2 = loss_2*loss_mask
            #     return tf.reduce_mean(loss+loss_2)
            self.log('train_loss_yhat', torch.mean(yhatloss))#, on_epoch=True, on_step=False, sync_dist=True)#, sync_dist=True)#, prog_bar=True, on_epoch=True, on_step=False, logger=True)
        else: # normal loss
            y_pred = y_pred[:,~self.zeroout_mask]
            y = y[:,~self.zeroout_mask]
            loss = self.criterion(y_pred, y)
        # todo: confidence loss
        self.log('train_loss', loss)#, on_epoch=True, on_step=False, sync_dist=True)#, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        # print(x.size(), y.size())
        # y = y.view(y.size(1)*y.size(0), y.size(2))
        # x = x.view(x.size(1)*x.size(0), x.size(2), x.size(3))
        
        # if torch.cuda.is_available(): # GPU time (if using GPU)
        #     torch.cuda.synchronize()
        #     start_gpu_time = time.time()
        # else: # CPU time
        #     start_cpu_time = time.time()
            
        y_pred = self(x)
        
        # if torch.cuda.is_available(): # Measure GPU time (if using GPU)
        #     torch.cuda.synchronize()
        #     gpu_inference_time = time.time() - start_gpu_time
        #     self.gpu_times.append(gpu_inference_time)
        # else: # Measure CPU time
        #     cpu_inference_time = time.time() - start_cpu_time
        #     self.cpu_times.append(cpu_inference_time)
            
        if self.use_confidence:
            yhat = y_pred[:,:(self.n_column_targets*self.nlev+self.n_scalar_targets)//2]
            ylhat = y_pred[:,(self.n_column_targets*self.nlev+self.n_scalar_targets)//2:]
        
        y_std = Y_STD.to(y.device)
        y_mean = Y_MEAN.to(y.device)
        
        y = (y * y_std) + y_mean
        yhat[:, y_std < (1.1 * ERR)] = 0
        yhat = (yhat * y_std) + y_mean

        val_score = r2_score(yhat, y)
        self.log('val_score', val_score, on_epoch=True, on_step=False, sync_dist=True)#, logger=True, prog_bar=True)

        yhatloss = self.criterion(yhat, y)
        confloss = self.criterion(ylhat, yhatloss)
        loss = torch.mean(yhatloss+confloss)
        self.log('val_loss_yhat', torch.mean(yhatloss), on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_loss', loss, on_epoch=True, on_step=False, sync_dist=True)

        # yhat[:, ADJUSTMENT_MASK==0] = y[:, ADJUSTMENT_MASK==0]

        # yhat[:, MASK==0] = 0
        # y[:, MASK==0] = 0
        yhat[:, self.zeroout_index] = 0
        y[:, self.zeroout_index] = 0

        
        masked_val_score = r2_score(yhat, y)
        # self.val_score.append(val_score)
        # self.masked_val_score.append(masked_val_score)
        self.log('masked_val_score', masked_val_score, on_epoch=True, on_step=False, sync_dist=True)#, logger=True, prog_bar=True)
        # return {"val_score": val_score, "masked_val_score": masked_val_score}
        # self.log('n_model_params', self.n_model_params, on_epoch=True, on_step=False)#, logger=True, prog_bar=True)
        return val_score
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
            
        y_pred = self(x)
            
        if self.use_confidence:
            y_pred = y_pred[:,:(self.n_column_targets*self.nlev+self.n_scalar_targets)//2]
        
        y_std = Y_STD.to(y.device)
        y_mean = Y_MEAN.to(y.device)
        
        y = (y * y_std) + y_mean
        y_pred[:, y_std < (1.1 * ERR)] = 0
        y_pred = (y_pred * y_std) + y_mean

        test_score = r2_score(y_pred, y)
        self.log('test_score', test_score, on_epoch=True, on_step=False, sync_dist=True)#, logger=True, prog_bar=True)

        y_pred[:, self.zeroout_index] = 0
        y[:, self.zeroout_index] = 0

        masked_test_score = r2_score(y_pred, y)
        self.log('masked_test_score', masked_test_score, on_epoch=True, on_step=False, sync_dist=True)#, logger=True, prog_bar=True)
        return test_score

    # def on_validation_epoch_end(self):
    #     avg_val_score = statistics.fmean(self.val_score)
    #     avg_masked_val_score = statistics.fmean(self.masked_val_score)
    #     self.log("val_score", avg_val_score, sync_dist=True)
    #     self.log("masked_val_score", avg_masked_val_score, sync_dist=True)
    #     self.val_score.clear()
    #     self.masked_val_score.clear()
        
#     def on_train_end(self):
#         # Calculate and log average inference times
        
#         if len(self.cpu_times) > 0:
#             avg_cpu_time = sum(self.cpu_times) / len(self.cpu_times)
#             self.log("CPU_inference_time", avg_cpu_time)
#         if len(self.gpu_times) > 0:
#             avg_gpu_time = sum(self.gpu_times) / len(self.gpu_times) if self.gpu_times else None
#             self.log("GPU_inference_time", avg_gpu_time)
    
    
#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         x = x.to(self.device)
#         # start_time = time.time()
#         # _ = self(x)
#         # inference_time = time.time() - start_time
#         # self.log("inference_time", inference_time)
#         # self.log("n_model_params", self.n_model_params)
        
#         # GPU time (if using GPU)
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#             start_gpu_time = time.time()
#         else: # CPU time
#             start_cpu_time = time.time()
        
#         # Perform inference
#         output = self(x)
        
#         # Measure GPU time (if using GPU)
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#             gpu_inference_time = time.time() - start_gpu_time
#             self.gpu_times.append(gpu_inference_time)
#         else:
#             # Measure CPU time
#             cpu_inference_time = time.time() - start_cpu_time
#             self.cpu_times.append(cpu_inference_time)
#         print(self.cpu_times, self.gpu_times)

#         return output
    
#     def on_test_end(self):
#         # Calculate and log average inference times
        
#         if len(self.cpu_times) > 0:
#             avg_cpu_time = sum(self.cpu_times) / len(self.cpu_times)
#             self.log("CPU_inference_time", avg_cpu_time)
#         if len(self.gpu_times) > 0:
#             avg_gpu_time = sum(self.gpu_times) / len(self.gpu_times) if self.gpu_times else None
#             self.log("GPU_inference_time", avg_gpu_time)
        
    def configure_optimizers(self):

#         ###################################
#         LEARNING_RATE = 6.5e-4

#         optimizer = optim.AdamW(self.parameters(), lr=LEARNING_RATE, weight_decay=2e-3)

#         milestones = [3, 6, 8, 11, 14, 17]  # 4번째 에포크(인덱스 3)에서 학습률 변경
#         gamma = 0.65
#         scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

#         return [optimizer], [scheduler]
    
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=2)
        if self.scheduler == 'None':
            return optimizer
        elif self.scheduler == 'cosanh':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(2*3e6/self.batch_size), T_mult=2)

            return {
                "optimizer": optimizer,
                "lr_scheduler" : {
                    "scheduler": scheduler,
                    "interval": "step"
                },
            }
        elif self.scheduler == 'reduce_plat':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=2)
            return {
                "optimizer": optimizer,
                "lr_scheduler" : {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val_score"
                },
            }




JOINT_MASK = MASK * ADJUSTMENT_MASK




def r2_score(y_pred, y_true):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()




available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
available_gpus




# import time

# class InferenceTimeCallback(Callback):
#     def __init__(self):
#         self.inference_time = None

#     def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
#         self.start_time = time.time()

#     def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
#         # Measure the time taken for the batch inference
#         inference_time = time.time() - self.start_time
#         trainer.logger.log_metrics({"inference_time_per_batch": inference_time})




default_config = {
    'encode_dim': 200,
    'hidden_dim': 100,
    'iter_dim': 1000,
    'lstm_layers': 2,
    'dropout_rate': 0.01,
    'learning_rate': 5e-3,
    'weight_decay': 2e-4,
    'batch_size': 512,
    'scheduler': 'cosanh',
}

# search_space = {
#     "encode_dim": tune.qrandint(50, 500, 50),
#     "hidden_dim": tune.qrandint(50, 400, 50),
#     "iter_dim": tune.qrandint(300, 1000, 100),
#     'lstm_layers': tune.randint(1, 10),
# }

search_space = {
    "encode_dim": tune.qrandint(10, 800, 10),
    "hidden_dim": tune.qrandint(10, 800, 10),
    "iter_dim": tune.qrandint(100, 900, 10),
    'lstm_layers': tune.randint(1, 10),
    'dropout_rate': tune.choice([0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3]),
    'learning_rate': tune.choice([1e-3, 5e-3, 6.5e-4, 1e-4]),
    'weight_decay': tune.choice([2e-4, 0.01]),
    'batch_size': tune.choice([256, 512, 1024, 2048]),
    'scheduler': tune.choice(['None', 'cosanh', 'reduce_plat']),
}

# small_search_space = {
#     "encode_dim": tune.qrandint(10, 20, 10),
#     "hidden_dim": tune.qrandint(10, 20, 10),
#     "iter_dim": tune.qrandint(10, 80, 10),
#     'lstm_layers': tune.randint(1, 2),
#     'dropout_rate': tune.choice([0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3]),
#     'learning_rate': tune.choice([1e-3, 5e-3, 6.5e-4, 1e-4]),
#     'weight_decay': tune.choice([2e-4, 0.01]),
#     'batch_size': tune.choice([256, 512, 1024, 2048]),
#     'scheduler': tune.choice(['None', 'cosanh', 'reduce_plat']),
# }




# small_search_space = {
#     "encode_dim": tune.qrandint(300, 400, 10),
#     "hidden_dim": tune.qrandint(390, 400, 10),
#     "iter_dim": tune.qrandint(500, 900, 10),
#     'lstm_layers': tune.randint(9, 10),
#     'dropout_rate': tune.choice([0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3]),
#     'learning_rate': tune.choice([1e-3, 5e-3, 6.5e-4, 1e-4]),
#     'weight_decay': tune.choice([2e-4, 0.01]),
#     'batch_size': tune.choice([256, 512, 1024, 2048]),
#     'scheduler': tune.choice(['None', 'cosanh', 'reduce_plat']),
# }




# feature_target_lengths = [n_column_features,  n_scalar_features, n_column_targets, n_scalar_targets]

# test_config = {
#     'encode_dim': 50,
#     'hidden_dim': 50,
#     'iter_dim': 300,
#     'lstm_layers': 2,
# }

# model = HighResLeapModel(
#     feature_target_lengths=feature_target_lengths,
#     config=test_config,
#     zeroout_index=zeroout_index,
#     use_confidence=True,
# )

# print(count_parameters(model))

# train_loader, val_loader, test_loader = get_dataloaders()
# for b in test_loader:
#     x, y = b
#     print(model(x))
#     break




# class PostFitCallback(Callback):
#     def on_fit_end(self, trainer, pl_module):
#         print("Callback: Fit process has ended.")

feature_target_lengths = [n_column_features,  n_scalar_features, n_column_targets, n_scalar_targets]
logs_per_epoch = 50

def train_func(config):
    model = HighResLeapModel(
        feature_target_lengths=feature_target_lengths,
        config=config,
        zeroout_index=zeroout_index,
        use_confidence=True,
    )
    
    n_model_params = count_parameters(model)
    # if n_model_params > 10e6:
        # raise Exception(f'Model has {n_model_params}; more than 10e6 params, not continuing')

    # if n_model_params > 10e6:
    #     print(f'Model has {n_model_params}; more than 10e6 params, not continuing')
    #     train.report(success=False)#, num_parameters=num_parameters)
    #     return
        
    test_loader = get_valdataloader(config['batch_size'])
    
    x, y = next(iter(test_loader))
    x = x.to(model.device)
    gpu_times = []
    cpu_times = []
    model.to('cuda')
    x = x.to(model.device)
    for it in range(100):
        torch.cuda.synchronize()
        start_gpu_time = time.time()
        output = model(x)
        torch.cuda.synchronize()
        gpu_inference_time = time.time() - start_gpu_time
        gpu_times.append(gpu_inference_time)
        
    model.to('cpu')
    x = x.to(model.device)
    for it in range(100):
        start_cpu_time = time.time()
        output = model(x)
        cpu_inference_time = time.time() - start_cpu_time
        cpu_times.append(cpu_inference_time)
 
    avg_gpu_time = sum(gpu_times) / len(gpu_times)
    avg_cpu_time = sum(cpu_times) / len(cpu_times)
    time_metrics = {"avg_gpu_time_seconds": avg_gpu_time,
                    "avg_cpu_time_seconds": avg_cpu_time,
                    "n_model_params": n_model_params}
    
    trial_dir = train.get_context().get_trial_dir()
    with open(os.path.join(trial_dir, 'model_complexity.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        keys = []
        values = []
        for key, value in time_metrics.items():
            keys.append(key)
            values.append(value)
        writer.writerow(keys)
        writer.writerow(values)
        
    train_loader = get_traindataloader(config['batch_size'])
    val_loader = get_valdataloader(config['batch_size'])
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_score",
        mode="max",
        save_top_k=1,
        filename="{val_score:.4f}-best_model",
        dirpath=trial_dir,
    )
    
    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        precision = 16,
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback(),
                  #  TuneReportCheckpointCallback(
                  #       metrics={"train_loss": "train_loss", "train_loss_yhat": "train_loss_yhat"},
                  #       filename="trainer_train.ckpt", on="train_epoch_end"),
                  #  TuneReportCheckpointCallback(
                  #       metrics={"val_score": "val_score", "masked_val_score": "masked_val_score"},
                  #       filename="trainer_val.ckpt", on="validation_epoch_end"),
                  #  PostFitCallback(),
                   checkpoint_callback,
                  ],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
        log_every_n_steps=int(3e6/(logs_per_epoch*config['batch_size']))
    )
    trainer = prepare_trainer(trainer)
    # torch.manual_seed(41)
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model, train_loader, val_loader)
    
    # Nothing works after fit
#     print('Beginning testing')
#     test_loader = get_testdataloader(config['batch_size'])
#     trainer.test(model, test_loader)
#     print('Finished testing')




scaling_config = ScalingConfig(
    num_workers=1, use_gpu=True, resources_per_worker={"CPU": 2, "GPU": 1}
)

run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="val_score",
        checkpoint_score_order="max",
    ),
    # checkpoint_config=CheckpointConfig(checkpoint_frequency=0),
    # checkpoint_config = air.CheckpointConfig(num_to_keep=0),
    storage_path="/work/bd1179/b309215/ClimSimKaggle/leap-climsim-kaggle-5th/ray_results_3e6train/",
    # storage_path="/scratch/b/b309215/ray_results_test/",
)




# Define a TorchTrainer without hyper-parameters for Tuner
ray_trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    run_config=run_config,
)




# The maximum training epochs
num_epochs = 20

# Number of sampls from parameter space
num_samples = 10000




# import logging
# logging.basicConfig(level=logging.DEBUG)

def tune_mnist_asha(num_samples=10, num_epochs=10):
    # scheduler = ASHAScheduler(max_t=num_epochs, grace_period=4, reduction_factor=2)
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=4, reduction_factor=2)

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        # param_space={"params": search_space},
        # param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="masked_val_score",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()


results = tune_mnist_asha(num_samples=num_samples, num_epochs=num_epochs)
# results = tune_mnist_asha(num_samples=4, num_epochs=3)




# get_ipython().system('scancel 10609055')