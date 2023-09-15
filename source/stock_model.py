import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
# import torch.nn.MaxPool1d as maxpool
sys.path.append('../../')
sys.path.append('../../../')
from src import config

class LossFunc():
    def __init__(self, omega=0.0):
        self.omega = omega
        self.relu = torch.nn.ReLU()
        self.device = torch.device('cpu')
    def regression_loss(self,x,y,_close):
        # percent change
        chg = x.add(_close.mul(-1)).div(_close)
        _chg = y.add(_close.mul(-1)).div(_close)
#         chg = (chg - chg.mean()).div(chg.std())
#         _chg = (_chg - _chg.mean()).div(_chg.std())
        # regression_loss
        regression_loss = chg.add(_chg.mul(-1)).pow(2).mean()
        loss = regression_loss
        return loss
    def rank_loss(self,x,y,_close):
        # percent change
        chg = x.add(_close.mul(-1)).div(_close)
        _chg = y.add(_close.mul(-1)).div(_close)
        chg = (chg - chg.mean()).div(chg.std())
        _chg = (_chg - _chg.mean()).div(_chg.std())
        # regression_loss
        regression_loss = chg.add(_chg.mul(-1)).pow(2).mean()
        # rank_loss
        ones = torch.ones(chg.shape).to(x.device)
        item_x = chg.mul(ones.t()) - ones.mul(chg.t())
        item_y = _chg.mul(ones.t()) - ones.mul(_chg.t())
        rank_loss = self.relu(item_x.mul(item_y).mul(-1)).mean()
        loss = regression_loss.add(rank_loss.mul(self.omega))
        return loss
    def dw_rank_loss(self,x,y,_close):
        # percent change
        chg = x.add(_close.mul(-1)).div(_close)
        _chg = y.add(_close.mul(-1)).div(_close)
        # regression_loss
        regression_loss = chg.add(_chg.mul(-1)).pow(2).mean()
        # rank_loss
        chg = (chg - chg.mean()).div(chg.std())
        _chg = (_chg - _chg.mean()).div(_chg.std())
        ones = torch.ones(chg.shape).to(x.device)
        item_x = chg.mul(ones.t()) - ones.mul(chg.t())
        item_y = _chg.mul(ones.t()) - ones.mul(_chg.t())
        relu = torch.nn.ReLU()
        # rank_weight
        rank_loss = relu(item_x.mul(item_y).mul(-1))
        _, temp = _chg.sort(dim=0, descending=True)
        _, _chg_rank = temp.sort(dim=0) 
        rank_x = _chg_rank.mul(ones.t())
        rank_y = ones.mul(_chg_rank.t())
        gap_weight = (rank_x - rank_y).div(ones.shape[0]-1).pow(2)
        pos_weight = torch.exp(-torch.min(rank_x,rank_y).div(ones.shape[0]))
        rank_weight = gap_weight.mul(pos_weight)
        rank_loss = rank_loss.mul(rank_weight).mean()
        return regression_loss.add(rank_loss*self.omega)
    def topkprob(self, vec, k=5):
        vec_sort = torch.sort(vec)[-1::-1]
        print(vec_sort)
        topk = vec_sort[:k]
        print(topk)
        ary = np.arange(k)
        return torch.mul([torch.exp(topk[i]) / torch.sum(torch.exp(topk[i:])) for i in ary])

    def listwise_cost(self, x, y,_close):
        chg = x.add(_close.mul(-1)).div(_close)
        _chg = y.add(_close.mul(-1)).div(_close)
        list_ans=_chg
        list_pred=chg
        return - torch.sum(self.topkprob(list_ans) * torch.log(self.topkprob(list_pred)))
    
    def listnet_loss(self,x,y,_close):
        chg = x.add(_close.mul(-1))
        _chg = y.add(_close.mul(-1))
        y_pred=_chg
        y_true=chg
#         print("top1_target2122:",x)
#         print("top1_predict:",y)
        #y是true 
#         return torch.mean(listmle_loss.sum(dim=1))
        regression_loss = chg.add(_chg.mul(-1)).pow(2).mean()

        pre_prob_dist = F.softmax(_chg,dim=0)
        gt_prob_dist = F.softmax(chg,dim=0)
        per_example_loss= -100*torch.mul(gt_prob_dist,torch.log(pre_prob_dist))
        batch_loss = torch.mean(per_example_loss)
        loss=regression_loss.add(batch_loss*self.omega)
        return loss
        #_chg是true
#         print("top1_target2122:",x)
#         print("top1_predict:",y)
#         print("close22:",torch.sum(top1_target * torch.log(top1_predict)))
        ret = -torch.mean(torch.sum(top1_target * torch.log(top1_predict)))
#         ret=ret.add(10000000000)
        return ret

class TimeConv(torch.nn.Module):
    def __init__(self, conv_size, window_size, expand_feature_size, **kwargs):
        super(TimeConv, self).__init__(**kwargs)
        self.window_size = window_size
        self.conv_size = conv_size
        self.expand_feature_size = expand_feature_size
        self.device = torch.device('cpu')
    def mean_kernel(self, x):
        return x.mean(axis=1).reshape(x.shape[0],1,x.shape[2])
    def std_kernel(self, x):
        return x.std(axis=1).reshape(x.shape[0],1,x.shape[2])
    def rank_kernel(self, x):
        _, idx = x[:,-1:].sort(dim=0, descending=True)
        _, rank = idx.sort(dim=0) 
        return rank/x.shape[0]
    def max_kernel(self, x):    
        values,_ = x.max(axis=1)
        return values.reshape(x.shape[0],1,x.shape[2])
    def min_kernel(self, x):    
        values,_ = x.min(axis=1)
        return values.reshape(x.shape[0],1,x.shape[2])
    def forward(self, x):
        conv_result = torch.zeros((x.shape[0],0,5*self.expand_feature_size))
        begin_index = x.shape[1] - (self.conv_size+self.window_size-1)
        if begin_index < 0:
            raise Exception('数据集或设置出现错误，可能与expand_window_size、expand_feature参数有关！')
        for n in range(0,self.window_size):
            target = x[:,begin_index+n:begin_index+self.conv_size+n,:self.expand_feature_size]
            _mean = self.mean_kernel(target)
            _std = self.std_kernel(target)
            _rank = self.rank_kernel(target)
            _max = self.max_kernel(target)
            _min = self.min_kernel(target)
            target_result = torch.cat((_mean,_std,_rank,_max,_min),axis=2)
            conv_result = torch.cat((conv_result.to(x.device), target_result.to(x.device)) ,axis=1)
        return conv_result
       



class Rank_CNN(torch.nn.Module):
    def __init__(self, plan, suffix_name):
        super(Rank_CNN, self).__init__()
        self.model_name = 'Rank_CNN'
        self.suffix_name = suffix_name
        self.hyperparameter = plan.hyperparameter[self.model_name]
        self.model_file_path = config.MINING_MODEL_FILE_PATH + self.model_name  + '/' + suffix_name
        if not os.path.exists(self.model_file_path):
            os.makedirs(self.model_file_path)
        self.loss_func = LossFunc(self.hyperparameter['omega'])
        self.window_size = plan.window_size
        self.week_timeconv = TimeConv(conv_size=5, window_size=self.window_size, expand_feature_size=len(plan.expand_feature))
        self.month_timeconv = TimeConv(conv_size=20, window_size=self.window_size, expand_feature_size=len(plan.expand_feature))
        self.cnn = torch.nn.Conv1d(in_channels=len(plan.feature)+10*len(plan.expand_feature),out_channels=self.hyperparameter['out_channels'],kernel_size=self.hyperparameter['kernel_size'])
        self.leaky_relu = torch.nn.LeakyReLU(self.hyperparameter['leaky_relu_alpha'])
        self.linear = torch.nn.Linear(in_features=
                    (self.window_size+1-self.hyperparameter['kernel_size']) , out_features=len(plan.label))
    def forward(self, data):
        week_x = self.week_timeconv(data.x)
        month_x = self.month_timeconv(data.x)
        x, edge_index =  torch.cat((data.x[:,-self.window_size:,:], week_x, month_x) ,axis=2),data.edge_index  
#         print(x.shape)
        cnn_x = x.permute(0,2,1)
#         print(cnn_x.shape)
        part_x =self.cnn(cnn_x)
        x = part_x[:,-1,:] 
        x = self.leaky_relu(x)
        x = self.linear(x)
        return x
    def loss_function(self,x,y,_close):
        return self.loss_func.dw_rank_loss(x,y,_close)
    

    
class Regression_CNN(torch.nn.Module):
    def __init__(self, plan, suffix_name):
        super(Regression_CNN, self).__init__()
        self.model_name = 'Regression_CNN'
        self.suffix_name = suffix_name
        self.hyperparameter = plan.hyperparameter[self.model_name]
        self.model_file_path = config.MINING_MODEL_FILE_PATH + self.model_name  + '/' + suffix_name
        if not os.path.exists(self.model_file_path):
            os.makedirs(self.model_file_path)
        self.loss_func = LossFunc(self.hyperparameter['omega'])
        self.window_size = plan.window_size
        self.week_timeconv = TimeConv(conv_size=5, window_size=self.window_size, expand_feature_size=len(plan.expand_feature))
        self.month_timeconv = TimeConv(conv_size=20, window_size=self.window_size, expand_feature_size=len(plan.expand_feature))
        self.cnn = torch.nn.Conv1d(in_channels=69,out_channels=self.hyperparameter['out_channels'],kernel_size=self.hyperparameter['kernel_size'])
        self.leaky_relu = torch.nn.LeakyReLU(self.hyperparameter['leaky_relu_alpha'])
        self.linear = torch.nn.Linear(in_features=(self.hyperparameter['out_channels']+1-self.hyperparameter['kernel_size']) , out_features=len(plan.label))
    def forward(self, data):
        week_x = self.week_timeconv(data.x)
        month_x = self.month_timeconv(data.x)
        x, edge_index =  torch.cat((data.x[:,-self.window_size:,:], week_x, month_x) ,axis=2),data.edge_index  
        cnn_x = x.permute(0,2,1)
        part_x =self.cnn(cnn_x)
        x = part_x[:,-1,:] 
        x = self.relu(x)
        x = self.linear(x)
        return x
    def loss_function(self,x,y,_close):
        return self.loss_func.regression_loss(x,y,_close)
    
    
class Rank_CNN_GAT(torch.nn.Module):
    def __init__(self, plan, suffix_name):
        super(Rank_CNN_GAT, self).__init__()
        self.model_name = 'Rank_CNN_GAT'
        self.suffix_name = suffix_name
        self.hyperparameter = plan.hyperparameter[self.model_name]
        self.model_file_path = config.MINING_MODEL_FILE_PATH + self.model_name  + '/' + suffix_name
        if not os.path.exists(self.model_file_path):
            os.makedirs(self.model_file_path)
        self.loss_func = LossFunc(self.hyperparameter['omega'])
        self.window_size = plan.window_size
        self.week_timeconv = TimeConv(conv_size=5, window_size=self.window_size, expand_feature_size=len(plan.expand_feature))
        self.month_timeconv = TimeConv(conv_size=20, window_size=self.window_size, expand_feature_size=len(plan.expand_feature))
        self.cnn = torch.nn.Conv1d(in_channels=69,out_channels=self.hyperparameter['out_channels'],kernel_size=self.hyperparameter['kernel_size'])
        self.pl=torch.nn.MaxPool1d(3,stride=2)
        self.gnn = GATConv(16-self.hyperparameter['kernel_size'],self.hyperparameter['gnn_hidden_size'])
        self.leaky_relu = torch.nn.LeakyReLU(self.hyperparameter['leaky_relu_alpha'])
        self.linear = torch.nn.Linear(in_features=(16-self.hyperparameter['kernel_size']+self.hyperparameter['gnn_hidden_size']) , out_features=len(plan.label))
    def forward(self, data):
        week_x = self.week_timeconv(data.x)
        month_x = self.month_timeconv(data.x)
        x, edge_index = torch.cat((data.x[:,-self.window_size:,:], week_x, month_x) ,axis=2),data.edge_index
        print(x.shape)
        cnn_x = x.permute(0,2,1)
        part_x =self.cnn(cnn_x)
        time_x = part_x[:,-1,:] 
        space_x = self.gnn(time_x, edge_index)  
        x = torch.cat((time_x,space_x),1)
        x = self.leaky_relu(x)
        x = self.linear(x)
        return x
    def loss_function(self,x,y,_close):
        return self.loss_func.dw_rank_loss(x,y,_close)
    
class Rank_CNN_GAT11(torch.nn.Module):
    def __init__(self, plan, suffix_name):
        super(Rank_CNN_GAT11, self).__init__()
        self.model_name = 'Rank_CNN_GAT11'
        self.suffix_name = suffix_name
        self.hyperparameter = plan.hyperparameter[self.model_name]
        self.model_file_path = config.MINING_MODEL_FILE_PATH + self.model_name  + '/' + suffix_name
        if not os.path.exists(self.model_file_path):
            os.makedirs(self.model_file_path)
        self.loss_func = LossFunc(self.hyperparameter['omega'])
        self.window_size = plan.window_size
        self.week_timeconv = TimeConv(conv_size=5, window_size=self.window_size, expand_feature_size=len(plan.expand_feature))
        self.month_timeconv = TimeConv(conv_size=20, window_size=self.window_size, expand_feature_size=len(plan.expand_feature))
        self.cnn = torch.nn.Conv1d(in_channels=69,out_channels=self.hyperparameter['out_channels'],kernel_size=self.hyperparameter['kernel_size'])
        self.gnn = GATConv(self.hyperparameter['lstm_hidden_size'],self.hyperparameter['gnn_hidden_size'])
        self.leaky_relu = torch.nn.LeakyReLU(self.hyperparameter['leaky_relu_alpha'])
        self.linear = torch.nn.Linear(in_features=(self.hyperparameter['out_channels']+1-self.hyperparameter['kernel_size']+self.hyperparameter['gnn_hidden_size']) , out_features=len(plan.label))
    def forward(self, data):
        week_x = self.week_timeconv(data.x)
        month_x = self.month_timeconv(data.x)
        x, edge_index =  torch.cat((data.x[:,-self.window_size:,:], week_x, month_x) ,axis=2),data.edge_index  
        cnn_x = x.permute(0,2,1)
        part_x =self.cnn(cnn_x)
        time_x = part_x[:,-1,:] 
        space_x = self.gnn(time_x, edge_index)  
        x = torch.cat((time_x,space_x),1)
        x = self.relu(x)
        x = self.linear(x)
        return x
    def loss_function(self,x,y,_close):
        return self.loss_func.dw_rank_loss(x,y,_close)
    


class MHSA(torch.nn.Module):
    def __init__(self, num_heads, dim):
        super().__init__()
        # Q, K, V 转换矩阵，这里假设输入和输出的特征维度相同
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)
        self.num_heads = num_heads

    def forward(self, x):
        B, N, C = x.shape
        # 生成转换矩阵并分多头
        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        
        # 点积得到attention score
        attn = q @ k.transpose(2, 3) * (x.shape[-1] ** -0.5)
        attn = attn.softmax(dim=-1)
        
        # 乘上attention score并输出
        v = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        return v

class DeepAlpha018D(torch.nn.Module):
    def __init__(self, plan, suffix_name):
        super(DeepAlpha018D, self).__init__()
        self.model_name = 'rank_STAN'
        self.suffix_name = suffix_name
        self.hyperparameter = plan.hyperparameter[self.model_name]
        self.model_file_path = config.MINING_MODEL_FILE_PATH + self.model_name  + '/' + suffix_name
        if not os.path.exists(self.model_file_path):
            os.makedirs(self.model_file_path)
        self.loss_func = LossFunc(self.hyperparameter['omega'])
        self.window_size = plan.window_size
        self.week_timeconv = TimeConv(conv_size=5, window_size=self.window_size, expand_feature_size=len(plan.expand_feature))
        self.month_timeconv = TimeConv(conv_size=20, window_size=self.window_size, expand_feature_size=len(plan.expand_feature))
        self.lstm = torch.nn.LSTM(input_size=len(plan.feature)+10*len(plan.expand_feature), hidden_size=self.hyperparameter['lstm_hidden_size'],
                 num_layers=self.hyperparameter['num_layers'], batch_first=True)
        self.mhsa=MHSA(num_heads=4,dim=self.hyperparameter['lstm_hidden_size'])
#         self.cnn = torch.nn.Conv1d(in_channels=5,out_channels=self.hyperparameter['cnn_filters'],kernel_size=self.hyperparameter['cnn_kernel_size'])
        self.cnn = torch.nn.Conv1d(in_channels=69,out_channels=15,kernel_size=self.hyperparameter['kernel_size'])

        #
#         self.gnn = GATConv(self.hyperparameter['lstm_hidden_size']+4,self.hyperparameter['gnn_hidden_size'])
        self.gnn = GATConv(self.hyperparameter['lstm_hidden_size']+(16-self.hyperparameter['kernel_size']),self.hyperparameter['gnn_hidden_size'])

        self.leaky_relu = torch.nn.LeakyReLU(self.hyperparameter['leaky_relu_alpha'])
        self.linear = torch.nn.Linear(in_features=self.hyperparameter['gnn_hidden_size']+self.hyperparameter['lstm_hidden_size']+(16-self.hyperparameter['kernel_size']) , out_features=len(plan.label))
    def forward(self, data):
        week_x = self.week_timeconv(data.x)
        month_x = self.month_timeconv(data.x)
        x, edge_index =  torch.cat((data.x[:,-self.window_size:,:], week_x, month_x) ,axis=2),data.edge_index  
        global_x,hidden = self.lstm(x)
        global_x = self.mhsa(global_x)
        
#         print(global_x.shape)
#         [3413,5,64]
        cnn_x = x.permute(0,2,1)
#         print(cnn_x.shape)
        part_x =self.cnn(cnn_x)
#         pert_x=self.pl(part_x)
#         print(part_x.shape)
#         [3413,5,64]
        
        time_x = torch.cat((global_x,part_x), 2)
#         print(time_x.shape)
        time_x = time_x[:,-1,:] 
        
#         print(time_x.shape)
#         gat_x = x[:,-1,:] 
#         print(gat_x.shape)
#         space_x = self.gnn(time_x, edge_index)

        space_x = self.gnn(time_x, edge_index)
#         print(space_x.shape)
        x = torch.cat((time_x,space_x),1)
#         print(x.shape)
        
        x = self.leaky_relu(x)
        x = self.linear(x)
        return x
    def loss_function(self,x,y,_close):
        return self.loss_func.dw_rank_loss(x,y,_close)
    
class Rank_STAN_new(torch.nn.Module):
    def __init__(self, plan, suffix_name):
        super(Rank_STAN, self).__init__()
        self.model_name = 'Rank_STAN'
        self.suffix_name = suffix_name
        self.hyperparameter = plan.hyperparameter[self.model_name]
        self.model_file_path = config.MINING_MODEL_FILE_PATH + self.model_name  + '/' + suffix_name
        if not os.path.exists(self.model_file_path):
            os.makedirs(self.model_file_path)
        self.loss_func = LossFunc(self.hyperparameter['omega'])
        self.window_size = plan.window_size
        self.week_timeconv = TimeConv(conv_size=5, window_size=self.window_size, expand_feature_size=len(plan.expand_feature))
        self.month_timeconv = TimeConv(conv_size=20, window_size=self.window_size, expand_feature_size=len(plan.expand_feature))
        self.lstm = torch.nn.LSTM(input_size=len(plan.feature)+10*len(plan.expand_feature), hidden_size=self.hyperparameter['lstm_hidden_size'],
                 num_layers=self.hyperparameter['num_layers'], batch_first=True)
        self.mhsa=MHSA(num_heads=4,dim=self.hyperparameter['lstm_hidden_size'])

        self.cnn = torch.nn.Conv1d(in_channels=len(plan.feature)+10*len(plan.expand_feature),out_channels=15,kernel_size=self.hyperparameter['kernel_size'])
        self.pl= torch.nn.MaxPool1d(2, stride=1)

#         self.gnn = GATConv(self.hyperparameter['lstm_hidden_size']+((16-self.hyperparameter['kernel_size']-3)//2+1),self.hyperparameter['gnn_hidden_size'])
        self.gnn = GATConv(self.hyperparameter['lstm_hidden_size']+((15-self.hyperparameter['kernel_size'])),self.hyperparameter['gnn_hidden_size'])

        self.leaky_relu = torch.nn.LeakyReLU(self.hyperparameter['leaky_relu_alpha'])
        self.linear = torch.nn.Linear(in_features=self.hyperparameter['gnn_hidden_size']+self.hyperparameter['lstm_hidden_size']+((15-self.hyperparameter['kernel_size'])) , out_features=len(plan.label))
    def forward(self, data):
        week_x = self.week_timeconv(data.x)
        month_x = self.month_timeconv(data.x)
        x, edge_index =  torch.cat((data.x[:,-self.window_size:,:], week_x, month_x) ,axis=2),data.edge_index  
        global_x,hidden = self.lstm(x)
        global_x = self.mhsa(global_x)

        cnn_x = x.permute(0,2,1)
        part_x =self.cnn(cnn_x)
        part_x=self.pl(part_x)
        time_x = torch.cat((global_x,part_x), 2)
        time_x = time_x[:,-1,:] 
        space_x = self.gnn(time_x, edge_index)
        x = torch.cat((time_x,space_x),1)
        x = self.leaky_relu(x)
        x = self.linear(x)
        return x
    def loss_function(self,x,y,_close):
        return self.loss_func.dw_rank_loss(x,y,_close)

    
class Rank_STAN(torch.nn.Module):
    def __init__(self, plan, suffix_name):
        super(Rank_STAN, self).__init__()
        print(891)
        self.model_name = 'Rank_STAN'
        self.suffix_name = suffix_name
        self.hyperparameter = plan.hyperparameter[self.model_name]
        self.model_file_path = config.MINING_MODEL_FILE_PATH + self.model_name  + '/' + suffix_name
        if not os.path.exists(self.model_file_path):
            os.makedirs(self.model_file_path)
        self.loss_func = LossFunc(self.hyperparameter['omega'])
        self.window_size = plan.window_size
        self.week_timeconv = TimeConv(conv_size=5, window_size=self.window_size, expand_feature_size=len(plan.expand_feature))
        self.month_timeconv = TimeConv(conv_size=20, window_size=self.window_size, expand_feature_size=len(plan.expand_feature))
        self.lstm = torch.nn.LSTM(input_size=len(plan.feature)+10*len(plan.expand_feature), hidden_size=self.hyperparameter['lstm_hidden_size'],
                 num_layers=self.hyperparameter['num_layers'], batch_first=True)
        self.mhsa=MHSA(num_heads=self.hyperparameter['lstm_hidden_size'],dim=self.hyperparameter['lstm_hidden_size'])
        self.bn = torch.nn.BatchNorm1d(self.hyperparameter['lstm_hidden_size']+(15-self.hyperparameter['kernel_size']),momentum=0.1)
        self.bn2 = torch.nn.BatchNorm1d(self.hyperparameter['gnn_hidden_size'],momentum=0.1)
        self.cnn = torch.nn.Conv1d(in_channels=len(plan.feature)+10*len(plan.expand_feature),out_channels=15,kernel_size=self.hyperparameter['kernel_size'])
        self.pl= torch.nn.MaxPool1d(2,stride=1)
        #pct 3,2
#         self.gnn = GATConv(self.hyperparameter['lstm_hidden_size']+((16-self.hyperparameter['kernel_size']-3)//2+1),self.hyperparameter['gnn_hidden_size'])
        self.gnn = GATConv(self.hyperparameter['lstm_hidden_size']+(15-self.hyperparameter['kernel_size']),self.hyperparameter['gnn_hidden_size'])

        self.leaky_relu = torch.nn.LeakyReLU(self.hyperparameter['leaky_relu_alpha'])
        self.linear = torch.nn.Linear(in_features=self.hyperparameter['gnn_hidden_size']+self.hyperparameter['lstm_hidden_size']+(15-self.hyperparameter['kernel_size']) , out_features=len(plan.label))
    def forward(self, data):
        week_x = self.week_timeconv(data.x)
        month_x = self.month_timeconv(data.x)
        x, edge_index =  torch.cat((data.x[:,-self.window_size:,:], week_x, month_x) ,axis=2),data.edge_index  
        global_x,hidden = self.lstm(x)
        global_x = self.mhsa(global_x)
        print(global_x.shape)
        cnn_x = x.permute(0,2,1)
        part_x =self.cnn(cnn_x)
        part_x=self.pl(part_x)
        print(part_x.shape)
        time_x = torch.cat((global_x,part_x),2)
        time_x = time_x[:,-1,:] 
        space_x = self.gnn(time_x, edge_index)
        x = torch.cat((time_x,space_x),1)
        x = self.leaky_relu(x)
        x = self.linear(x)
        return x
    def loss_function(self,x,y,_close):
        return self.loss_func.rank_loss(x,y,_close)
    
class Rank_LSTM_SA_CNN(torch.nn.Module):
    def __init__(self, plan, suffix_name):
        super(Rank_LSTM_SA_CNN, self).__init__()
        self.model_name = 'Rank_LSTM_SA_CNN'
        self.suffix_name = suffix_name
        print(self.model_name)
        self.hyperparameter = plan.hyperparameter[self.model_name]
        self.model_file_path = config.MINING_MODEL_FILE_PATH + self.model_name  + '/' + suffix_name
        if not os.path.exists(self.model_file_path):
            os.makedirs(self.model_file_path)
        self.loss_func = LossFunc(self.hyperparameter['omega'])
        self.window_size = plan.window_size
        self.week_timeconv = TimeConv(conv_size=5, window_size=self.window_size, expand_feature_size=len(plan.expand_feature))
        self.month_timeconv = TimeConv(conv_size=20, window_size=self.window_size, expand_feature_size=len(plan.expand_feature))
        self.lstm = torch.nn.LSTM(input_size=len(plan.feature)+10*len(plan.expand_feature), hidden_size=self.hyperparameter['lstm_hidden_size'],
                 num_layers=self.hyperparameter['num_layers'], batch_first=True)
        self.mhsa=MHSA(num_heads=4,dim=self.hyperparameter['lstm_hidden_size'])
        self.pl= torch.nn.MaxPool1d(2, stride=1)
        self.cnn = torch.nn.Conv1d(in_channels=69,out_channels=15,kernel_size=self.hyperparameter['kernel_size'])
#         self.gnn = GATConv(self.hyperparameter['lstm_hidden_size']+(16-self.hyperparameter['kernel_size']),self.hyperparameter['gnn_hidden_size'])
        self.leaky_relu = torch.nn.LeakyReLU(self.hyperparameter['leaky_relu_alpha'])
        self.linear = torch.nn.Linear(in_features=self.hyperparameter['lstm_hidden_size']+(15-self.hyperparameter['kernel_size']) , out_features=len(plan.label))
    def forward(self, data):
        week_x = self.week_timeconv(data.x)
        month_x = self.month_timeconv(data.x)
        x, edge_index =  torch.cat((data.x[:,-self.window_size:,:], week_x, month_x) ,axis=2),data.edge_index  
        global_x,hidden = self.lstm(x)
        global_x = self.mhsa(global_x)
        cnn_x = x.permute(0,2,1)
        part_x =self.cnn(cnn_x)
        part_x=self.pl(part_x)
        time_x = torch.cat((global_x,part_x), 2)
        x = time_x[:,-1,:] 
        x = self.leaky_relu(x)
        x = self.linear(x)
        return x
    def loss_function(self,x,y,_close):
        return self.loss_func.dw_rank_loss(x,y,_close)
    
    
class Rank_LSTM(torch.nn.Module):
    def __init__(self, plan, suffix_name):
        super(Rank_LSTM, self).__init__()
        self.model_name = 'Rank_LSTM'
        self.suffix_name = suffix_name
        self.hyperparameter = plan.hyperparameter[self.model_name]
        self.model_file_path = config.MINING_MODEL_FILE_PATH + self.model_name  + '/' + suffix_name
        if not os.path.exists(self.model_file_path):
            os.makedirs(self.model_file_path)
        self.window_size = plan.window_size
        self.loss_func = LossFunc(self.hyperparameter['omega'])
        self.lstm = torch.nn.LSTM(input_size=len(plan.feature)+10*len(plan.expand_feature),         hidden_size=self.hyperparameter['lstm_hidden_size'],num_layers=self.hyperparameter['num_layers'], batch_first=True)
        self.mhsa=MHSA(num_heads=4,dim=self.hyperparameter['lstm_hidden_size'])
        self.leaky_relu = torch.nn.LeakyReLU(self.hyperparameter['leaky_relu_alpha'])
        self.week_timeconv = TimeConv(conv_size=5, window_size=self.window_size, expand_feature_size=len(plan.expand_feature))
        self.month_timeconv = TimeConv(conv_size=20, window_size=self.window_size, expand_feature_size=len(plan.expand_feature))
        self.linear = torch.nn.Linear(in_features=self.hyperparameter['lstm_hidden_size'] , out_features=len(plan.label))
    def forward(self, data):
        week_x = self.week_timeconv(data.x)
        month_x = self.month_timeconv(data.x)
        x =  torch.cat((data.x[:,-self.window_size:,:], week_x, month_x) ,axis=2)
        x,hidden = self.lstm(x)
#         x = self.mhsa(x)
        x = x[:,-1,:]
        x = self.leaky_relu(x)
        x = self.linear(x)
        return x
    def loss_function(self,x,y,_close):
        return self.loss_func.dw_rank_loss(x,y,_close)
    

class Rank_LSTM_GAT(torch.nn.Module):
    def __init__(self, plan, suffix_name):
        super(Rank_LSTM_GAT, self).__init__()
        self.model_name = 'Rank_LSTM_GAT'
        self.suffix_name = suffix_name
        self.hyperparameter = plan.hyperparameter[self.model_name]
        self.model_file_path = config.MINING_MODEL_FILE_PATH + self.model_name  + '/' + suffix_name
        if not os.path.exists(self.model_file_path):
            os.makedirs(self.model_file_path)
        self.loss_func = LossFunc(self.hyperparameter['omega'])
        self.window_size = plan.window_size
        self.week_timeconv = TimeConv(conv_size=5, window_size=self.window_size, expand_feature_size=len(plan.expand_feature))
        self.month_timeconv = TimeConv(conv_size=20, window_size=self.window_size, expand_feature_size=len(plan.expand_feature))
        self.lstm = torch.nn.LSTM(input_size=len(plan.feature)+10*len(plan.expand_feature), hidden_size=self.hyperparameter['lstm_hidden_size'],
                 num_layers=self.hyperparameter['num_layers'], batch_first=True)
        self.gnn = GATConv(self.hyperparameter['lstm_hidden_size'],self.hyperparameter['gnn_hidden_size'])
        self.leaky_relu = torch.nn.LeakyReLU(self.hyperparameter['leaky_relu_alpha'])
        self.linear = torch.nn.Linear(in_features=self.hyperparameter['gnn_hidden_size']+self.hyperparameter['lstm_hidden_size'] , out_features=len(plan.label))
        self.mhsa=MHSA(num_heads=4,dim=self.hyperparameter['lstm_hidden_size'])

    def forward(self, data):
        week_x = self.week_timeconv(data.x)
        month_x = self.month_timeconv(data.x)
        x, edge_index =  torch.cat((data.x[:,-self.window_size:,:], week_x, month_x) ,axis=2),data.edge_index  
        time_x,hidden = self.lstm(x) 
#         time_x = self.mhsa(time_x)
        time_x = time_x[:,-1,:]   
        space_x = self.gnn(time_x, edge_index)  
        x = torch.cat((time_x,space_x),1)
        x = self.leaky_relu(x)
        x = self.linear(x)
        return x
    def loss_function(self,x,y,_close):
        return self.loss_func.dw_rank_loss(x,y,_close)  
    
# class Rank_STAN(torch.nn.Module):
#     def __init__(self, plan, suffix_name):
#         super(Rank_STAN, self).__init__()
#         self.model_name = 'Rank_STAN'
#         self.suffix_name = suffix_name
#         self.hyperparameter = plan.hyperparameter[self.model_name]
#         self.model_file_path = config.MINING_MODEL_FILE_PATH + self.model_name  + '/' + suffix_name
#         if not os.path.exists(self.model_file_path):
#             os.makedirs(self.model_file_path)
#         self.loss_func = LossFunc(self.hyperparameter['omega'])
#         self.window_size = plan.window_size
#         self.week_timeconv = TimeConv(conv_size=5, window_size=self.window_size, expand_feature_size=len(plan.expand_feature))
#         self.month_timeconv = TimeConv(conv_size=20, window_size=self.window_size, expand_feature_size=len(plan.expand_feature))
#         self.lstm = torch.nn.LSTM(input_size=len(plan.feature)+10*len(plan.expand_feature), hidden_size=self.hyperparameter['lstm_hidden_size'],
#                  num_layers=self.hyperparameter['num_layers'], batch_first=True)
#         self.mhsa=MHSA(num_heads=4,dim=self.hyperparameter['lstm_hidden_size'])
# #         self.cnn = torch.nn.Conv1d(in_channels=5,out_channels=self.hyperparameter['cnn_filters'],kernel_size=self.hyperparameter['cnn_kernel_size'])
#         self.cnn = torch.nn.Conv1d(in_channels=45,out_channels=15,kernel_size=self.hyperparameter['kernel_size'])

#         #
# #         self.gnn = GATConv(self.hyperparameter['lstm_hidden_size']+4,self.hyperparameter['gnn_hidden_size'])
#         self.gnn = GATConv(self.hyperparameter['lstm_hidden_size']+(16-self.hyperparameter['kernel_size']),self.hyperparameter['gnn_hidden_size'])

#         self.leaky_relu = torch.nn.LeakyReLU(self.hyperparameter['leaky_relu_alpha'])
#         self.linear = torch.nn.Linear(in_features=self.hyperparameter['gnn_hidden_size']+self.hyperparameter['lstm_hidden_size']+(16-self.hyperparameter['kernel_size']) , out_features=len(plan.label))
#     def forward(self, data):
#         week_x = self.week_timeconv(data.x)
#         month_x = self.month_timeconv(data.x)
#         x, edge_index =  torch.cat((data.x[:,-self.window_size:,:], week_x, month_x) ,axis=2),data.edge_index  
#         global_x,hidden = self.lstm(x)
#         global_x = self.mhsa(global_x)
        
# #         print(global_x.shape)
# #         [3413,5,64]
#         cnn_x = x.permute(0,2,1)
# #         print(cnn_x.shape)
#         part_x =self.cnn(cnn_x)
# #         print(part_x.shape)
# #         [3413,5,64]
        
#         time_x = torch.cat((global_x,part_x), 2)
# #         print(time_x.shape)
#         time_x = time_x[:,-1,:] 
        
# #         print(time_x.shape)
# #         gat_x = x[:,-1,:] 
# #         print(gat_x.shape)
# #         space_x = self.gnn(time_x, edge_index)

#         space_x = self.gnn(time_x, edge_index)
# #         print(space_x.shape)
#         x = torch.cat((time_x,space_x),1)
# #         print(x.shape)
        
#         x = self.leaky_relu(x)
#         x = self.linear(x)
#         return x
#     def loss_function(self,x,y,_close):
#         return self.loss_func.dw_rank_loss(x,y,_close)
    