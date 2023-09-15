import os
import sys
import pynvml
import torch
import traceback
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

sys.path.append('../../')
sys.path.append('../../../')

from src import config
from src.data_processing import data_loader
from src.data_processing import data_sync
# 模型预测服务,预测日期区间
def model_predict(model, dataset, plan, norm_config, draw_list):
    print('===============','model_predict() running...')
    print('=========','model_predict(): %s model, from %s to %s'%(model.model_name,plan.predict_begin_date,plan.predict_end_date))
    pynvml.nvmlInit()
#     free0 = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free/(1024*1024)
#     free1 = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(1)).free/(1024*1024)
#     if free0 > free1:
#         device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     else:
#         device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    
    print('device',device)
#     device = torch.device('cuda:0')
    model = model.to(device)
    res_df = {}
    res_set = {}
    stock_code_list = data_loader.get_stock_code_list(plan.index_code_list, plan.predict_begin_date)
#     print(stock_code_list)
    for stock in stock_code_list:
        res_df[stock] = []
    print(model.model_file_path + model.model_name+'.pth')
    #230907修改
    model.load_state_dict(torch.load(model.model_file_path + model.model_name+'.pth'))   # 加载模型参数
    print('model path',model.model_file_path + model.model_name+'.pth')
    result = torch.Tensor().to(device)
    # 预测过程
    model.eval()
#     loader = DataLoader(dataset,batch_size=1)
    for _data in dataset: 
        pred_y = model(_data.to(device))
        score = pred_y.add(_data.pre_close.mul(-1)).div(_data.pre_close)
        for i in range(0,len(pred_y)):
            _label = _data.label[i]
            res_df[_label[1]].append({'ts_code':_label[1],'trade_date':str(_label[0]),'close':_label[3],
                                  'pct_chg':_label[2],'pred':float(pred_y[i]),'score':float(score[i])})
    # 归一化处理
    for stock in stock_code_list:
        res_df[stock] = pd.DataFrame(res_df[stock])
        _label_real = np.array(res_df[stock].dropna()['pct_chg'])
        _label_norm = (_label_real - norm_config[stock]['min_value']) / (norm_config[stock]['max_value'] - norm_config[stock]['min_value'])
        _pred_norm =  np.array(res_df[stock].dropna()['pred'])
        result_evaluate={}
        result_evaluate['MSE'] = np.sum((_label_norm-_pred_norm)**2)/len(_pred_norm)
        result_evaluate['RMSE'] = np.sqrt(result_evaluate['MSE'])
        _pred_real = res_df[stock].loc[:,['pred']]*(norm_config[stock]['max_value'] - norm_config[stock]['min_value']) + norm_config[stock]['min_value']
        res_df[stock].loc[:,['pred']] = _pred_real
        acc_count = 0
        for i in range(0,len(_label_real)):
            if (_label_real[i])*(_pred_real.loc[i,'pred']) > 0 :
                acc_count += 1
        result_evaluate['ACC'] =  round(100*acc_count/len(_pred_real),2)
        result_evaluate['BALANCE'] = round(100*len(_pred_real[_pred_real['pred'] > 0])/len(_pred_real),2)
        res_set[stock] = result_evaluate   
        pred_result_path = config.FACTOR_DATA_PATH + model.suffix_name 
        if not os.path.exists(pred_result_path):
                os.makedirs(pred_result_path)
        _data_to_save = res_df[stock][['trade_date','close','pct_chg','score']]
        _data_to_save = _data_to_save.rename(columns={'score':model.model_name})
        try:
            data_sync.csv_factor_insert(_data_to_save ,pred_result_path+'/'+stock + '.csv','first',model.model_name)  
        except:
            print(stock+'--write error')
            traceback.print_exc()
    for stock in draw_list:     
        # 构建预测值集
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(16, 8))
        plt.title(stock + ':')
        plt.xlabel('date', fontsize=18)
        plt.ylabel(plan.label[0], fontsize=18)
        plt.plot(res_df[stock][['pct_chg','score']])
        plt.legend(['Label', 'Predictions'], loc='lower right')
#         print(stock,res_set[stock])
        plt.show()
        plt.close()
    return res_df, res_set, stock_code_list
