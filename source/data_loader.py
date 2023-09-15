import sys
import math
import time
import torch
import pandas as pd
import numpy as np
import datetime 
from torch_geometric.data import Data
import imp

sys.path.append('../../')
sys.path.append('../../../')
from src import config
imp.reload(config)

def get_mdoel_name(suffix_name,name):
    return suffix_name+'/'+name.name

#获取股票列表
def get_stock_code_list(index_code_list, predict_begin_date):
    print('获取股票列表')
    print('index_code_list:',index_code_list)
    try:
#         begin_year = str(predict_begin_date)[0:4]
#         begin_year = '2021'
        stock_code_list = []
        for index_code in index_code_list:
            stock_code_list += pd.read_csv(config.LIST_DATA_PATH  + 'ALL_2021_stock_list.csv', index_col=0)['ts_code'].tolist()
#             stock_code_list += pd.read_csv(config.LIST_DATA_PATH  +index_code+'_'+ begin_year + '_stock_list.csv', index_col=0)['ts_code'].tolist()
        print(len(stock_code_list))
        return list(set(stock_code_list))
    except:
        print('get_stock_code_list(): load list error!')
        return []
        
# 加载股票数据 
def load_stock_data(plan, begin_date, end_date, offset = 0):
    print('=========','load_stock_data() running...')
    #
    print(begin_date,end_date)
    benchmark_stock_df =  pd.read_csv(config.STOCK_DATA_PATH  + config.BENCHMARK_STOCK + '.csv', index_col=0,usecols=['trade_date']).sort_values(by=['trade_date'],ascending=True).reset_index()
    if benchmark_stock_df[benchmark_stock_df['trade_date'] >= int(begin_date)].empty or benchmark_stock_df[benchmark_stock_df['trade_date'] <= int(end_date)].empty:
        print('load_stock_data(): Dataset is empty in the date range!')
        return None,None,None
    begin_index = benchmark_stock_df[benchmark_stock_df['trade_date'] >= int(begin_date)].index[0]
    benchmark_stock_df = benchmark_stock_df.iloc[begin_index-offset:]
    benchmark_stock_df = benchmark_stock_df[benchmark_stock_df['trade_date'] <= int(end_date)].copy()
    date_list = benchmark_stock_df['trade_date'].tolist()
    df_col_list = ['trade_date','ts_code','orign_pct_chg','orign_close'] + plan.feature.copy()
    stock_data = {}
    norm_config = {}
    for stock in get_stock_code_list(plan.index_code_list, plan.predict_begin_date):
        _stock_data = pd.read_csv(config.STOCK_DATA_PATH  + stock + '.csv', index_col=0) 
        _stock_data.loc[_stock_data['pct_chg'] >= 10,['pct_chg']] = 10.0
        _stock_data.loc[_stock_data['pct_chg'] <= -10,['pct_chg']] = -10.0
        _stock_data.loc[:,['ts_code'] + plan.feature] = _stock_data.loc[:,['ts_code'] + plan.feature].fillna(method='ffill')
        _stock_data.loc[:,['ts_code'] + plan.feature] = _stock_data.loc[:,['ts_code'] + plan.feature].fillna(method='bfill')  
        _stock_data.insert(3,'orign_close', _stock_data.loc[:,['close']])
        _stock_data.insert(3,'orign_pct_chg', _stock_data.loc[:,['pct_chg']])
        norm_feature = plan.feature.copy()
        for _feature in plan.not_norm_feature:
            if _feature in norm_feature:
                norm_feature.remove(_feature)
        if plan.norm_method == 'overall':
            _norm_data = _stock_data.loc[(_stock_data['trade_date']>=int(plan.train_begin_date))&(_stock_data['trade_date']<=int(plan.predict_end_date)),norm_feature]
        else:
            _norm_data = _stock_data.loc[(_stock_data['trade_date']>=int(plan.train_begin_date))&(_stock_data['trade_date']<=int(plan.train_end_date)),norm_feature]
        max_value = _norm_data.max()
        min_value = _norm_data.min()
        norm_config[stock] = {'max_value':float(_stock_data.loc[:,plan.label].max()),'min_value': float(_stock_data.loc[:,plan.label].min())}
        _stock_data.loc[:,norm_feature] = (_stock_data.loc[:,norm_feature]-min_value)/(max_value-min_value)
        _stock_data = pd.merge(benchmark_stock_df,_stock_data,on='trade_date',how='left')
        _stock_data = _stock_data.fillna(method='ffill')
        _stock_data = _stock_data.fillna(method='bfill')
        #添加后面
#         _stock_data = _stock_data.fillna(0.1) 
#         print(_stock_data.isnull().any()) #用来判断某列是否有缺失值


        zero_value_index = _stock_data.loc[_stock_data['close'] <= 0,:].index 
#         print(zero_value_index)
        if not zero_value_index.empty:
            _stock_data.loc[_stock_data['close'] == 0,['close']]  =  float(_stock_data[_stock_data['close'] > 0].sort_values(by=['close']).iloc[0]['close'])

        norm_config[stock]['norm_mean_value'] = _stock_data.loc[:,plan.label].mean()[plan.label[0]]
        stock_data[stock] =  _stock_data.loc[:,df_col_list]
    for stock in stock_data: 
        try:
            stock_data[stock][list(stock_data[stock].loc[:,stock_data[stock].dtypes=='float64'].columns)] = stock_data[stock][ list(stock_data[stock].loc[:,stock_data[stock].dtypes=='float64'].columns)].astype('float32')
            stock_data[stock]['trade_date'] = stock_data[stock]['trade_date'].astype('int32')
            stock_data[stock]['close_up_status'] = stock_data[stock]['close_up_status'].astype('int16')
            stock_data[stock]['close_down_status'] = stock_data[stock]['close_down_status'].astype('int16')
        except:
            pass
    print('=========','load_stock_data() finished, range %s - %s, Norm method %s'%(str(date_list[0]),end_date,plan.norm_method))
    return date_list,stock_data,norm_config

# 获取图关系
def get_graph_edge(stock_code_list, relation_type):
    print('=========','get_graph_edge(): relation type is <%s>'%(relation_type))
    if relation_type in ['industry','theme','cluster']:
        if relation_type == 'industry':
            file_path = config.RELATION_DATA_PATH +'industry.csv'
        if relation_type == 'theme':
            file_path = config.RELATION_DATA_PATH +'theme.csv'
        if relation_type == 'cluster':
            file_path = config.RELATION_DATA_PATH +'cluster.csv'
        
        relation_df = pd.read_csv(file_path, index_col = 0)
#         print(relation_df)
#         print(stock_code_list)
#         print(len(stock_code_list))
        res_set = []
        size = len(stock_code_list)
        class_list = []
        for i in range(0,size):
            class_list.append(str(relation_df[relation_df['ts_code'] == stock_code_list[i]].iloc[0]['class']).split(','))
#             print(str(relation_df[relation_df['ts_code'] == stock_code_list[i]].iloc[0]['class']).split(','))
        for i in range(0,size):
            class_i = class_list[i]
            for j in range(i + 1,size):
                class_j = class_list[j]
                for c in class_i:
                    if c in class_j:
                        res_set.append([i,j])
                        res_set.append([j,i])
                        break
        return torch.tensor(res_set,dtype=torch.long)
    elif relation_type in ['fund','corr']:
        if relation_type == 'fund':
            file_path = config.RELATION_DATA_PATH +'fund_relation/'
        _res_set = {}
        for date in config.FUND_DATE_LIST:
            relation_df = pd.read_csv(file_path + date + '.csv', index_col = 0)
            res_set = []
            size = len(stock_code_list)
            class_list = []
            for i in range(0,size):
                class_list.append(str(relation_df[relation_df['ts_code'] == stock_code_list[i]].iloc[0]['class']).split(','))
            for i in range(0,size):
                class_i = class_list[i]
                for j in range(i + 1,size):
                    class_j = class_list[j]
                    for c in class_i:
                        if c in class_j:
                            res_set.append([i,j])
                            res_set.append([j,i])
                            break
            _res_set[date] = torch.tensor(res_set,dtype=torch.long)
        return _res_set
    else:
        return torch.tensor([],dtype=torch.long)
            
# 加载训练集  
def load_train_data(plan):
    print('===============','load_train_data() running...')
    window_size = plan.window_size + plan.expand_window_size
    date_list,stock_data,norm_config = load_stock_data(plan,plan.train_begin_date, plan.train_end_date,window_size)
    dataset = []
    graph_edge = get_graph_edge(get_stock_code_list(plan.index_code_list, plan.predict_begin_date), plan.graph_edge_type)
    feature_list = plan.feature.copy()
    for feature in plan.expand_feature.copy():
        if feature in feature_list:
            feature_list.remove(feature)
        else:
            plan.expand_feature.remove(feature)
    feature_list = plan.expand_feature + feature_list
    for i in range(0,len(date_list)-window_size):
        if i % int(len(date_list)/10) == 0:
            print( '=========','%.2f'%(100*i/len(date_list)),'%')
        x = []
        y = []
        pre_close = []
        label = []  
        for data in stock_data.values():
            _x = data.loc[i:i+window_size-1,feature_list].values
            x.append(_x)
            _y = data.loc[i+window_size,plan.label].values
            y.append(_y)
            _pre_close = data.loc[i+window_size-1,['close']].values
            pre_close.append(_pre_close)    
            _label = data.loc[i+window_size,['trade_date','ts_code','orign_pct_chg','orign_close']].values
            label.append(_label)
        if plan.graph_edge_type in ['fund','corr']:
            _date_list = ['20091231'] + list(graph_edge.keys()) + ['20211231']
            _date = str(_label[0])
            for k in range(0,len(_date_list) - 1):
                if _date >_date_list[k] and _date <= _date_list[k + 1]:
                    edge_index = graph_edge[_date_list[k + 1]]
                    break
        else:
            edge_index = graph_edge
        dataset.append(Data(x=torch.Tensor(x), y=torch.Tensor(y),edge_index=edge_index.t().contiguous(),label = label, pre_close = torch.Tensor(pre_close))) 
    print( '=========','%.2f'%(100),'%')
    print('===============','load_train_data(): finished, range %s - %s , %d trade days'%( plan.train_begin_date, plan.train_end_date, len(date_list)))
    return dataset,norm_config


# 加载测试集  
def load_predict_data(plan):
    print('===============','load_predict_data() running...')
    window_size = plan.window_size + plan.expand_window_size
    date_list,stock_data,norm_config = load_stock_data(plan,plan.predict_begin_date, plan.predict_end_date,window_size)
    dataset = []
    graph_edge = get_graph_edge(get_stock_code_list(plan.index_code_list, plan.predict_begin_date), plan.graph_edge_type)
    feature_list = plan.feature.copy()
    for feature in plan.expand_feature.copy():
        if feature in feature_list:
            feature_list.remove(feature)
        else:
            plan.expand_feature.remove(feature)
    feature_list = plan.expand_feature + feature_list
    _len = len(date_list)-window_size+1
    for i in range(0,len(date_list)-window_size+1):
        if i % int(_len/10) == 0:
            print('=========','%.2f'%(100*i/_len),'%')
        x = []
        y = []
        pre_close = []
        label = []         
        for data in stock_data.values():
            data['trade_date'] = data['trade_date'].astype('str')
            _x = data.loc[i:i+window_size-1,feature_list].values
            x.append(_x)
            _pre_close = data.loc[i+window_size-1,['close']].values
            pre_close.append(_pre_close)    
            if i+window_size != len(date_list):
                _y = data.loc[i+window_size,plan.label].values
                y.append(_y)
                _label = data.loc[i+window_size,['trade_date','ts_code','orign_pct_chg','orign_close']].values
                label.append(_label)
            else:
                _label = data.loc[i+window_size - 1,['trade_date','ts_code','orign_pct_chg','orign_close']].values
                _label[0] = 'next_trade_day'
                _label[2] = 0
                _label[3] = 0
                label.append(_label)
        if plan.graph_edge_type in ['fund','corr']:
            _date_list = ['20091231'] + list(graph_edge.keys()) + ['20211231']
            _date = str(_label[0])
            for k in range(0,len(_date_list) - 1):
                if _date >_date_list[k] and _date <= _date_list[k + 1]:
                    edge_index = graph_edge[_date_list[k + 1]]
                    break
        else:
            edge_index = graph_edge
        dataset.append( Data(x=torch.Tensor(x), y=torch.Tensor(y),edge_index=edge_index.t().contiguous(),label = label, pre_close = torch.Tensor(pre_close)))   
    print('=========','%.2f'%(100),'%')
    print('===============','load_predict_data(): finished, range %s - %s , %d trade days'%(plan.predict_begin_date, plan.predict_end_date, len(date_list)))
    return dataset,norm_config

def get_ranked_stocks(res_df):
    stock_code_list = list(res_df.keys())
    stock_data_df = pd.DataFrame({'trade_date':[]})
    pct_chg_data_df = stock_data_df.copy()
    pred_data_df = stock_data_df.copy()
    for stock in stock_code_list:
        _pred_data =  res_df[stock]
        data = _pred_data.loc[_pred_data['trade_date']== 'next_trade_day']
        if data['score'].values>0:
            print(data)
#         _pred_data = _pred_data[_pred_data['trade_date'] >= _date]
            pred_data_df = pd.merge(pred_data_df,_pred_data[['trade_date','score']],on='trade_date',how='outer')
            pred_data_df = pred_data_df.rename(columns={'score':stock}) 
        
    selected_stocks = {}
#     pred_data_df=pred_data_df[pred_data_df['score']>0]
    print(pred_data_df)
    for i,row in pred_data_df.iterrows():
        _sort_list = row.drop(['trade_date']).sort_values(ascending=False)
        selected_stocks[row['trade_date']] = _sort_list.index.tolist()
    return selected_stocks

def get_model_name(suffix_name,name):
    tran={'018D':'014D','015D':'001M','017R':'001R','008D':'010D','015R':'012D','017D':'013D','007D':'002R','013D':'006D','014D':'005D','005D':'005R','011D':'006R','016R':'003R','004D':'002M','006D':'003M','009D':'015D','006R':'011D','004R':'017D','010D':'017R','012D':'008D','005R':'004D','001R':'018D','003R':'009D','001M':'007D','003M':'016R','002R':'015R','002M':'004R'}
    inds_tran = {'018D':'012D','017D':'006D','016D':'002R','015D':'003R','014D':'018D','013D':'004R'}
    fun_tran = {'018D':'011D','017D':'018R','016D':'009D','015D':'008D','014D':'013D','013D':'012D'}
    if suffix_name == 'DAILY/' or suffix_name == 'DAILY_SAVE/':
        try:
            return 'DeepAlpha'+tran[name[-4:]]
        except:
            return name
    elif suffix_name == 'DAILY_INDUSTRY/':
        try:
            return 'DeepAlpha'+inds_tran[name[-4:]]
        except:
            return name
    elif suffix_name == 'DAILY_FUND/':
        try:
            return 'DeepAlpha'+fun_tran[name[-4:]]
        except:
            return name
    else:
        return name

def get_ranked_returns(ranked_stocks, res_df, predict_date, top_k):
    returns = 0
    count = 0
    for stock in ranked_stocks[predict_date]:
        try:
            _item = res_df[stock][ res_df[stock]['trade_date'] == predict_date].iloc[0]
            returns += _item['pct_chg']/100
            count += 1
        except:
            pass
        if count >= top_k:
            break
    return round((returns/count)*100,2)

def write_predict_result(model,plan,ranked_stocks,res_df,mode):
    data_list = list(ranked_stocks.keys())
    db = config.DbConnection_aliyun()
    _sql = "SELECT table_name FROM information_schema.TABLES WHERE table_name ='"+"yc_"+model.model_name+"';"
    if len(db.execute('stock_system',_sql).fetchall())==0:
        _sql = "CREATE TABLE `"+"yc_"+model.model_name+"` (`predict_date` varchar(8) DEFAULT NULL,`model_name` varchar(255) DEFAULT NULL,\
              `suffix_name` varchar(255) DEFAULT NULL,`ranked_stocks` text,`top_1_return` float DEFAULT NULL,`top_10_return` float DEFAULT NULL,\
                `top_50_return` float DEFAULT NULL,`top_100_return` float DEFAULT NULL,`top_300_return` float DEFAULT NULL,\
              `predict_time` varchar(20) DEFAULT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8;"  
        db.execute('stock_system',_sql)
        import time
        time.sleep(5)
    if mode == 'period' or  mode == 'peroid':
        for predict_date in data_list: 
            if predict_date == 'next_trade_day':
                continue
            try:                 
                # 查询数据库
                _sql = "SELECT * FROM "+"yc_"+model.model_name+" WHERE predict_date='"+predict_date+"';"
                result = db.execute('stock_system',_sql).fetchall()
                if len(result)>0:
                    _sql = "DELETE FROM "+"yc_"+model.model_name+" WHERE predict_date='"+predict_date+"';"
                    db.execute('stock_system',_sql)
                # 写入数据库
                db = config.DbConnection_aliyun()
                item = {}
                item['predict_date'] = predict_date
                item['model_name'] =model.model_name
                item['suffix_name'] =model.suffix_name
                item['ranked_stocks'] = str(ranked_stocks[predict_date]).replace('[','').replace(']','').replace("'",'')                   
                item['top_1_return'] = get_ranked_returns(ranked_stocks,res_df,predict_date,1)
                item['top_10_return'] = get_ranked_returns(ranked_stocks,res_df,predict_date,10)
                item['top_50_return'] = get_ranked_returns(ranked_stocks,res_df,predict_date,50)
                item['top_100_return'] = get_ranked_returns(ranked_stocks,res_df,predict_date,100)
                item['top_300_return'] = get_ranked_returns(ranked_stocks,res_df,predict_date,300)
                item['predict_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                sql = "INSERT INTO "+"yc_"+model.model_name+" VALUES ('%(predict_date)s', '%(model_name)s',\
                     '%(suffix_name)s',\
                     '%(ranked_stocks)s','%(top_1_return)s', '%(top_10_return)s', '%(top_50_return)s', '%(top_100_return)s', '%(top_300_return)s',\
                      '%(predict_time)s')"%item
                db.execute('stock_system',sql)
                print('*'*1+predict_date+'--数据库写入'+model.model_name+'预测结果成功'+'*'*1)
            except:
                print('*'*5+predict_date+'--数据库：写入'+model.model_name+'预测结果失败'+'*'*5)
                traceback.print_exc()
                return False
    if mode == 'day':
        today_date = pd.to_datetime(plan.predict_end_date)
        if today_date.strftime("%w") == '5':
            tomorrow_date = (today_date + datetime.timedelta(days=3)).strftime('%Y%m%d')
        else:
            tomorrow_date = (today_date + datetime.timedelta(days=1)).strftime('%Y%m%d')
        predict_date = tomorrow_date
        try:                 
            # 查询数据库
            model_name="rank_STAN"
#             _sql = "SELECT * FROM "+"yc_"+model.model_name+" WHERE predict_date='"+predict_date+"';" 改之前
            _sql = "SELECT * FROM "+"yc_"+model.model_name+" WHERE predict_date='"+predict_date+"';"

            result = db.execute('stock_system',_sql).fetchall()
            if len(result)>0:
#                 _sql = "DELETE FROM "+"yc_"+model.model_name+" WHERE predict_date='"+predict_date+"';"
                _sql = "DELETE FROM "+"yc_"+model.model_name+" WHERE predict_date='"+predict_date+"';"

                db.execute('stock_system',_sql)
            # 写入数据库
            db = config.DbConnection_aliyun()
            item = {}
            item['predict_date'] = predict_date
#             item['model_name'] =model.model_name
            item['model_name'] =model.model_name

            item['suffix_name'] =model.suffix_name
            item['ranked_stocks'] = str(ranked_stocks['next_trade_day']).replace('[','').replace(']','').replace("'",'')                   
            item['predict_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#             sql = "INSERT INTO "+"yc_"+model.model_name+" VALUES ('%(predict_date)s', '%(model_name)s',\
#                  '%(suffix_name)s',\
#                  '%(ranked_stocks)s',NULL, NULL, NULL, NULL, NULL,\
#                   '%(predict_time)s')"%item
            sql = "INSERT INTO "+"yc_"+model.model_name+" VALUES ('%(predict_date)s', '%(model_name)s',\
                 '%(suffix_name)s',\
                 '%(ranked_stocks)s',NULL, NULL, NULL, NULL, NULL,\
                  '%(predict_time)s')"%item
            db.execute('stock_system',sql)
#             print(sql)
#             print('*'*1+predict_date+'--数据库写入'+model.model_name+'预测结果成功'+'*'*1)
            print('*'*1+predict_date+'--数据库写入'+model_name+'预测结果成功'+'*'*1)

        except:
            print('*'*5+predict_date+'--数据库：写入'+model.model_name+'预测结果失败'+'*'*5)
            traceback.print_exc()
            return False
        try:
#             df = db.read_table("yc_"+model.model_name,'stock_system')
            df = db.read_table(model_name,'stock_system')

            df = df[df['top_1_return'].isnull()]
            for i,row in df.iterrows():  
                try:
                    if row['predict_date'] >= plan.predict_end_date:
                        continue
                    _data = {}
                    _data['top_1_return'] = get_ranked_returns(ranked_stocks,res_df,row['predict_date'],1)
                    _data['top_10_return'] = get_ranked_returns(ranked_stocks,res_df,row['predict_date'],10)
                    _data['top_50_return'] = get_ranked_returns(ranked_stocks,res_df,row['predict_date'],50)
                    _data['top_100_return'] = get_ranked_returns(ranked_stocks,res_df,row['predict_date'],100)
                    _data['top_300_return'] = get_ranked_returns(ranked_stocks,res_df,row['predict_date'],300)
#                     sql = "UPDATE "+"yc_"+model.model_name+" SET top_1_return='%(top_1_return)s', top_10_return='%(top_10_return)s', top_50_return='%(top_50_return)s', \
#                         top_100_return='%(top_100_return)s',top_300_return='%(top_300_return)s'" %_data
                    sql = "UPDATE "+model_name+" SET top_1_return='%(top_1_return)s', top_10_return='%(top_10_return)s', top_50_return='%(top_50_return)s', \
                        top_100_return='%(top_100_return)s',top_300_return='%(top_300_return)s'" %_data
                    sql +=  "WHERE predict_date = '"+row['predict_date']+ "'"
                    db.execute('stock_system',sql)
                except:
                    pass
            print('*'*5+predict_date+'--数据库：更新'+model.model_name+'收益率成功'+'*'*5)
        except:
            print('*'*5+predict_date+'--数据库：更新'+model.model_name+'收益率失败'+'*'*5)
            return False
            traceback.print_exc()
    return True

            