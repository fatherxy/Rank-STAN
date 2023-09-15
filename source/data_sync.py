import sys
import os
import ta
import time
import datetime
import traceback
import numpy as np
import tushare as ts
import efinance as ef
import pandas as pd

sys.path.append('../../')
sys.path.append('../../../')
from src import config

pro = ts.pro_api(config.TUSHARE_TOKEN)
ts.set_token(config.TUSHARE_TOKEN)
cn_list_data = pd.read_csv(config.LIST_DATA_PATH + 'cn_stock_list.csv', parse_dates=True, index_col=0)
cn_list = cn_list_data['ts_code'].tolist()
index_code_list = config.BENCHMARK_INDEX


def csv_head_insert(df, path, keep, drop_same = True):  
    import warnings
    warnings.filterwarnings("ignore")
    if os.path.exists(path):
        try:
            if not len(df) > 0:
                return df
            original_df = pd.read_csv(path,index_col=0,error_bad_lines=False)
            merge_df = df.append(original_df)
            merge_df = merge_df.astype({"trade_date": str})
            if drop_same:
                merge_df.drop_duplicates(subset=['trade_date'],keep=keep,inplace=True)
            merge_df.sort_values(by=['trade_date'],ascending=False,inplace=True)
            merge_df.reset_index(inplace=True,drop = True) 
            merge_df.to_csv(path)
            return merge_df
        except:
            df.sort_values(by=['trade_date'],ascending=False,inplace=True)
            df.reset_index(inplace=True,drop = True) 
            df.to_csv(path)
            return df
    else:
        df.sort_values(by=['trade_date'],ascending=False,inplace=True)
        df.reset_index(inplace=True,drop = True) 
        df.to_csv(path)
        return df
    
def csv_factor_insert(df, path, keep, col_name): 
    df = df[['trade_date',col_name]]
    if os.path.exists(path):
        original_df = pd.read_csv(path,index_col=0,error_bad_lines=False)
        original_df = original_df.astype({"trade_date": str})
        df = df.astype({"trade_date": str})
        if col_name in original_df.columns:
            _original_df = original_df[['trade_date',col_name]]
            new_df = pd.concat([df,_original_df])
            new_df.drop_duplicates(subset=['trade_date'],keep='first',inplace=True)
            original_df = original_df.drop([col_name],axis=1)
            merge_df = pd.merge(new_df,original_df,on=['trade_date'],how='outer')
        else:
            merge_df = pd.merge(original_df,df,on=['trade_date'],how='outer')
        merge_df.sort_values(by=['trade_date'],ascending=False,inplace=True)
        merge_df.reset_index(inplace=True,drop = True) 
        merge_df.to_csv(path)
        return merge_df
    else:
        df.sort_values(by=['trade_date'],ascending=False,inplace=True)
        df.reset_index(inplace=True,drop = True) 
        df.to_csv(path)
        return df   
# def csv_factor_insert(df, path, keep, col_name): 
#     df = df.loc[:,['trade_date',col_name]]
#     if os.path.exists(path):
#         original_df = pd.read_csv(path,index_col=0)
#         original_df = original_df.astype({"trade_date": str})
#         df = df.astype({"trade_date": str})
#         if col_name in original_df.columns:
#             df = df.rename(columns={col_name:'the_new_col'})
#             merge_df = pd.merge(original_df,df,on=['trade_date'],how='outer')
#             _bool = ~merge_df['the_new_col'].isnull()
#             merge_df.loc[_bool,col_name] = merge_df.loc[_bool,'the_new_col']
#             merge_df = merge_df.drop(['the_new_col'],axis=1)
#         else:
#             merge_df = pd.merge(original_df,df,on=['trade_date'],how='outer')
#         merge_df.sort_values(by=['trade_date'],ascending=False,inplace=True)
#         merge_df.reset_index(inplace=True,drop = True) 
#         merge_df.to_csv(path)
#         return merge_df
#     else:
#         df.sort_values(by=['trade_date'],ascending=False,inplace=True)
#         df.reset_index(inplace=True,drop = True) 
#         df.to_csv(path)
#         return df
    
def get_today_date():
    return datetime.datetime.today().strftime('%Y%m%d')

def download_stock_data(start_date = config.START_DATE, end_date = get_today_date(), is_update = False):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'download_stock_data() start!')
    stock_index_data = {}
    try:
        for stock_index_code in ['399006.SZ','399005.SZ','000688.SH','399300.SZ']:
            temp_data = pro.index_daily(ts_code=stock_index_code, start_date=start_date, end_date=end_date)[['trade_date','pct_chg']]
            print(temp_data,33333)
            temp_data.rename(columns={'pct_chg':'market_chg'},inplace=True)
            stock_index_data[stock_index_code] = temp_data
    except:
        traceback.print_exc()
        print('========== download error')  
        return 0
    progress = 0
    error_count = 0
    for stock_code in cn_list:
        progress += 1
        try:
            # 获取量价数据
            use_col_01 = ['trade_date','ts_code','open','high','low',
                          'close','pre_close','pct_chg','vol','amount','turnover_rate',
                          'volume_ratio','ma5','ma20']
            data_01 = ts.pro_bar(ts_code=stock_code,start_date= start_date,end_date= end_date,asset='E',
                                                  adj='qfq',freq='D',ma=[5, 20],factors=['tor', 'vr'],adjfactor=True)
            data_01 = data_01[use_col_01].sort_values(by='trade_date')
            data_01.loc[:,['vol_m5']] = data_01['vol'].rolling(5).mean()
            data_01.loc[:,['vol_m20']] = data_01['vol'].rolling(20).mean()
            data_01.loc[:,['pct_chg_s5']] = data_01['pct_chg'].rolling(5).sum()
            data_01.loc[:,['pct_chg_s20']] = data_01['pct_chg'].rolling(20).sum()
            data_01.loc[:,['turnover_rate_m5']] = data_01['turnover_rate'].rolling(5).mean()
            data_01.loc[:,['turnover_rate_m20']] = data_01['turnover_rate'].rolling(20).mean()
            data_01.loc[:,['close_up_status']] = 0
            data_01.loc[:,['close_down_status']] = 0
            data_01.loc[data_01['pct_chg'] >= 9.8,['close_up_status']] = 1
            data_01.loc[data_01['pct_chg'] <= -9.8,['close_down_status']] = 1
            # 获取财务数据
            use_col_02 = ['trade_date','pe_ttm','pb','ps_ttm','dv_ttm','total_mv']
            data_02 = pro.daily_basic(ts_code=stock_code,start_date= start_date,end_date= end_date, adjfactor=True)
            data_02 = data_02[use_col_02].sort_values(by='trade_date')
            # 获取资金流数据
            use_col_03 = ['trade_date','buy_sm_vol','buy_md_vol','buy_lg_vol','buy_elg_vol',
                          'sell_sm_vol','sell_md_vol','sell_lg_vol','sell_elg_vol']
            data_03 = pro.moneyflow(ts_code=stock_code,start_date= start_date,end_date= end_date, adjfactor=True)
            data_03 = data_03[use_col_03]
            data_03.loc[:,['buy_total_vol']] = data_03['buy_sm_vol'] +  data_03['buy_md_vol']  +  data_03['buy_lg_vol'] +  data_03['buy_elg_vol'] 
            data_03.loc[:,['buy_sm_vol_ratio']] =  data_03['buy_sm_vol'] /  data_03['buy_total_vol']
            data_03.loc[:,['buy_md_vol_ratio']] =  data_03['buy_md_vol'] /  data_03['buy_total_vol']
            data_03.loc[:,['buy_lg_vol_ratio']] =  data_03['buy_lg_vol'] /  data_03['buy_total_vol']
            data_03.loc[:,['buy_elg_vol_ratio']] =  data_03['buy_elg_vol'] /  data_03['buy_total_vol']
            data_03.loc[:,['sell_total_vol']] = data_03['sell_sm_vol'] +  data_03['sell_md_vol']  +  data_03['sell_lg_vol'] +  data_03['sell_elg_vol'] 
            data_03.loc[:,['sell_sm_vol_ratio']] =  data_03['sell_sm_vol'] /  data_03['sell_total_vol']
            data_03.loc[:,['sell_md_vol_ratio']] =  data_03['sell_md_vol'] /  data_03['sell_total_vol']
            data_03.loc[:,['sell_lg_vol_ratio']] =  data_03['sell_lg_vol'] /  data_03['sell_total_vol']
            data_03.loc[:,['sell_elg_vol_ratio']] =  data_03['sell_elg_vol'] /  data_03['sell_total_vol']
            data_03 = data_03[['trade_date','buy_sm_vol_ratio','buy_md_vol_ratio','buy_lg_vol_ratio','buy_elg_vol_ratio','sell_sm_vol_ratio','sell_md_vol_ratio','sell_lg_vol_ratio','sell_elg_vol_ratio']]
            data_all = data_01.merge(data_02,on='trade_date',how='left').merge(data_03,on='trade_date',how='left')
            
            # 计算技术指标
            df = data_all
            data_all['MACD'] = ta.trend.MACD(df['close']).macd()
            data_all['EMA'] = ta.trend.EMAIndicator(df['close']).ema_indicator()
            data_all['ADX'] = ta.trend.ADXIndicator(df['high'],df['low'],df['close'],fillna=True).adx()
            data_all['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'],df['vol'],fillna=True).on_balance_volume()
            data_all['ROC'] = ta.momentum.ROCIndicator(df['close']).roc()
            data_all['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
            data_all['BB_high_sign'] = ta.volatility.BollingerBands(df['close']).bollinger_hband_indicator()
            data_all['BB_low_sign'] = ta.volatility.BollingerBands(df['close']).bollinger_lband_indicator()
            data_all['BB_width'] = ta.volatility.BollingerBands(df['close']).bollinger_wband() 
            # 获取股指数据
            if cn_list_data.loc[cn_list_data['ts_code'] == stock_code,'market'].iloc[0] == '创业板':
                data_all = pd.merge(data_all,stock_index_data['399006.SZ'],on=['trade_date'],how='left')
            elif cn_list_data.loc[cn_list_data['ts_code'] == stock_code,'market'].iloc[0] == '中小板':
                data_all = pd.merge(data_all,stock_index_data['399005.SZ'],on=['trade_date'],how='left')
            elif cn_list_data.loc[cn_list_data['ts_code'] == stock_code,'market'].iloc[0] == '科创板':
                data_all = pd.merge(data_all,stock_index_data['000688.SH'],on=['trade_date'],how='left')
            else:
                data_all = pd.merge(data_all,stock_index_data['399300.SZ'],on=['trade_date'],how='left')
            data_all = data_all.sort_values(by='trade_date',ascending=False)
            if is_update :
                data_all.dropna(subset = ['MACD'],inplace = True)
            if not os.path.exists(config.STOCK_DATA_PATH):
                os.makedirs(config.STOCK_DATA_PATH)
            csv_head_insert( data_all.reset_index(drop=True),config.STOCK_DATA_PATH + stock_code + '.csv','last')
            print('========== download %s succeeded %d/%d'%(stock_code,progress,len(cn_list)))    
            error_count = 0
        except:
            error_count += 1
            traceback.print_exc()
            print('========== download %s failed %d/%d'%(stock_code,progress,len(cn_list)))   
            if progress < 10 or error_count >= 8:
                print('========== download error')   
                return 0
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'download_stock_data() stopped!')
    
def get_stock_code_list(index_code_list, train_begin_date, test_begin_date, end_date, empty_rate = 0.15):
    stock_code_list = []
    for index_code in index_code_list:
        if index_code == 'ALL':
            stock_code_list = pd.read_csv(config.LIST_DATA_PATH + 'cn_stock_list.csv', index_col=0)['ts_code'].tolist()
        else:
            index_df = pro.index_weight(index_code=index_code, start_date= test_begin_date, end_date=end_date).drop_duplicates(subset=['con_code'])
            stock_code_list += (index_df['con_code'].tolist())
    history_path = config.STOCK_DATA_PATH 
    standard_stock_df =  pd.read_csv(history_path +config.BENCHMARK_STOCK +'.csv', index_col=0,usecols=['trade_date']).sort_values(by=['trade_date'],ascending=True).reset_index()
    if standard_stock_df[standard_stock_df['trade_date'] >= int(train_begin_date)].empty or standard_stock_df[standard_stock_df['trade_date'] <= int(end_date)].empty:
        print('get_stock_code_list(): List is empty in the date range!')
        return None
    else:
        standard_stock_df = standard_stock_df[(standard_stock_df['trade_date'] >= int(train_begin_date))&(standard_stock_df['trade_date'] <= int(end_date))]
    stock_code_list = list(set(stock_code_list))
    orgin_list_length = len(stock_code_list)
    for stock in stock_code_list.copy():
        try:
            _stock_data =  pd.read_csv(history_path + stock + '.csv', index_col=0) 
            _stock_data = pd.merge(standard_stock_df,_stock_data,on='trade_date',how='left')
            _stock_data_last = _stock_data.iloc[-50:]
            last_data_empty_rate = len(_stock_data_last[_stock_data_last['ts_code'].isnull()])/len(_stock_data_last)
            data_empty_rate = len(_stock_data[_stock_data['ts_code'].isnull()])/len(_stock_data)
            if data_empty_rate > empty_rate or len(_stock_data) < 10 or last_data_empty_rate > 0.3:
                stock_code_list.pop(stock_code_list.index(stock))
        except:
            stock_code_list.pop(stock_code_list.index(stock))
    list_length = len(stock_code_list)
    print('get_stock_code_list(): return %d stocks, %d stocks were removed'%(list_length,orgin_list_length-list_length))
    return stock_code_list

def download_stock_list(index_code_list):
    for index_code in index_code_list:
        begin_year = 2014
        end_year = 2023
        train_length = 2
        res_set_set = {}
        for i in range(0,end_year - begin_year):  
            train_begin_date=str(begin_year - train_length  + i)+'0101'
            train_end_date=str(begin_year + i)+'0101'
            predict_begin_date = train_end_date
            predict_end_date=str(begin_year + i + 1)+'0101'
            print(index_code+'_'+str(begin_year+i) + '_stock_list.csv')
            stock_code_list = get_stock_code_list([index_code],train_begin_date, predict_begin_date, predict_end_date,0.15)
            file_path = config.LIST_DATA_PATH + index_code+'_'+str(begin_year+i) + '_stock_list.csv'
            if len(stock_code_list) > 0:
                pd.DataFrame({'ts_code':stock_code_list}).to_csv(file_path)
                 
def update_industry_relation(stock_code_list):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'开始更新股票行业关系')
    file_path = config.RELATION_DATA_PATH +'industry.csv'
    res_set = {}
    for code in stock_code_list:
        res_set[code] = []
    concept_df =  pro.ths_index()
    concept_df = concept_df[concept_df['exchange'] =='A']
    industry_df = concept_df[concept_df['type'] =='I'].reset_index(drop=True)
    for i,row in industry_df.iterrows():
        if i % 199 == 0:
            time.sleep(61)
        _industry_df = pro.ths_member(ts_code=row['ts_code'])
        for _i,_row in _industry_df.iterrows():
            if _row['code'] in stock_code_list:
                res_set[_row['code']].append(row['name'])
    for code in stock_code_list:
        res_set[code] = str(res_set[code]).replace('[','').replace(']','').replace('\'','').replace(' ','')
    res_df = pd.DataFrame({'ts_code':res_set.keys(),'class':res_set.values()})
    if len(res_df) > 0:
        res_df.to_csv(file_path)
    print('update_industry_relation() success! 更新股票行业关系完成!')
    return res_df

def update_theme_relation(stock_code_list):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'开始更新股票题材概念关系')
    file_path = config.RELATION_DATA_PATH +'theme.csv'
    res_set = {}
    for code in stock_code_list:
        res_set[code] = []
    concept_df =  pro.ths_index()
    concept_df = concept_df[concept_df['exchange'] =='A']
    industry_df = concept_df[(concept_df['type'] =='N') & (concept_df['count'] < 200)].reset_index(drop=True)
    for i,row in industry_df.iterrows():
        if (i+1) % 199 == 0:
            print('========== %d/%d'%(i,len(industry_df)))  
            time.sleep(60)
        _industry_df = pro.ths_member(ts_code=row['ts_code'])
        for _i,_row in _industry_df.iterrows():
            if _row['code'] in stock_code_list:
                res_set[_row['code']].append(row['name'])
    for code in stock_code_list:
        res_set[code] = str(res_set[code]).replace('[','').replace(']','').replace('\'','').replace(' ','')
    res_df = pd.DataFrame({'ts_code':res_set.keys(),'class':res_set.values()})
    if len(res_df) > 0:
        res_df.to_csv(file_path)
    print('update_industry_relation() success! 更新股票题材概念关系完成!')
    return res_df
        

    
def update_fund_relation(stock_code_list): 
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'开始更新基金持仓关系')
    df = pro.fund_basic(market='O')
    _fund_type = ['股票型','混合型']
    _invest_type = ['灵活配置型', '混合型', '股票型', '平衡型', '价值型', '普通股票型', '价值增长型', '稳健增长型','成长型']
    df = df[(df.fund_type.isin(_fund_type)) & (df.invest_type.isin(_invest_type)) & (df.status.isin(['I','L']))]
    df = df.reset_index(drop=True)
    fund_df = {}
    for i,row in df.iterrows():
        if (i+1) % 200 == 0:
            print('========== %d/%d'%(i,len(df)))  
            time.sleep(60)
        fund = pro.fund_portfolio(ts_code=row['ts_code'])
        if not fund.empty:
            fund_df[row['ts_code']] = fund
    fund_code_list = list(fund_df.keys())
    _fund = fund_df[config.BENCHMARK_FUND]
    file_path = config.RELATION_DATA_PATH +'fund_relation/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    date_list = _fund.drop_duplicates(subset=['end_date'])['end_date'].tolist()[::-1]
    date_list = ['20091231'] + date_list
    for i in range(0,len(date_list) - 1):
        res_set = {}
        for code in stock_code_list:
            res_set[code] = []
        for fund_code in fund_code_list:
            df = fund_df[fund_code]
            content_df = df[(df.end_date > date_list[i]) &(df.end_date <= date_list[i + 1])]
            content_df = content_df.sort_values(by=['mkv'])[0:20]
            for stock_code in content_df['symbol'].tolist():
                if stock_code in stock_code_list:
                    res_set[stock_code].append(fund_code)
        res_df = pd.DataFrame({'ts_code':res_set.keys(),'class':res_set.values()})
        res_df.to_csv(file_path + date_list[i + 1] + '.csv')
        print(date_list[i + 1],'finished')
    print('update_fund_relation() success! 更新基金持仓关系完成!')

def F_download_stock_data(start_date = config.START_DATE, end_date = get_today_date(), is_update = False):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'download_stock_data() start!')
    cn_list_data = pd.read_csv(config.LIST_DATA_PATH + 'cn_stock_list.csv', parse_dates=True, index_col=0)
    cn_list = cn_list_data['ts_code'].tolist()
    progress = 0
    error_count = 0
    start = time.time()
    for stock_code in cn_list:
        progress += 1
        if progress%199 == 0:
            time.sleep(10)
        try:
            # 获取量价数据
            use_col_01 = ['trade_date','ts_code','open','high','low',
                          'close','pre_close','pct_chg','vol','amount','turnover_rate',
                          'volume_ratio']
            print(222222,3333333666)
            data_01 = ts.pro_bar(ts_code=stock_code,start_date= start_date,end_date= end_date,asset='E',
                                                  adj='qfq',freq='D',ma=[5, 20],factors=['tor', 'vr'],adjfactor=True)
            if len(data_01) <= 0:
                print('========== download %s failed %d/%d'%(stock_code,progress,len(cn_list)))   
                continue
            data_01 = data_01[use_col_01].sort_values(by='trade_date')
            data_01.loc[:,['close_up_status']] = 0
            data_01.loc[:,['close_down_status']] = 0
            data_01.loc[data_01['pct_chg'] >= 9.8,['close_up_status']] = 1
            data_01.loc[data_01['pct_chg'] <= -9.8,['close_down_status']] = 1
            data_all = data_01 
            # 计算技术指标
            df = data_all
            data_all['MACD'] = ta.trend.MACD(df['close']).macd()
            data_all['EMA'] = ta.trend.EMAIndicator(df['close']).ema_indicator()
            data_all['ADX'] = ta.trend.ADXIndicator(df['high'],df['low'],df['close'],fillna=True).adx()
            data_all['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'],df['vol'],fillna=True).on_balance_volume()
            data_all['ROC'] = ta.momentum.ROCIndicator(df['close']).roc()
            data_all['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
            data_all['BB_high_sign'] = ta.volatility.BollingerBands(df['close']).bollinger_hband_indicator()
            data_all['BB_low_sign'] = ta.volatility.BollingerBands(df['close']).bollinger_lband_indicator()
            data_all['BB_width'] = ta.volatility.BollingerBands(df['close']).bollinger_wband() 
            data_all = data_all.sort_values(by='trade_date',ascending=False)
            if is_update :
                data_all.dropna(subset = ['MACD'],inplace = True)
            if not os.path.exists(config.STOCK_DATA_PATH):
                os.makedirs(config.STOCK_DATA_PATH)
            csv_head_insert( data_all.reset_index(drop=True),config.STOCK_DATA_PATH + stock_code + '.csv','first')
            print('========== download %s succeeded %d/%d'%(stock_code,progress,len(cn_list)))    
            error_count = 0
        except:
            error_count += 1
            traceback.print_exc()
            print('========== download %s failed %d/%d'%(stock_code,progress,len(cn_list)))   
            if progress < 10 or error_count >= 8:
                print('========== download error')   
                return False
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'download_stock_data() stopped!')
    return True

def F_realtime_download_stock_data(start_date = config.START_DATE, end_date = get_today_date(),bias = [0,0],is_update = False):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'download_stock_data() start!')
    cn_list_data = pd.read_csv(config.LIST_DATA_PATH + 'cn_stock_list.csv', parse_dates=True, index_col=0)
    cn_list = cn_list_data['ts_code'].tolist()
    progress = 0
    error_count = 0
    start = time.time()
    if bias[1] == 0:
        bias[1] = len(cn_list)
    cn_list = cn_list[bias[0]:bias[1]]
    for stock_code in cn_list:
        progress += 1
        try:
            # 获取量价数据
            use_col_01 = ['trade_date','ts_code','open','high','low',
                          'close','pre_close','pct_chg','vol','amount','turnover_rate',
                          'volume_ratio']
#             data_01 = ts.pro_bar(ts_code=stock_code,start_date= start_date,end_date= end_date,asset='E',
#                                                   adj='qfq',freq='D',ma=[5, 20],factors=['tor', 'vr'],adjfactor=True)
            data_01 = pd.read_csv(config.STOCK_DATA_PATH + stock_code + '.csv',index_col=0)
            data_01 = data_01.astype({"trade_date": str})
            data_01 = data_01[(data_01['trade_date'] >= start_date)&(data_01['trade_date'] < end_date)]
            data_new = ef.stock.get_quote_history(stock_code[0:6])
            _row = data_new[data_new['日期'] == datetime.datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d")]
            if len(data_01) <= 0 or len(data_new) <= 0 or len(_row) <= 0:
                print('========== download %s failed %d/%d'%(stock_code,progress,len(cn_list)))  
                continue
            _row = _row.iloc[0]
            beta = 1.2
            _volume_ratio = data_01[data_01['trade_date']<= end_date].iloc[0:5]['volume_ratio'].fillna(0.7).mean()
            data_01=data_01.append({'trade_date':end_date,'ts_code':stock_code,'open':_row['开盘'],'high':_row['最高'],'low':_row['最低'],
                                        'close':_row['收盘'],'pre_close':data_new.iloc[_row.name-1]['收盘'],'pct_chg':_row['涨跌幅'],'vol':_row['成交量']*beta,'amount':_row['成交额']*beta,'turnover_rate':_row['换手率'],
                                       'volume_ratio':_volume_ratio} , ignore_index=True)
            data_01 = data_01.sort_values(by=['trade_date']).reset_index(drop=True)
            data_01.loc[:,['close_up_status']] = 0
            data_01.loc[:,['close_down_status']] = 0
            data_01.loc[data_01['pct_chg'] >= 9.8,['close_up_status']] = 1
            data_01.loc[data_01['pct_chg'] <= -9.8,['close_down_status']] = 1
            data_all = data_01 
            # 计算技术指标
            df = data_all
            data_all['MACD'] = ta.trend.MACD(df['close']).macd()
            data_all['EMA'] = ta.trend.EMAIndicator(df['close']).ema_indicator()
            data_all['ADX'] = ta.trend.ADXIndicator(df['high'],df['low'],df['close'],fillna=True).adx()
            data_all['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'],df['vol'],fillna=True).on_balance_volume()
            data_all['ROC'] = ta.momentum.ROCIndicator(df['close']).roc()
            data_all['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
            data_all['BB_high_sign'] = ta.volatility.BollingerBands(df['close']).bollinger_hband_indicator()
            data_all['BB_low_sign'] = ta.volatility.BollingerBands(df['close']).bollinger_lband_indicator()
            data_all['BB_width'] = ta.volatility.BollingerBands(df['close']).bollinger_wband() 
            data_all = data_all.sort_values(by='trade_date',ascending=False)
            if is_update :
                data_all.dropna(subset = ['MACD'],inplace = True)
            if not os.path.exists(config.STOCK_DATA_PATH):
                os.makedirs(config.STOCK_DATA_PATH)
            csv_head_insert( data_all.reset_index(drop=True),config.STOCK_DATA_PATH + stock_code + '.csv','last')
            print('========== download %s succeeded %d/%d'%(stock_code,progress,len(cn_list)))    
            error_count = 0
        except:
            traceback.print_exc()
            error_count += 1
            traceback.print_exc()
            print('========== download %s failed %d/%d'%(stock_code,progress,len(cn_list)))   
            if progress < 10 or error_count >= 8:
                print('========== download error')   
                return False
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'download_stock_data() stopped!')
    return True


def download_relation_data(stock_code_list):
    update_industry_relation(stock_code_list)
    update_theme_relation(stock_code_list)
    update_fund_relation(stock_code_list)