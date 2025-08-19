from datetime import datetime

import yfinance as yf
import tushare as ts

pro = ts.pro_api("6c996ae295d0eeda12b288f0348aa841c59377ff4b21d2a4faa213c6")
import pandas as pd
import os


# # 创建一个Ticker对象来获取Google股票的数据

# 这个函数的作用是对csv进行一些处理，每一个列的列名的首字母修改大写
def capitalize_column_names(csv_file_path):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file_path)

    # 将所有列名的首字母改为大写
    df.columns = df.columns.str.capitalize()

    # 使用 rename() 方法将某个列名替换为指定的列名
    df.rename(columns={'Trade_date': 'Date'}, inplace=True)

    # 保存修改后的 DataFrame 到 CSV 文件
    df.to_csv(csv_file_path, index=False)


def get_chinese_stock(stock_code):
    # 获取当前工作目录
    current_directory = os.getcwd()
    print("当前工作目录:", current_directory)
    final_dir = f"{current_directory}/predict/dataset"
    df = pro.daily(ts_code=stock_code, start_date='20000701', end_date='20240718')
    if df.empty:
        return None
    # df.to_csv('data.csv', index=False)
    # 将日期列转换为日期时间类型
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    # 按日期列对 DataFrame 进行排序
    df.sort_values(by='trade_date', ascending=True, inplace=True)

    # 将排序后的数据保存到 CSV 文件
    df.to_csv(f'{final_dir}/{stock_code}.csv', index=False)

    # 对数据的列明进行修正
    capitalize_column_names(f"{final_dir}/{stock_code}.csv")

    # 返回保存后的文件路径
    return f"{final_dir}/{stock_code}.csv"


def check_stock_data_update(old_stock_file_path, current_stock_code):
    flag = False
    # 获取当前日期
    end_date = datetime.today().strftime('%Y%m%d')
    # 获取股票历史数据
    df_current_stock_data = pro.daily(ts_code=current_stock_code, start_date='20000701', end_date=end_date)
    df_old_stock_file_path = pd.read_csv(old_stock_file_path, parse_dates=['Date'])  # 加载数据
    update_line_site = len(df_old_stock_file_path)
    # 如果数据更新了60条，那么就要进行增量学习的代码
    if len(df_current_stock_data) - len(df_old_stock_file_path) >= 60:
        print(f"update {len(df_current_stock_data) - len(df_old_stock_file_path)} number stock data")
        df_current_stock_data['trade_date'] = pd.to_datetime(df_current_stock_data['trade_date'])
        df_current_stock_data.sort_values(by='trade_date', ascending=True, inplace=True)
        df_current_stock_data.to_csv(old_stock_file_path, index=False)
        # 对数据的列明进行修正
        capitalize_column_names(old_stock_file_path)
        flag = True
    return flag, update_line_site





def get_foreign_stock(stock_code):
    return None
