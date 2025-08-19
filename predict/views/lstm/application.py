import torch
import numpy as np
import pandas as pd
from predict.views.lstm.model import LSTM
import os
from predict.views.lstm.test import test
from predict.views.lstm.dataloader import DataGenerator
from predict.views.get_stock import get_chinese_stock
from predict.views.get_stock import check_stock_data_update
from predict.views.lstm.train import train_lstm
from predict.views.lstm.verfiy import verify
from datetime import datetime

global_models = {}


def find_file_in_directory(directory, filename):
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


# update the model
def update_lstm_model(stock_path, model, predictionTarget='Close', update_line_site=1, window=5):
    # call the training function
    model = train_lstm(stock_path, model, predictionTarget, update_line_site, window=window)
    return model


def lstm(model_name, data_path, window=5, future_days=5, predictionTarget='Close'):
    # model_name 就是股票的代码
    global global_models
    stock_file_name = os.path.basename(data_path)
    stock_name = stock_file_name.replace('.csv', '')

    # 初始化模型参数
    input_dim = 1
    hidden_dim = 16
    num_layers = 1
    output_dim = 1

    # 获取当前工作目录
    current_directory = os.getcwd()
    print("当前工作目录:", current_directory)
    final_dir = f"{current_directory}"

    # 首先判断是否存在要进行预测的股票的模型
    model_path = find_file_in_directory(f"{final_dir}/predict/views/lstm/model/", f"{model_name}")
    dataset = None
    if model_path is not None:
        # make data
        dataset = DataGenerator(data_path, predictionTarget=predictionTarget)
        # load the model
        if model_name not in global_models.keys():
            # 实例化模型、损失函数和优化器
            model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim)
            params = torch.load(f"{final_dir}/predict/views/lstm/model/{model_name}")
            print(f"加载的模型为：{final_dir}/predict/views/lstm/model/{model_name}", )
            try:
                model.load_state_dict(params)
            except Exception as e:
                print("加载模型参数时发生异常:", e)
            # 在加载这个模型之前要检测是否模型需要更新，更新标准：如果数据更新了20条数据，则要对模型更新
            # flag1, update_line_site = check_stock_data_update(data_path, stock_name)
            # if flag1 is True:
            #     # executed the update model
            #     model = update_lstm_model(data_path, model, predictionTarget, update_line_site, window=window)
            #     dataset = DataGenerator(data_path, predictionTarget=predictionTarget)
            global_models[model_name] = model
    else:
        """
            if this stock model doesn't exit, then we need to further determine whether the stock exits, because 
            there may be this stock, but there is no training model for this feature
        """
        if find_file_in_directory(f"{final_dir}/predict/dataset/", f"{stock_file_name}") is None:
            # 首先通过接口获取该股票的历史交易数据
            stock_path = get_chinese_stock(stock_name)
            # make data
            dataset = DataGenerator(stock_path, predictionTarget=predictionTarget)
            # 使用股票数据模型训练
            global_models[model_name] = train_lstm(stock_path, predictionTarget=predictionTarget, window=window)
        else:
            """
                represent the existence of this stock, but there is no training model for this specific feature of this
                stock, so it is only necessary to train for this feature
            """
            # check_stock_data_update(data_path, stock_name)
            dataset = DataGenerator(data_path, predictionTarget)
            global_models[model_name] = train_lstm(data_path, predictionTarget=predictionTarget, window=window)
    # 使用模型进行未来几天的预测
    data_normalized = dataset.get_data()
    last_seq = data_normalized[-window:]
    last_seq_tensor = torch.FloatTensor(last_seq).unsqueeze(0)
    forecast = []
    global_models[model_name].eval()
    for i in range(future_days):
        with torch.no_grad():
            next_day = global_models[model_name](last_seq_tensor)
            forecast.append(next_day.item())
            last_seq = np.roll(last_seq, -1)
            last_seq[-1] = next_day
            last_seq_tensor = torch.FloatTensor(last_seq).unsqueeze(0)

    scaler = dataset.get_scaler()
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    print(forecast)

    # 获取最近30天的真实股票价格数据的日期和价格
    last_days_actual = dataset.get_df()['Date'].values[-30:]
    target_data = dataset.get_df()[predictionTarget].values.astype(float).reshape(-1, 1)
    last_days_price = target_data[-30:]

    # 将未来forecast_days天的预测数据添加到最近7天的真实数据后面
    future_days_forecast = forecast[:future_days]
    combined_data = np.concatenate((last_days_price, future_days_forecast))

    # 获取最近last_days_dates天和未来forecast_days天的日期
    # last_days_dates = pd.date_range(start=last_days_actual[0], periods=30).strftime('%Y-%m-%d').tolist()
    last_days_dates = pd.to_datetime(last_days_actual).strftime('%Y-%m-%d').tolist()
    future_days_dates = pd.date_range(start=last_days_actual[-1], periods=future_days + 1)[1:].strftime(
        '%Y-%m-%d').tolist()
    combined_dates = last_days_dates + future_days_dates
    combined_data = np.squeeze(combined_data)
    combined_data = combined_data.tolist()

    # get the test result
    y_pred_inv, y_true_inv = test(model=global_models[model_name], data_path=data_path,
                                  predictionTarget=predictionTarget, window=window)
    y_pred_inv = y_pred_inv.tolist()
    y_true_inv = y_true_inv.tolist()

    # get the verify result
    combined_verify_data, combined_verify_ture_data, combined_verify_dates = verify(dataset, window,
                                                                                    global_models[model_name],
                                                                                    future_days, predictionTarget)

    return combined_data, combined_dates, y_pred_inv, y_true_inv, combined_verify_data, combined_verify_ture_data, combined_verify_dates
