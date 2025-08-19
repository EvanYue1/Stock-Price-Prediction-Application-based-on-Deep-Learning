from predict.views.lstm_cnn_attention.dataloader import DataGenerator
from predict.views.lstm_cnn_attention.model import *
from predict.views.lstm_cnn_attention.train import train_lstm_cnn_attention
from predict.views.lstm_cnn_attention.test import test
from predict.views.get_stock import get_chinese_stock
from predict.views.get_stock import check_stock_data_update
from predict.views.lstm_cnn_attention.verify import verify

import numpy as np
import pandas as pd
import os

global_attention_models = {}


def find_file_in_directory(directory, filename):
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


def update_lstm_cnn_attention_model(stock_path,
                                    selectedFeatures,
                                    model_type,
                                    model,
                                    predictionTarget='Close',
                                    update_line_site=1):
    # call the training function
    model = train_lstm_cnn_attention(stock_path, selectedFeatures, model_type, predictionTarget, update_line_site,
                                     model)
    return model


def lstm_cnn_se(model_name, data_path, selectedFeatures, window=5, future_days=5, predictionTarget='Close'):
    global global_attention_models
    dim = len(selectedFeatures)
    stock_file_name = os.path.basename(data_path)
    stock_name = stock_file_name.replace('.csv', '')
    forecast = []
    # make data
    dataset = None
    # 获取当前工作目录
    current_directory = os.getcwd()
    print("当前工作目录:", current_directory)
    model_dict = {
        "Base": CNNLSTMModel,
        "SE": CNNLSTMModel_SE,
        "ECA": CNNLSTMModel_ECA,
        "CBAM": CNNLSTMModel_CBAM,
        "HW": CNNLSTMModel_HW
    }
    model_type = model_name.split("_")[0]
    if model_type == 'HW':
        window = 5
    elif model_type == 'SE':
        window = 5
    elif model_type == 'CBAM':
        window = 5
    elif model_type == 'Base':
        window = 5
    # 首先判断是否存在要进行预测的股票的模型
    model_path = find_file_in_directory(f"{current_directory}/predict/views/lstm_cnn_attention/model/", f"{model_name}")
    if model_path is not None:
        dataset = DataGenerator(data_path, selectedFeatures, predictionTarget, window=window, isPredict=True)

        if model_name not in global_attention_models.keys():
            final_dir = "{}/predict/views/lstm_cnn_attention".format(current_directory)
            model = model_dict[model_type](dim=dim)
            print("------加载的模型类型是:", model_type)
            params = torch.load(f"{final_dir}/model/{model_name}")
            try:
                model.load_state_dict(params)
            except Exception as e:
                print("加载模型参数时发生异常:", e)
            # 在加载这个模型之前要检测是否模型需要更新，更新标准：如果数据更新了20条数据，则要对模型更新
            # flag1, update_line_site = check_stock_data_update(data_path, stock_name)
            # if flag1 is True:
            #     # executed the update model
            #     model = update_lstm_cnn_attention_model(data_path, selectedFeatures, model_type, model,
            #                                             predictionTarget, update_line_site)
            global_attention_models[model_name] = model
            print("----model----: \n", model)

    else:
        """
            if this stock model doesn't exit, then we need to further determine whether the stock exits, because 
            there may be this stock, but there is no training model for this feature
        """
        if find_file_in_directory(f"{current_directory}/predict/dataset", f"{stock_file_name}") is None:
            # 首先通过接口获取该股票的历史交易数据
            stock_path = get_chinese_stock(stock_name)
            # make data
            dataset = DataGenerator(stock_path, selectedFeatures, predictionTarget, window=window, isPredict=True)
            # 使用股票数据模型训练
            global_attention_models[model_name] = train_lstm_cnn_attention(stock_path,
                                                                           model_type=model_type,
                                                                           selectedFeatures=selectedFeatures,
                                                                           predictionTarget=predictionTarget)
        else:
            """
                represent the existence of this stock, but there is no training model for this specific feature of this
                stock, so it is only necessary to train for this feature
            """
            dataset = DataGenerator(data_path, selectedFeatures, predictionTarget, isPredict=True, window=window)
            # check_stock_data_update(data_path, stock_name)
            global_attention_models[model_name] = train_lstm_cnn_attention(data_path,
                                                                           model_type=model_type,
                                                                           selectedFeatures=selectedFeatures,
                                                                           predictionTarget=predictionTarget)
            print("----model----: \n", global_attention_models[model_name])
            print("I am Here")

    data = dataset.get_data()
    last_seq = data[-window:, :]
    last_seq_tensor = torch.FloatTensor(last_seq).unsqueeze(0)
    (global_attention_models[model_name]).eval()
    for i in range(future_days):
        with torch.no_grad():
            next_day = global_attention_models[model_name](last_seq_tensor)
            forecast.append(next_day.item())
            last_seq = np.roll(last_seq, -1)
            last_seq[-1] = next_day
            last_seq_tensor = torch.FloatTensor(last_seq).unsqueeze(0)
    # 进行返归一化
    min_val_close, max_val_close = dataset.get_min_max_val_close()
    forecast = np.array(forecast)
    forecast = forecast * (max_val_close - min_val_close) + min_val_close
    print(f"the future {future_days} data is :\n", forecast)

    # 获取最近30天的真实股票价格数据的日期和价格
    last_days_actual = dataset.get_df()['Date'].values[-30:]
    target_data = dataset.get_df()[predictionTarget].values.astype(float).reshape(-1, 1)
    last_days_price = target_data[-30:]

    # 将未来forecast_days天的预测数据添加到最近7天的真实数据后面
    future_days_forecast = forecast[:future_days]
    last_days_price = np.squeeze(last_days_price)
    combined_data = np.concatenate((last_days_price, future_days_forecast))

    # 获取最近last_days_dates天和未来forecast_days天的日期
    # last_days_dates = pd.date_range(start=last_days_actual[0], periods=30).strftime('%Y-%m-%d').tolist()
    last_days_dates = pd.to_datetime(last_days_actual).strftime('%Y-%m-%d').tolist()
    future_days_dates = pd.date_range(start=last_days_actual[-1], periods=future_days + 1)[1:].strftime(
        '%Y-%m-%d').tolist()
    combined_dates = last_days_dates + future_days_dates  # 折线图的横坐标
    combined_data = np.squeeze(combined_data)  # 折线图的纵坐标
    combined_data = combined_data.tolist()

    # get the test result
    y_pred_inv, y_true_inv = test(model=global_attention_models[model_name],
                                  stock_path=data_path,
                                  selectedFeatures=selectedFeatures,
                                  predictionTarget=predictionTarget,
                                  window=window)

    y_pred_inv = y_pred_inv.tolist()
    y_true_inv = y_true_inv.tolist()

    # get the verify result
    combined_verify_data, combined_verify_ture_data, combined_verify_dates = verify(dataset, window, global_attention_models[model_name], future_days, predictionTarget)

    return combined_data, combined_dates, y_pred_inv, y_true_inv, combined_verify_data, combined_verify_ture_data, combined_verify_dates
