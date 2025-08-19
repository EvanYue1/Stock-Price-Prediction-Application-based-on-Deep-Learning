from matplotlib import pyplot as plt

from predict.views.lstm_cnn_attention.model import *
import numpy as np
import pandas as pd


def verify(dataset, window, model, future_days, predictionTarget):
    data = dataset.get_data()
    last_seq = data[-(window + future_days): -future_days, :]
    last_seq_tensor = torch.FloatTensor(last_seq).unsqueeze(0)
    model.eval()
    forecast = []
    for i in range(future_days):
        with torch.no_grad():
            next_day = model(last_seq_tensor)
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
    last_days_actual = dataset.get_df()['Date'].values[-(window+future_days):-future_days]
    target_data = dataset.get_df()[predictionTarget].values.astype(float).reshape(-1, 1)
    last_days_price = target_data[-(window+future_days):-future_days]

    # 将未来forecast_days天的预测数据添加到最近7天的真实数据后面
    future_days_forecast = forecast[:future_days]
    last_days_price = np.squeeze(last_days_price)
    combined_data = np.concatenate((last_days_price, future_days_forecast))

    # 将真实未来forecast_days天的预测数据添加最近window天的真实数据后面
    ture_data = dataset.get_df()[predictionTarget].values.astype(float).reshape(-1, 1)
    ture_last_price = ture_data[-future_days:]
    ture_last_price = np.squeeze(ture_last_price, axis=-1)
    combined_ture_data = np.concatenate((last_days_price, ture_last_price))

    # 获取最近last_days_dates天和未来forecast_days天的日期
    # last_days_dates = pd.date_range(start=last_days_actual[0], periods=window).strftime('%Y-%m-%d').tolist()
    last_days_dates = pd.to_datetime(last_days_actual).strftime('%Y-%m-%d').tolist()
    # future_days_dates = pd.date_range(start=last_days_actual[-1], periods=future_days + 1)[1:].strftime('%Y-%m-%d').tolist()
    future_days_dates = pd.to_datetime(dataset.get_df()['Date'].values[-future_days:]).strftime('%Y-%m-%d').tolist()
    combined_dates = last_days_dates + future_days_dates  # 折线图的横坐标
    combined_data = np.squeeze(combined_data)  # 折线图的纵坐标-forecast
    combined_ture_data = np.squeeze(combined_ture_data)  # 折线图的纵坐标-ture

    combined_data = combined_data.tolist()
    combined_ture_data = combined_ture_data.tolist()
    return combined_data, combined_ture_data, combined_dates
