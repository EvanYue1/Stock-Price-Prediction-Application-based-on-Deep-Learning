import os
import torch
from predict.views.lstm_cnn_attention.dataloader import DataGenerator
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error



def directional_accuracy(y_pred, y_gt):
    # 计算方向准确度
    correct_predictions = np.sum(np.sign(y_pred[1:] - y_pred[:-1]) == np.sign(y_gt[1:] - y_gt[:-1]))
    total_predictions = len(y_pred) - 1
    accuracy = correct_predictions / total_predictions
    return accuracy


def test(model, stock_path, selectedFeatures, predictionTarget='Close', window=5):
    # 获取当前工作目录
    current_directory = os.getcwd()
    print("当前工作目录:", current_directory)
    final_dir = f"{current_directory}/predict/views/lstm_cnn_attention"

    # 准备测试机数据
    test_data = DataGenerator(stock_path,
                              selectedFeatures=selectedFeatures,
                              predictionTarget=predictionTarget,
                              isPredict=False,
                              isTest=True,
                              window=window)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    model = model
    criterion = nn.MSELoss()

    eval_loss = 0.0

    with torch.no_grad():
        y_gt = []
        y_pred = []
        for data, label in test_loader:
            y_gt += label.numpy().squeeze(axis=1).tolist()
            out = model(data)
            loss = criterion(out, label)
            eval_loss += loss.item()
            y_pred += out.numpy().squeeze(axis=1).tolist()
        print(len(y_gt), len(y_pred))

    y_gt = np.array(y_gt)
    y_gt = y_gt[:, np.newaxis]
    y_pred = np.array(y_pred)
    y_pred = y_pred[:, np.newaxis]

    # 进行返归一化
    min_val_close, max_val_close = test_data.get_min_max_val_close()
    y_pred = y_pred * (max_val_close - min_val_close) + min_val_close
    y_gt = y_gt * (max_val_close - min_val_close) + min_val_close

    y_pred = np.squeeze(y_pred)
    y_gt = np.squeeze(y_gt)
    y_gt_copy = y_gt.copy()
    y_pred_copy = y_pred.copy()

    accuracy = directional_accuracy(y_pred_copy, y_gt_copy)
    print("方向准确性：", accuracy)

    # 计算RMSE

    test_rmse = np.sqrt(mean_squared_error(y_gt, y_pred))
    print('Test RMSE: {:.6f}'.format(test_rmse))

    return y_pred, y_gt
