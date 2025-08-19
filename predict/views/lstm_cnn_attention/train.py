import os

from torch.utils.data import DataLoader
from predict.views.lstm_cnn_attention.model import *
import torch.nn as nn
from predict.views.lstm_cnn_attention.dataloader import DataGenerator
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync


def train_lstm_cnn_attention(stock_path,
                             selectedFeatures,
                             model_type,
                             predictionTarget='Close',
                             update_line_site=1,
                             model=None):
    selectedFeatures_simply_name = ''.join(sorted(word.title()[0] for word in selectedFeatures))
    dim = len(selectedFeatures)
    window = 5
    if model_type == 'HW':
        window = 5
    elif model_type == 'SE':
        window = 5
    elif model_type == 'CBAM':
        window = 5
    elif model_type == 'Base':
        window = 5
    train_data = DataGenerator(stock_path,
                               selectedFeatures=selectedFeatures,
                               predictionTarget=predictionTarget,
                               isPredict=False,
                               window=window,
                               update_line_site=update_line_site)
    test_data = DataGenerator(stock_path,
                              selectedFeatures=selectedFeatures,
                              predictionTarget=predictionTarget,
                              isPredict=False,
                              isTest=True,
                              window=window,
                              update_line_site=update_line_site)

    print_step = 10

    train_loader = DataLoader(train_data, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    model_dict = {
        "Base": CNNLSTMModel,
        "SE": CNNLSTMModel_SE,
        "ECA": CNNLSTMModel_ECA,
        "CBAM": CNNLSTMModel_CBAM,
        "HW": CNNLSTMModel_HW
    }
    if model is None:
        model = model_dict[model_type](window=window, dim=dim)


    print(model)
    print(f"training model is {model_type}")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    min_loss = 99999
    for epoch in range(500):
        print(f'epoch:{epoch}')
        running_loss = 0.0
        for step, (data, label) in enumerate(train_loader):
            out = model(data)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % print_step == 0:  # 每500个batch打印一次训练状态
                with torch.no_grad():
                    mse_loss = 0.0
                    for data, label in test_loader:
                        out = model(data)
                        loss = criterion(out, label)
                        mse_loss += loss.item()
                    if mse_loss / len(test_loader) < min_loss:
                        # 获取当前工作目录
                        current_directory = os.getcwd()
                        print("当前工作目录:", current_directory)
                        final_dir = f"{current_directory}/predict/views/lstm_cnn_attention"

                        # 保存模型
                        stock_file_name = os.path.basename(stock_path)
                        stock_name = stock_file_name.replace('.csv', '')
                        model_name = f"{model_type}_{stock_name}_{predictionTarget}_best#{selectedFeatures_simply_name}.pth"
                        # torch.save(model.state_dict(), f'{final_dir}/model/{model_name}')
                        print("save_best")
                        min_loss = mse_loss / len(test_loader)
                    print(f"step:{step}, test loss:{mse_loss / len(test_loader)}")
            # send progress bar information
            progress = int((epoch / 500) * 100)
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                'progress-bar-1',
                {
                    'type': 'send.progress',
                    'progress': progress,
                }
            )
    print("done")
    torch.save(model.state_dict(), f'{final_dir}/model/{model_name}')
    return model
