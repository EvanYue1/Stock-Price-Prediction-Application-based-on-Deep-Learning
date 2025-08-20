# Stock Price Multi-Day Prediction Web Application Based on Deep Learning

## System Demo Video

A short demonstration of the system is available at:

`videos/display.mp4`
[System Demo Video](videos/display.mp4)

## Project Overview

This is a web application for **multi-day stock price prediction**, using deep learning models to continuously forecast future stock prices with interactive visualization.
The system integrates **LSTM, LSTM-CNN, and CBAM-enhanced LSTM-CNN** models, supporting multi-day prediction and incremental learning for real-time updates and efficient forecasting.

> **Core highlight:** The multi-day prediction method iteratively forecasts the next n days using a sliding window, without retraining the model each time.

---

## Key Features

* **Multi-Day Prediction**: Continuously forecast the stock prices for the next n days
* **Multiple Model Support**: LSTM / LSTM-CNN / LSTM-CNN-CBAM
* **Self update**: Automatically update models with new data
* **Automatic Stock Data Collection**: Supports multiple stocks
* **Interactive Visualization**: Line charts for historical, predicted, and validation data
* **Efficient Model Management**: Django global variable prevents repeated model loading

---

## Technology Stack

* **Backend**: Python, Django 5.0.4
* **Frontend**: HTML, Bootstrap 3.4.1, ECharts, jQuery 3.7.1
* **Deep Learning Framework**: PyTorch 2.2.2
* **Data Interface**: Tushare 1.4.16
* **Other Libraries**: NumPy 1.26.4, Pandas 2.2.1, Channels 4.1.0

---

## System Architecture

### Frontend

* **Left input panel**: stock code, forecast days, model selection, feature indicators, prediction target
* **Right display panel**: predicted results, test results, validation results

### Backend

* `/view` → Returns HTML pages
* `/predict` → Performs prediction, training, testing, and validation

### Global Variable Optimization

* First loaded model is stored in Django `global_model`

---

## Multi-Day Prediction (Core Highlight)

The models are trained using a **10-day input → next-day prediction** approach.
To forecast multiple days, the system uses the following method:

### 1. Sliding Window Prediction

* Predict the next day
* Append prediction to input sequence
* Slide the window and iterate for day 2, 3 … n

### 2. Workflow

The system workflow integrates frontend requests, backend processing, neural network computation, and multi-day prediction. The detailed steps are as follows:

#### 2.1. User Request

* Client sends a prediction request via frontend (stock code, forecast days, model selection, feature indicators)

#### 2.2. Model Check & Loading

* Backend checks if trained model exists for requested stock
* If exists, checks whether already loaded in `global_model`

  * If not loaded → load into `global_model`
* If not exists / outdated →

  * Collect stock data via Tushare
  * Train neural network with new data
  * Save & load into `global_model`

#### 2.3. Incremental Learning (if necessary)

* For existing models, if new data exceeds threshold → update with **incremental learning** (avoid full retraining)

#### 2.4. Prediction, Testing, and Validation

* Model predicts stock prices
* Multi-day prediction via sliding window iterative method:

  * Predict next day → append result → slide window → repeat until n days
* Test & validation: compare predictions with actual stock data

#### 2.5. Return Results to Frontend

* Return predictions, test results, validation charts
* Frontend renders **interactive line charts** (historical, predicted, validation data)

---

### 3. Validation Method

* Construct validation set: `window_size + forecast_day` days
* Iteratively predict future days
* Compare predictions with actual values

### 4. Advantages

* No retraining required for multi-day forecasting
* Supports user-defined prediction horizons
* Long-term accuracy maintained with incremental learning

---

## Data Preprocessing

* Fetch daily stock trading data via **Tushare API**
* **Min-Max Normalization**:

  $$
  x'_i = \frac{x_i - \min}{\max - \min}
  $$
* **Time-Series Transformation**:

  * Sliding window, default **10-day input → 1-day prediction**
  * Shift window to generate training sequences

---

## Model Architectures

### LSTM

* Structure: LSTM + Dropout + Fully Connected → Output
* Dropout prevents overfitting

### LSTM-CNN

* CNN extracts features → LSTM models time dependencies → FC layer output

### LSTM-CNN-CBAM

* **CBAM Attention Mechanism**:

  * **SE module**: channel attention
  * **HW module**: spatial attention

---

## Model Training

* **Loss Function**: Mean Squared Error (MSE)
* **Optimizer**: Adam
* Batch training with **500 epochs**

---

## Project Structure

```
/project-root
│
├── predict/             # Django prediction app
├── templates/           # HTML templates
├── static/              # CSS/JS files
├── models/              # Trained model weights
├── data/                # Dataset
├── requirements.txt
└── manage.py
```

---

## Evaluation Metrics

* **RMSE (Root Mean Squared Error)**
* **DA (Directional Accuracy)**
* Validation results graded: **A+, A, B, C**

---

## Self Learning & Multi-Stock Prediction

* Automatically update models
* New stocks trigger data collection & model training
* Efficient model management prevents repeated loading


---

## Effect display:

* Stock Symbol: 000869.SZ
* Prediction Days: 3
* Units Day
* Model: LSTM-CNN-CBAM
* Target Feature: Open Price
<img width="1440" height="790" alt="Screenshot 2025-08-19 at 6 27 00 PM" src="https://github.com/user-attachments/assets/86bdc98c-a181-4045-92ba-6d3517ee63af" />
<img width="1438" height="837" alt="Screenshot 2025-08-19 at 6 27 23 PM" src="https://github.com/user-attachments/assets/6681ed88-5772-4e44-82d8-946e3c34f1da" />
<img width="1440" height="835" alt="Screenshot 2025-08-19 at 6 27 40 PM" src="https://github.com/user-attachments/assets/31740e59-0a9a-4236-b885-d27c42b80457" />




* Stock Symbol: 000869.SZ
* Prediction Days: 3
* Units: Day
* Model: LSTM-CNN
* Target Feature: Open Price
<img width="1440" height="786" alt="Screenshot 2025-08-19 at 6 29 45 PM" src="https://github.com/user-attachments/assets/db4298a6-60b2-4042-8a2b-f3a7c8ed267d" />
<img width="1439" height="831" alt="Screenshot 2025-08-19 at 6 29 59 PM" src="https://github.com/user-attachments/assets/7cf2f9e6-4d95-4d30-865a-271c2249e41d" />
<img width="1439" height="832" alt="Screenshot 2025-08-19 at 6 30 13 PM" src="https://github.com/user-attachments/assets/292885d0-b01f-42e4-a3f2-bace098837b6" />



* Stock Symbol: 002007.SZ
* Prediction Days: 3
* Units: Day
* Model: LSTM-CNN-CBAM
* Target Feature: Open Price
<img width="1440" height="787" alt="Screenshot 2025-08-19 at 6 43 01 PM" src="https://github.com/user-attachments/assets/3b4540a9-aa2b-4117-9f4d-11b5b830678b" />
<img width="1438" height="830" alt="Screenshot 2025-08-19 at 6 43 18 PM" src="https://github.com/user-attachments/assets/131edd05-e447-482a-acac-d18a05c012df" />
<img width="1438" height="833" alt="Screenshot 2025-08-19 at 6 43 37 PM" src="https://github.com/user-attachments/assets/74b1fe65-741c-46d7-b7b9-69f71f338b97" />












