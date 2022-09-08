"""
2021 2학년 2학기
20105 설규원 수학2 주제탐구
주제: 딥러닝을 활용해 함수의 값, 미분계수, 정적분 값 예측하기!
사용 라이브러리: Tensorflow(딥러닝), numpy(어레이 데이터), matplotlib(데이터 시각화)
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import random


# 3차부터 5차까지의 차수 정하기
dimension = random.randint(3, 5)

# 계수 리스트(상수항부터 고차항으로 올라감)
factor_list = []

# 차수 + 1개의 계수 필요
for i in range(dimension + 1):
    # 계수 리스트에 -10부터 10까지의 정수 추가하기
    factor_list.append(random.randint(-10, 10))


# 함수식 설정하기 (인공지능에게는 일부 함숫값만이 주어짐)
def f(x):
    result_value = 0
    for current_dim in range(dimension + 1):
        # result_value에, (각 항의 계수) * (x값^current_dim) 을 더함
        result_value += factor_list[current_dim] * (x**current_dim)
    return result_value


# f(x)의 도함수
def der_f(x):
    result_value = 0
    # 한 차수 작음
    for current_dim in range(dimension):
        # 계수 = (current+1차항 계수) * (current+1) * (x값^current_dim)
        result_value += (
            factor_list[current_dim + 1] * (current_dim + 1) * (x**current_dim)
        )
    return result_value


# 학습 데이터 만들기
train_x_data = []
train_y_data = []
for i in range(500):
    # -50부터 50 사이의 x값에 대한 데이터 500개
    random_int = random.randint(-50, 50)
    train_x_data.append(random_int)
    train_y_data.append(f(random_int))
train_x_data = np.array(train_x_data)
train_y_data = np.array(train_y_data)

# 모델 생성, 층 설정
model = keras.Sequential()
model.add(layers.Dense(12, input_dim=1, activation="relu"))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(1))

adam = optimizers.Adam(learning_rate=0.01)
model.compile(loss="mse", optimizer=adam)

# 모델 학습시키고 history 에 저장
history = model.fit(
    train_x_data, train_y_data, epochs=100, batch_size=1, shuffle=False, verbose=1
)


# 학습된 모델로 함수 예측하기
x_data = []
y_data = []
predict_y_data = []
# 반복문으로 -50부터 50까지 예측
for i in range(-50, 51):
    x_data.append(i)
    y_data.append(f(i))
    predict_y_data.append(float(model.predict([i])))
x_data = np.array(x_data)
y_data = np.array(y_data)
predict_y_data = np.array(predict_y_data)


# 예측한 모델의 도함수 만들기
def predict_der_f(x):
    # 극한 대신 0에 가까운 값(0.0001) 사용
    delta_x = 1e-4
    # 도함수의 정의 사용 ( lim:h->0 에서 (f(x+h)-f(x))/h )
    return float(model.predict([x + delta_x]) - model.predict([x])) / delta_x


# 실제 도함수와 예측된 도함수 비교
der_x_data = []
der_y_data = []
predict_der_y_data = []
# -50부터 50까지 예측
for i in range(-50, 51):
    der_x_data.append(i)
    der_y_data.append(der_f(i))
    predict_der_y_data.append(predict_der_f(i))
der_x_data = np.array(der_x_data)
der_y_data = np.array(der_y_data)
predict_der_y_data = np.array(predict_der_y_data)


# 함수 그래프
plt.subplot(221)
plt.plot(x_data, y_data, color="red")
plt.plot(x_data, predict_y_data, color="blue")

# 도함수 그래프
plt.subplot(222)
plt.plot(der_x_data, der_y_data, color="red")
plt.plot(der_x_data, predict_der_y_data, color="blue")
plt.savefig("results.png")
