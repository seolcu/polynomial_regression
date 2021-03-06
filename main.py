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


def f(x):
    # 함수식 설정하기 (인공지능에게는 일부 함숫값만이 주어짐)
    return 8 * (x ** 3) + 6 * (x ** 2) + 4 * (x ** 1) + 10

def der_f(x):
    # f(x)를 미분한 식 (다르면 안됨!)
    return 24 * (x ** 2) + 12 * (x ** 1) + 4

def int_f(x):
    # f(x)를 적분한 식 (정적분에 사용되므로 적분상수 C 무시하기!)
    return 2 * (x ** 4) + 2 * (x ** 3) + 2 * (x ** 2) + 10 * (x ** 1)


# 빈 데이터 어레이 생성
x_data = []
y_data = []
for i in range(-100, 101, 2):
    # 반복문으로 -100부터 100까지 짝수인 x값에 대한 데이터 생성
    x_data.append(i)
    y_data.append(f(i))
x_data = np.array(x_data)
y_data = np.array(y_data)


# 모델 생성, 은닉층 설정
model = keras.Sequential()
model.add(layers.Dense(30, input_dim=1, activation="relu"))
model.add(layers.Dense(30, activation="relu"))
model.add(layers.Dense(30, activation="relu"))
model.add(layers.Dense(1))


adam = optimizers.Adam(learning_rate=0.01)
model.compile(loss="mse", optimizer=adam, metrics=["accuracy"])

# 모델 학습시키고 history 에 저장
history = model.fit(x_data, y_data, epochs=500, batch_size=1, shuffle=False, verbose=1)

# 빈 테스트 데이터 어레이 생성
predict_x_data = []
predict_y_data = []
for i in range(-99, 101, 2):
    # 반복문으로 -99부터 99까지 홀수인 x값에 대한 데이터 예측
    predict_x_data.append(i)
    # float: 실수화
    predict_y_data.append(float(model.predict([i])))
predict_x_data = np.array(predict_x_data)
predict_y_data = np.array(predict_y_data)


# 예측한 모델의 도함수
def predict_der_f(x):
    # 0에 가까운 값
    delta_x = 1e-4
    # 도함수의 정의 사용 ( lim:h->0 에서 (f(x+h)-f(x))/h )
    return float(model.predict([x + delta_x]) - model.predict([x]))/delta_x


# 도함수 비교를 위한 빈 배열 생성
der_x_data = []
der_y_data = []
der_predict_y_data = []
for i in range(-100, 101, 1):
    der_x_data.append(i)
    der_y_data.append(der_f(i))
    der_predict_y_data.append(float(predict_der_f(i)))
der_x_data = np.array(der_x_data)
der_y_data = np.array(der_y_data)
der_predict_y_data = np.array(der_predict_y_data)



# epoch에 대한 오차값 그래프
plt.subplot(221)
plt.plot(history.history["loss"])
plt.title("Model accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Accuracy", "Loss"], loc="upper left")

# 함수 그래프
plt.subplot(222)
plt.plot(x_data, y_data, color="red")
plt.plot(predict_x_data, predict_y_data, color="blue")

# 도함수 그래프
plt.subplot(223)
plt.plot(der_x_data, der_y_data, color="red")
plt.plot(der_x_data, der_predict_y_data, color="blue")
plt.savefig("results.png")