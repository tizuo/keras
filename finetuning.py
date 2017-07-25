import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Dense, Dropout, Flatten
from keras.utils import np_utils
from sklearn import preprocessing
import matplotlib.pyplot as plt

a = l = []

def build_multilayer_perceptron():
    """多層パーセプトロンモデルを構築"""
    model = Sequential()
    model.add(Dense(4, input_shape=(4, ), name='l1'))
    model.add(Activation('relu'))
    model.add(Dense(n_hidden, input_shape=(4, ), name='l2'))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    return model

# Irisデータをロード
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# データの標準化
X = preprocessing.scale(X)

# ラベルをone-hot-encoding形式に変換
Y = np_utils.to_categorical(Y)


# 訓練データとテストデータに分割
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=0.8)
print(train_X.shape, test_X.shape, train_Y.shape, test_Y.shape)

# モデル構築
model = build_multilayer_perceptron()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# モデル訓練
model.fit(train_X, train_Y, nb_epoch=50, batch_size=1, verbose=1)

# モデル評価
loss, accuracy = model.evaluate(test_X, test_Y, verbose=0)
print("Accuracy = {:.2f}".format(accuracy))

#モデル保存
model.save_weights('my_model_weights.h5')

#モデルのロード
model = Sequential()
model.add(Dense(4, input_shape=(4, ), name='l1'))
model.add(Activation('relu'))
model.add(Dense(n_hidden, input_shape=(4, ), name='l2'))
model.add(Activation('relu'))
model.add(Dense(n_hidden, input_shape=(4, ), name='l3'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))

model.load_weights('my_model_weights.h5', by_name=True)
model.summary()
