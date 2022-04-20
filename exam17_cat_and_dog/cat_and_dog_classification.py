import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

X_train, X_test, Y_train, Y_test = np.load('../dataset/binary_image_data.npy', allow_pickle=True)  # 피클 누구세요?
print('X_train shape :', X_train.shape)
print('Y_train shape :', Y_train.shape)
print('X_test shape :', X_test.shape)
print('Y_test shape :', Y_test.shape)

model = Sequential()
model.add(Conv2D(32, input_shape=(64, 64, 3), kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
early_stopping = EarlyStopping(monitor='val_loss', patience=7)  # 뭘 모니터링할지는 판단해서 정하면 됨 ex. val_acc 하든가
# 얘가 7에폭 동안 더 좋아지지 않으면 추가 학습이 의미가 없다고 판단해서 스탑

fit_hist = model.fit(X_train, Y_train, batch_size=64, epochs=50, validation_split=0.15)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Evaluation loss :', score[0])
print('Evaluation acc :', score[1])

model.save('../models/cat_and_dog_binary_classfication{}.h5'.format(score[1]))

plt.plot(fit_hist.history['loss'], label='loss')
plt.plot(fit_hist.history['val_loss'], label='val_loss')
plt.plot(fit_hist.history['acc'], label='accuracy')
plt.plot(fit_hist.history['val_acc'], label='val_accuracy')
plt.legend()
plt.show()