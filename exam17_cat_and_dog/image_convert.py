from PIL import Image  # pillow 설치
import glob
import numpy as np
from sklearn.model_selection import train_test_split

img_dir = '../datasets/train/'
categories = ['cat', 'dog']

image_w = 64
image_h = 64

X = []
Y = []
files = None
for idx, category in enumerate(categories):
    files = glob.glob(img_dir + category + '*')
    for i, f in enumerate(files):
        try:
            img = Image.open(f)
            img = img.convert('RGB')
            img = img.resize((image_w, image_h))
            data = np.asarray(img)
            X.append(data)
            Y.append(idx)
            if i % 300 == 0:
                print(category, ':', f)
        except:
            print(category, i, f)

X = np.array(X)
Y = np.array(Y)

print(X[0])
print(Y[:5])

X = X / 255

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1)

xy = (X_train, X_test, Y_train, Y_test)
np.save('../datasets/binary_image_data.npy', xy)