from PIL import Image  # pillow 설치
import glob
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('../exam17_cat_and_dog/cat_and_dog.binary_classfication0.85.h5')
model.summary()

img_dir = '../dataset/cat_dog/train'
categories = ['cat', 'dog']

image_w = 64
image_h = 64

dog_files = glob.glob(img_dir + 'dog*')
dog_sample = np.random.randint(len(dog_files))
dog_sample_path = dog_files[dog_sample]

cat_files = glob.glob(img_dir + 'cat*')
cat_sample = np.random.randint(len(cat_files))
cat_sample_path = cat_files[dog_sample]

print(dog_sample_path)
print(cat_sample_path)

try:
    img = Image.open(dog_sample_path)
    img.show()
    img = img.convert('RGB')
    img = img.resize((image_w, image_h))
    img.show()
    data = np.asarray(img)
    data = data / 255
    dog_data = data.reshape(1, 64, 64, 3)

    img = Image.open(cat_sample_path)
    img.show()
    img = img.convert('RGB')
    img = img.resize((image_w, image_h))
    img.show()
    data = np.asarray(img)
    data = data / 255
    cat_data = data.reshape(1, 64, 64, 3)
except:
    print('error')
print('Dog data :', categories[int(model.predict(dog_data).round()[0][0])])
print('Cat data :', categories[int(model.predict(cat_data).round()[0][0])])


