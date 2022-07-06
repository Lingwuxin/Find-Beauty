from PIL import Image
import matplotlib.pyplot as plt
import os
import traceback
from keras import datasets, layers, models
from handlie_img import HandleImg
import cv2
import numpy as np
batch_size = 16


class FindBeauty():
    def __init__(self):
        self.hm = HandleImg()

    def creatMymodel(self, path):
        if os.path.exists(path):
            self.model = models.load_model(path)
        else:
            self.model = models.Sequential()
            self.model.add(layers.Conv2D(
                8, (3, 3), activation='relu', input_shape=(64, 64, 3)))
            self.model.add(layers.Conv2D(
                16, (3, 3), activation='relu'))
            self.model.add(layers.Flatten())
            self.model.add(layers.Dense(512, activation='relu'))
            self.model.add(layers.Dense(1, activation='sigmoid'))

    # Please place two different types of pictures in the following files
    # FindBeauty/img1/
    # FindBeauty/img2/
    # And use the number starting with 1 as the file name of the picture
    def load_data(self, img_type: str):
        images = []
        for i in range(1, 401):
            img1 = cv2.imread('FindBeauty/img1/'+str(i)+img_type)
            img2 = cv2.imread('FindBeauty/img2/'+str(i)+img_type)
            img1 = cv2.resize(img1, dsize=(64, 64))
            img2 = cv2.resize(img2, dsize=(64, 64))
            images.append(img1)
            images.append(img2)
        images = np.array(images)
        labels = []
        for i in range(1, 401):
            labels.append([0])
            labels.append([1])
        labels = np.array(labels)
        return images, labels

    # Please store the test pictures in the following directory
    # FindBeauty/testdata/
    def test(self, img_path: str):
        plt.draw()
        for root, dirs, files in os.walk(img_path):
            for path in files:
                img_path = root+path
                res = self.predict(img_path)
                print(res)

                print('Picture type:')
                if res < 0.5:
                    print('Painting!')
                else:
                    print('Photo!')
                plt.imshow(Image.open(img_path))
                plt.pause(3)

    def fit(self, model_path):

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        # self.model.fit(self.hm.__call__(path='FindBeauty/img'),epochs=16)
        x_train, y_train = self.load_data()
        self.model.fit(x_train, y_train, batch_size=16, epochs=4)
        self.model.save(model_path)

    def predict(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, dsize=(64, 64))
        img = img.reshape(1, 64, 64, 3)
        r = self.model.predict(img)
        res = r[0][0]
        return res


def __main__():
    path = 'FindBeauty/model-test4'
    fb = FindBeauty()
    fb.creatMymodel(path)
    fb.test('FindBeauty/testimg/')

#hellow
if __name__ == '__main__':
    __main__()
