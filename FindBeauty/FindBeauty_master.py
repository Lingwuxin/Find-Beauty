from math import sqrt
from PIL import Image
import matplotlib.pyplot as plt
import os
import keras
import threading
from keras import datasets, layers, models
from handlie_img import HandleImg
import cv2
import numpy as np
batch_size = 16


class FindBeauty():
    def __init__(self):
        self.hm = HandleImg()
        self.k_1 = 16
        self.k_2 = 32

        def get_size(k=0, size=[5, 5]):
            if k**0.5 > 5:
                size[0] = 5
                size[1] = (k//5)+1
            else:
                size[0] = int(k**0.5//1)+1
                size[1] = size[0]
            return size
        self.size_1 = get_size(k=self.k_1)
        self.size_2 = get_size(k=self.k_2)

    def creatMymodel(self, path):
        if os.path.exists(path):
            self.model = models.load_model(path)
        else:
            self.model = models.Sequential()
            self.model.add(layers.Conv2D(
                self.k_1, (3, 3), activation='relu', input_shape=(64, 64, 3)))
            self.model.add(layers.Conv2D(
                self.k_2, (3, 3), activation='relu'))
            self.model.add(layers.Flatten())
            self.model.add(layers.Dense(512, activation='relu'))
            self.model.add(layers.Dense(1, activation='sigmoid'))

    # Please place two different types of pictures in the following files
    # FindBeauty/img1/
    # FindBeauty/img2/
    # And use the number starting with 1 as the file name of the picture
    def load_data(self, img_type='.jpg'):
        images = []
        num=550
        for i in range(1, num):
            img1 = cv2.imread('FindBeauty/img_er/'+str(i)+img_type)
            img2 = cv2.imread('FindBeauty/img_vive/'+str(i)+img_type)
            img1 = cv2.resize(img1, dsize=(64, 64))
            img2 = cv2.resize(img2, dsize=(64, 64))
            images.append(img1)
            images.append(img2)
        images = np.array(images)
        labels = []
        for i in range(1, num):
            labels.append([0])
            labels.append([1])
        labels = np.array(labels)
        return images, labels

    def visualization(self, img_path: str):
        def flip_horizontally(arr: np):
            print(arr.shape)
            arr2 = arr.copy()
            arr2 = arr.reshape(int(arr.size/3), 3)
            arr2 = np.array(arr2[::-1])
            arr2 = arr2.reshape(arr.shape[0], arr.shape[1], arr.shape[2])
            return arr2[::-1]

        def loking_model(model, layers: list) -> keras.Model():
            sub_model = keras.Model(inputs=model.get_layer(layers[0]).input,
                                    outputs=model.get_layer(layers[1]).output)
            return sub_model

        def one_outputs(img_path):
            def resize_img(img_path, img_size=(), shape=[]):
                img = cv2.imread(img_path)
                img = cv2.resize(img, dsize=img_size)
                img = img.reshape(shape)
                return img
            f_img = resize_img(img_path=img_path,
                               img_size=(64, 64),
                               shape=[1, 64, 64, 3])
            model = loking_model(self.model, ['conv2d', 'conv2d'])
            return model(f_img)

        def two_outputs(imgs):
            model = loking_model(self.model, ['conv2d_1', 'conv2d_1'])
            return model(imgs)

        def show_one_out(one_out):
            imgs = np.array(one_out)
            imgs = imgs.reshape(62, 62, self.k_1)
            imgs = imgs.T
            # fig=plt.figure(figsize=(3,3))
            fig = plt.figure()
            fig.add_subplot(self.size_1[0], self.size_1[1], 1)
            plt.imshow(Image.open(img_path))
            plt.xticks([]), plt.yticks([])
            for i in range(2, self.k_1+2):
                fig.add_subplot(self.size_1[0], self.size_1[1], i)
                plt.imshow(np.rot90(imgs[i-2], -1))
                plt.xticks([]), plt.yticks([])

        def show_two_out(two_out):
            img = np.array(two_out)
            img = img.reshape(60, 60, self.k_2)
            img = img.T
            fig = plt.figure()
            fig.add_subplot(self.size_2[0], self.size_2[1], self.k_2+1)
            plt.imshow(Image.open(img_path))
            plt.xticks([]), plt.yticks([])
            for i in range(1, self.k_2+1):
                fig.add_subplot(self.size_2[0], self.size_2[1], i)
                plt.imshow(np.rot90(img[i-1], -1))
                plt.xticks([]), plt.yticks([])

            # plt.imshow(self.looking())
        layer_list = ['conv2d', 'conv2d_1', 'flatten', 'dense', 'dense_1']
        for root, dirs, files in os.walk(img_path):
            for path in files:
                img_path = root+path
                one_out = one_outputs(img_path=img_path)
                two_out = two_outputs(one_out)

                show_one_out(one_out)
                show_two_out(two_out)
        plt.show()

    # Please store the test pictures in the following directory
    # FindBeauty/testdata/
    def test(self, img_path: str):
        plt.draw()
        for root, dirs, files in os.walk(img_path):
            for path in files:
                img_path = root+path
                res = self.predict_result(img_path)
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
        self.model.fit(x_train, y_train, batch_size=8, epochs=3)
        self.model.save(model_path)

    def predict_result(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, dsize=(64, 64))
        img = img.reshape(1, 64, 64, 3)
        r = self.model(img)
        r = np.array(r)
        res = r[0][0]
        return res


def __main__():

    path = 'FindBeauty/models/model-test5'
    fb = FindBeauty()
    fb.creatMymodel(path)
    fb.model.summary()
    a=0
    if a:
        fb.fit(path)
    fb.visualization('FindBeauty/testdata/')
    # fb.test('FindBeauty/testdata/')


# hellow
if __name__ == '__main__':
    __main__()
