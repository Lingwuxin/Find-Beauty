from pyexpat import model
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

    def visualization(self, img_path: str):
        def flip_horizontally(arr:np):
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
            imgs = imgs.reshape(62, 62, 8)
            imgs = imgs.T
            # fig=plt.figure(figsize=(3,3))
            fig=plt.figure()
            fig.add_subplot(3,3,1)
            plt.imshow(Image.open(img_path))
            plt.xticks([]), plt.yticks([])
            for i in range(2, 10):
                fig.add_subplot(3, 3, i)
                plt.imshow(np.rot90(imgs[i-2], -1))
                plt.xticks([]), plt.yticks([])
            
        def show_two_out(two_out):
            img = np.array(two_out)
            img = img.reshape(60, 60, 16)
            img = img.T
            fig=plt.figure()
            fig.add_subplot(5, 4, 17)
            plt.imshow(Image.open(img_path))
            plt.xticks([]), plt.yticks([])
            for i in range(1, 17):
                fig.add_subplot(5, 4, i)
                plt.imshow(np.rot90(img[i-1], -1))
                plt.xticks([]), plt.yticks([])
            
            # plt.imshow(self.looking())
        layer_list = ['conv2d', 'conv2d_1', 'flatten', 'dense', 'dense_1']
        for root, dirs, files in os.walk(img_path):
            for path in files:
                img_path = root+path
                one_out = one_outputs(img_path=img_path)
                two_out = two_outputs(one_out)
                # t_one = threading.Thread(target=show_one_out,args=(one_out,))
                # t_two = threading.Thread(target=show_two_out,args=(two_out,))
                # t_one.start()
                # t_two.start()
                
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
        self.model.fit(x_train, y_train, batch_size=16, epochs=4)
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

    path = 'FindBeauty/models/model-test4'
    fb = FindBeauty()
    fb.creatMymodel(path)
    fb.model.summary()
    fb.visualization('FindBeauty/testimg/')


# hellow
if __name__ == '__main__':
    __main__()
