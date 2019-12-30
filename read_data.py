from random import shuffle

import pandas as pd

import numpy as np

import pickle

import cv2

import os

from random import randint



class Reader(object):


    def __init__(self):


        self.train_csv = './training/training.csv'


        self.test_csv = './test/test/csv'


        self.cursor = 0


        self.names_path = './names.txt'


        self.data_path = './data.pkl'


        self.train_image_path = './train_image'


        self.size = 96


        if not os.path.exists(self.train_image_path):


            os.makedirs(self.train_image_path)


            self.data = self.pre_process()


        else:


            with open(self.data_path, 'rb') as f:


                self.data = pickle.load(f)


        print('There is {} in total data.'.format(len(self.data)))


        shuffle(self.data)


        with open(self.names_path, 'r') as f:


            self.names = f.read().splitlines()


        self.data_num = len(self.data)


        self.label_num = len(self.names)


    def pre_process(self):


        data = pd.read_csv(self.train_csv)

        data = data.dropna()


        cols = data.columns[:-1]


        data = data.to_dict()


        for key, value in data['Image'].items():


            data['Image'][key] = np.fromstring(value, sep=' ')


        data_names = list(data.keys())

        data_names.remove('Image')


        with open(self.names_path, 'w') as f:


            for value in data_names:

                f.writelines(value+'\n')


        labels = []


        for index in data['Image'].keys():


            label = {}


            image = data['Image'][index].reshape((96, 96))

            image_name = 'image_{}.jpg'.format(index)

            image_path = os.path.join(self.train_image_path, image_name)


            cv2.imwrite(image_path, image)


            label['image_path'] = image_path


            for point_name in data_names:

                label[point_name] = data[point_name][index]


            labels.append(label)


        with open(self.data_path, 'wb') as f:

            pickle.dump(labels, f)


        return labels


    def random_flip(self, image, points):


        if randint(0, 1):


            image = np.flip(image, axis=0)

            points[1::2] = 1 - points[1::2]


        return image, points


    def generate(self, batch_size=1):


        images = []

        points = []


        for _ in range(batch_size):


            path = self.data[self.cursor]['image_path']

            image = cv2.imread(path)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


            images.append(image)


            tmp = []

            for key in self.names:


                value = self.data[self.cursor][key]

                tmp.append(value)


            points.append(tmp)


            self.cursor += 1


            if self.cursor >= self.data_num:

                self.cursor = 0

                shuffle(self.data)


        images = np.array(images).reshape(

            (batch_size, self.size, self.size, 1))

        images = images - 127.5


        points = np.array(points)

        points = points/self.size


        # images, points = self.random_flip(images, points)


        return images, points



if __name__ == "__main__":


    import matplotlib.pyplot as plt


    reader = Reader()


    for _ in range(10):


        image, point = reader.generate(1)


        image = np.squeeze(image)

        point = np.squeeze(point)


        image = (image + 127.5).astype(np.int)

        point = (point * 96).astype(np.int)


        result = image.copy()


        y_axis = point[1::2]

        x_axis = point[::2]


        color = (0, 0, 255)


        for y, x in zip(y_axis, x_axis):


            cv2.circle(result, (x, y), 1, color)


        plt.imshow(result, cmap='gray')

        plt.show()