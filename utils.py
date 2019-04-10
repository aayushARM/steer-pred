
import cv2
import random
import numpy as np
import tensorflow.python.keras as tfk
import math
import matplotlib.pyplot as plt

def crop(image, top_offset, bottom_offset):
    if bottom_offset is None:
        return image[top_offset:, :, :]
    else:
        return image[top_offset:bottom_offset, :, :]


#Note: Change dims for different model.
def resize(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def flip(image, angle):
    image = cv2.flip(image, 1)
    angle = -1*angle
    return image, angle

def random_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    multiplier = 1.0 + 0.4*(random.random() - 0.5)
    image[:,:, 2] = image[:,:, 2] * multiplier
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


def random_translate(image, angle):
    rows, cols = image.shape[0:2]
    trans_x = 100 * (random.random()-0.5)
    trans_y = 10 * (random.random()-0.5)
    angle = angle + trans_x*0.004
    M = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    image = cv2.warpAffine(image, M, (cols, rows))
    return image, angle


def random_shadow(image):
    cols, rows = (image.shape[0], image.shape[1])

    top_y = np.random.random_sample() * rows
    bottom_y = np.random.random_sample() * rows
    bottom_y_right = bottom_y + np.random.random_sample() * (rows - bottom_y)
    top_y_right = top_y + np.random.random_sample() * (rows - top_y)
    if np.random.random_sample() <= 0.5:
        bottom_y_right = bottom_y - np.random.random_sample() * (bottom_y)
        top_y_right = top_y - np.random.random_sample() * (top_y)

    poly = np.asarray([[[top_y, 0], [bottom_y, cols], [bottom_y_right, cols], [top_y_right, 0]]], dtype=np.int32)

    mask_weight = np.random.uniform(0.6, 0.85)
    origin_weight = 1 - mask_weight

    mask = np.copy(image).astype(np.int32)
    cv2.fillPoly(mask, poly, (0, 0, 0))
    image = cv2.addWeighted(image.astype(np.int32), origin_weight, mask, mask_weight, 0).astype(np.uint8)
    return image

# Non-thread safe implementation, kept for reference, not used.
def create_batch(X, y, batch_size, is_train, img_width=200, img_height=66):
    X_batch = np.empty([batch_size, img_height, img_width, 3])
    y_batch = np.empty(batch_size)

    while True:
        indices = random.sample(range(len(y)), batch_size)
        for i in indices:
            image = cv2.imread(X[i])
            #image = X[i]
            angle = y[i]
            image = crop(image, 0, 0)
            image = resize(image, img_width, img_height)
            if is_train:
                if random.random() > 0.5:
                    image = random_brightness(image)
                if random.random() > 0.5:
                    image, angle = random_translate(image, angle)
                if random.random() > 0.5:
                    image = random_shadow(image)
            image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            X_batch[i] = image
            y_batch[i] = angle

        yield (X_batch, y_batch)


# Thread safe implementation, used by model.
class BatchGenerator(tfk.utils.Sequence):

    def __init__(self, X, y, batch_size, is_train, img_height, img_width):
        self.batch_size = batch_size
        self.is_train = is_train
        self.X = X
        self.y = y
        self.img_height = img_height
        self.img_width = img_width
        self.X_batch = np.empty([self.batch_size, self.img_height, self.img_width, 3])
        self.y_batch = np.empty(self.batch_size)
        self.num_batches = int(math.ceil(len(self.X) / float(self.batch_size)))
        self.real = False
    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        n = 0
        if idx == self.__len__() - 1:
            indices = list(range(idx*self.batch_size, len(self.X)))
            indices.extend(random.sample(range(len(self.X)), self.batch_size - len(indices)))
        else:
            indices = range(idx*self.batch_size, (idx + 1)*self.batch_size)

        for i in indices:
            image = cv2.imread(self.X[i])
            angle = self.y[i]
            image = crop(image, 200, None)
            #image = crop(image, 60, 25)
            image = resize(image, self.img_width, self.img_height)
            if self.is_train:
                if random.random() > 0.5:
                    image, angle = flip(image, angle)
                # if random.random() > 0.5:
                #     image = random_brightness(image)
                # if random.random() > 0.5:
                #     image, angle = random_translate(image, angle)
                # if random.random() > 0.5:
                #     image = random_shadow(image)
            image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            self.X_batch[n] = image
            self.y_batch[n] = angle
            n += 1
            if n == self.batch_size:
                break

        return self.X_batch, self.y_batch