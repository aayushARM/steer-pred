import tensorflow.python.keras as tfk

image_height = 105
image_width = 240

def build_nvidia(img_height, img_width):
    model = tfk.Sequential()
    model.add(tfk.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2),
                                input_shape=(img_height, img_width, 3), kernel_initializer='he_normal'))
    model.add(tfk.layers.BatchNormalization())
    model.add(tfk.layers.Activation('relu'))

    model.add(tfk.layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), kernel_initializer='he_normal'))
    model.add(tfk.layers.BatchNormalization())
    model.add(tfk.layers.Activation('relu'))

    model.add(tfk.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), kernel_initializer='he_normal'))
    model.add(tfk.layers.BatchNormalization())
    model.add(tfk.layers.Activation('relu'))

    model.add(tfk.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='he_normal'))
    model.add(tfk.layers.BatchNormalization())
    model.add(tfk.layers.Activation('relu'))

    model.add(tfk.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='he_normal'))
    model.add(tfk.layers.BatchNormalization())
    model.add(tfk.layers.Activation('relu'))

    model.add(tfk.layers.Flatten())

    model.add(tfk.layers.Dense(1000, kernel_initializer='he_normal'))
    model.add(tfk.layers.BatchNormalization())
    model.add(tfk.layers.Activation('relu'))

    model.add(tfk.layers.Dense(100, kernel_initializer='he_normal'))
    model.add(tfk.layers.BatchNormalization())
    model.add(tfk.layers.Activation('relu'))

    model.add(tfk.layers.Dense(50, kernel_initializer='he_normal'))
    model.add(tfk.layers.BatchNormalization())
    model.add(tfk.layers.Activation('relu'))

    model.add(tfk.layers.Dense(10, kernel_initializer='he_normal'))
    model.add(tfk.layers.BatchNormalization())
    model.add(tfk.layers.Activation('relu'))

    model.add(tfk.layers.Dense(1))

    return model

if __name__ == '__main__':
    model = build_nvidia(image_height, image_width)
    print(model.summary())