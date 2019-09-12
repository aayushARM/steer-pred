# Modified version of DarkNet from YOLOv3(https://arxiv.org/abs/1804.02767), the total no. of trainable parameters(see model.summary())
# have been kept approximately same as NVIDIA PilotNet architecture for fair comparison on the same dataset.

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense
from tensorflow.python.keras.layers import add, Activation, BatchNormalization
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.regularizers import l2

image_height = 105
image_width = 240

def conv2d_unit(x, filters, kernels, strides=1):
    
    x = Conv2D(filters, kernels,
               padding='same',
               strides=strides,
               activation='linear',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x


def residual_block(inputs, filters):
    
    x = conv2d_unit(inputs, filters, (1, 1))
    x = conv2d_unit(x, 2 * filters, (3, 3))
    x = add([inputs, x])
    x = Activation('linear')(x)

    return x


def stack_residual_block(inputs, filters, n):
    
    x = residual_block(inputs, filters)

    for i in range(n - 1):
        x = residual_block(x, filters)

    return x


def darknet_base(inputs):

    x = conv2d_unit(inputs, 32, (3, 3))

    x = conv2d_unit(x, 64, (3, 3), strides=2)
    x = stack_residual_block(x, 32, n=1)

    x = conv2d_unit(x, 128, (3, 3), strides=2)
    x = stack_residual_block(x, 64, n=2)

    x = conv2d_unit(x, 256, (3, 3), strides=2)
    x = stack_residual_block(x, 128, n=8)

    x = conv2d_unit(x, 512, (3, 3), strides=2)
    x = stack_residual_block(x, 256, n=8)

    x = conv2d_unit(x, 1024, (3, 3), strides=2)
    x = stack_residual_block(x, 512, n=4)

    return x


def build_darknet(img_height, img_width):
    
    inputs = Input(shape=(img_height, img_width, 3))
    x = darknet_base(inputs)

    x = GlobalAveragePooling2D()(x)


    x = Dense(1000, activation='linear')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)

    x = Dense(100, activation='linear')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)

    x = Dense(10, activation='linear')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)

    x = Dense(1, activation='linear')(x)

    model = Model(inputs, x)
    return model


if __name__ == '__main__':
    model = build_darknet(image_height, image_width)
    print(model.summary())
