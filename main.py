
import tensorflow.python.keras as tfk
import utils
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import model_nvidia
import model_darknet


def build_model(batch_size, img_height, img_width, name):

    if name == 'nvidia':
        return model_nvidia.build_nvidia(img_height, img_width)
    elif name == 'darknet53':
        return model_darknet53.build_darknet(img_height, img_width)


def main():
    which = ['simulator', 'real'][1] #switch
    model_name = ['nvidia', 'darknet53', ''][1]

    checkpoint_path, data, n_train_samples = None, None, None
    if which == 'simulator':
        # n_total_samples = 34386
        n_train_samples = 27510  # 6876 test
        checkpoint_path = '/media/aayush/Other/beta_simulator_linux/checkpoints/weights.{epoch:02d}-{val_loss:.2f}'
        data = pd.read_csv('/media/aayush/Other/beta_simulator_linux/driving_log_mega.csv')
    elif which == 'real':
        # n_total_samples = 28382
        n_train_samples = 22706 #todo
        checkpoint_path = '/media/aayush/Other/Udacity Data Real/CH2_002/output/checkpoints_real/' + model_name \
                          + '/weights.{epoch:02d}-{val_loss:.2f}'
        data = pd.read_csv('/media/aayush/Other/Udacity Data Real/CH2_002/output/filtered_only_center.csv')

    batch_size = 50
    image_height = 105 #140 #66
    image_width = 240 #320 #200

    #HPs:
    learning_rate = 0.001 #todo: increased!, last was 0.0001
    n_epochs = 50

    image_paths = data['img'].values
    angles = data['angle'].values
    angles = 2*((angles - np.min(angles))/(np.max(angles) - np.min(angles))) - 1

    X_train = image_paths[0: n_train_samples]
    y_train = angles[0: n_train_samples]

    X_test = image_paths[n_train_samples:]
    y_test = angles[n_train_samples:]

    model = build_model(batch_size, image_height, image_width, model_name)
    model.summary()
    model.compile(optimizer=tfk.optimizers.Adam(lr=learning_rate), loss='mean_squared_error',
                  metrics=['accuracy'])

    train_batch_generator = utils.BatchGenerator(X_train, y_train, batch_size, True, image_height, image_width)

    # bx, by = train_batch_generator.__getitem__(22)
    # img, angle = bx[0], by[0]
    # img = img.astype('uint8')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # print(angle)
    # plt.show()

    val_batch_generator = utils.BatchGenerator(X_test, y_test, batch_size, False, image_height, image_width)

    ckpt_callback = tfk.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1, monitor='val_loss',
                                                  save_best_only=True, mode='min')

    tensorboard_callback = tfk.callbacks.TensorBoard(log_dir='/media/aayush/Other/Udacity Data Real/CH2_002/output/'+model_name+'_logs',
                                                                     histogram_freq=0, batch_size=50, write_graph=True,
                                                                     write_grads=True, update_freq='batch')
    history = model.fit_generator(generator=train_batch_generator, epochs=n_epochs, verbose=2,
                                  validation_data=val_batch_generator, validation_freq=1, workers=8,
                                  use_multiprocessing=True, shuffle=True, callbacks=[ckpt_callback, tensorboard_callback])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
