
import pandas as pd
import cv2
import utils
import main
import numpy as np

def predict():
    data = pd.read_csv('/media/aayush/Other/Udacity Data Real/CH2_002/output/filtered_only_center.csv')
    model_name = ['nvidia', 'darknet53', ''][1]
    checkpoint_path = '/media/aayush/Other/Udacity Data Real/CH2_002/output/checkpoints_real/'+model_name+'weights.47-0.00'
    n_train_samples = 22706
    image_paths = data['img'].values
    angles = data['angle'].values
    angles = 2 * ((angles - np.min(angles)) / (np.max(angles) - np.min(angles))) - 1
    speeds = data['speed'].values
    X_test = image_paths[n_train_samples:]
    y_test = angles[n_train_samples:]
    num_batches = 66
    batch_size = 86

    img_height = 105
    img_width = 240
    model = main.build_model(batch_size, img_height, img_width)
    model.load_weights(checkpoint_path)
    dframe_list = []

    for batch_no in range(num_batches):

        #preprocessing
        images = [cv2.imread(path) for path in X_test[batch_no*batch_size: (batch_no+1)*batch_size]]
        images = [utils.crop(image, 200, None) for image in images]
        images = [utils.resize(image, img_width, img_height) for image in images]
        images = [cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) for image in images]

        images = np.stack(images, 0)
        gt_angles = y_test[batch_no*batch_size: (batch_no+1)*batch_size]
        speed_batch = speeds[batch_no*batch_size: (batch_no+1)*batch_size]
        pred_angles = model.predict(images, batch_size=batch_size)
        pred_angles = np.squeeze(pred_angles)
        df = pd.DataFrame({'img': X_test[batch_no*batch_size: (batch_no+1)*batch_size], 'gt': gt_angles, 'preda': pred_angles, 'speed': speed_batch})
        dframe_list.append(df)

    dframes = pd.concat(dframe_list, ignore_index=True)



    dframes.to_csv('/media/aayush/Other/Udacity Data Real/CH2_002/output/out_angles.csv')

predict()