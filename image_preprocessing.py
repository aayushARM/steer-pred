
import cv2
import pandas as pd
import numpy as np
import glob


def flip_combine(path):
    new_img_path = '/'.join(path.split('/')[0:-1]) + '/Images/'
    data = pd.read_csv(path + '/driving_log.csv', header=None)
    data.columns = ['center', 'left', 'right', 'angle', 'throttle', 'unk', 'speed']
    data_frames_list = []

    for i in range(len(data)):
        imgc = cv2.imread(data.loc[i, 'center'])
        imgl = cv2.imread(data.loc[i, 'left'])
        imgr = cv2.imread(data.loc[i, 'right'])
        angle = data.loc[i, 'angle']
        throttle = data.loc[i, 'throttle']

        new_pathc = new_img_path + data.loc[i, 'center'].split('/')[-1]
        new_pathl = new_img_path + data.loc[i, 'left'].split('/')[-1]
        new_pathr = new_img_path + data.loc[i, 'right'].split('/')[-1]

        cv2.imwrite(new_pathc, imgc, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(new_pathl, imgl, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(new_pathr, imgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        angle_offset = 0.25
        data_frames_list.append(pd.DataFrame({'img': new_pathc, 'angle': angle, 'throttle': throttle}, index=[0]))
        data_frames_list.append(pd.DataFrame({'img': new_pathl, 'angle': np.clip(angle + angle_offset, -1, 1), 'throttle': throttle}, index=[0]))
        data_frames_list.append(pd.DataFrame({'img': new_pathr, 'angle': np.clip(angle - angle_offset, -1, 1), 'throttle': throttle}, index=[0]))

        imgc_flip = cv2.flip(imgc, 1)
        imgl_flip = cv2.flip(imgl, 1)
        imgr_flip = cv2.flip(imgr, 1)

        flipc_name = new_pathc[0:-4] + '_flip.jpg'
        flipl_name = new_pathl[0:-4] + '_flip.jpg'
        flipr_name = new_pathr[0:-4] + '_flip.jpg'

        cv2.imwrite(flipc_name, imgc_flip, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(flipl_name, imgl_flip, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(flipr_name, imgr_flip, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        data_frames_list.append(pd.DataFrame({'img': flipc_name, 'angle': -1*angle, 'throttle': throttle}, index=[0]))
        data_frames_list.append(pd.DataFrame({'img': flipl_name, 'angle': np.clip(-1*angle + angle_offset, -1, 1), 'throttle': throttle}, index=[0]))
        data_frames_list.append(pd.DataFrame({'img': flipr_name, 'angle': np.clip(-1*angle - angle_offset, -1, 1), 'throttle': throttle}, index=[0]))

    data_frame = pd.concat(data_frames_list, ignore_index=True)

    print(data_frame.shape)
    return data_frame

def create_mega_csv():
    paths = glob.glob('/media/aayush/Other/beta_simulator_linux/recording*')
    df_mega_list = []
    for path in paths:
        df = flip_combine(path)
        df_mega_list.append(df)
    df_mega = pd.concat(df_mega_list)
    print(df_mega.shape)
    with open('/media/aayush/Other/beta_simulator_linux/driving_log_mega.csv', 'w') as mega_csv:
        df_mega.to_csv(mega_csv, index=False)
    return df_mega