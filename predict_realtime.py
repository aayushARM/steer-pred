
import numpy as np
import pygame
import pandas as pd
import cv2
import utils
import main

pygame.init()
size = (640*2, 480*2)
pygame.display.set_caption("comma.ai data viewer")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)

camera_surface = pygame.surface.Surface((640, 480), 0, 24).convert()

# ***** get perspective transform for images *****
from skimage import transform as tf

rsrc = \
    [[43.45456230828867, 118.00743250075844],
     [104.5055617352614, 69.46865203761757],
     [114.86050156739812, 60.83953551083698],
     [129.74572757609468, 50.48459567870026],
     [132.98164627363735, 46.38576532847949],
     [301.0336906326895, 98.16046448916306],
     [238.25686790036065, 62.56535881619311],
     [227.2547443287154, 56.30924933427718],
     [209.13359962247614, 46.817221154818526],
     [203.9561297064078, 43.5813024572758]]
rdst = \
    [[10.822125594094452, 1.42189132706374],
     [21.177065426231174, 1.5297552836484982],
     [25.275895776451954, 1.42189132706374],
     [36.062291434927694, 1.6376192402332563],
     [40.376849698318004, 1.42189132706374],
     [11.900765159942026, -2.1376192402332563],
     [22.25570499207874, -2.1376192402332563],
     [26.785991168638553, -2.029755283648498],
     [37.033067044190524, -2.029755283648498],
     [41.67121717733509, -2.029755283648498]]

tform3_img = tf.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))


def perspective_tform(x, y):
    p1, p2 = tform3_img((x, y))[0]
    return p2, p1


# ***** functions to draw lines *****
def draw_pt(img, x, y, color, sz=1):
    row, col = perspective_tform(x, y)
    row, col = int(row)+200, int(col)+200
    if row >= 0 and row < img.shape[0] and \
            col >= 0 and col < img.shape[1]:
        img[row - sz:row + sz, col - sz:col + sz] = color


def draw_path(img, path_x, path_y, color):
    for x, y in zip(path_x, path_y):
        draw_pt(img, x, y, color)


# ***** functions to draw predicted path *****

def calc_curvature(v_ego, angle_steers, angle_offset=0):
    deg_to_rad = np.pi / 180.
    slip_fator = 0.0014  # slip factor obtained from real data
    steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
    wheel_base = 2.67  # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

    angle_steers_rad = (angle_steers - angle_offset) * deg_to_rad
    curvature = angle_steers_rad / (steer_ratio * wheel_base * (1. + slip_fator * v_ego ** 2))
    return curvature


def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=-1):
    # *** this function returns the lateral offset given the steering angle, speed and the lookahead distance
    curvature = calc_curvature(v_ego, angle_steers, angle_offset)

    # clip is to avoid arcsin NaNs due to too sharp turns
    y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999)) / 2.)
    return y_actual, curvature


def draw_path_on(img, speed_ms, angle_steers, color=(0, 0, 255)):
    path_x = np.arange(0., 50.1, 0.5)
    path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)
    draw_path(img, path_x, path_y, color)


if __name__ == "__main__":
    data = pd.read_csv('/media/aayush/Other/Udacity Data Real/CH2_002/output/out_angles.csv')
    img_height = 105
    img_width = 240
    model_name = ['nvidia', 'darknet53', ''][1]
    checkpoint_path = '/media/aayush/Other/Udacity Data Real/CH2_002/output/checkpoints_real/'+model_name+'/xxx'
    model = main.build_model(1, img_height, img_width, model_name)
    model.load_weights(checkpoint_path)

    img_paths = data['img']
    gt_angles = data['gt']
    pred_angles = data['preda']
    #pred_angles = 2 * ((pred_angles - np.min(pred_angles)) / (np.max(pred_angles) - np.min(pred_angles))) - 1
    speeds = data['speed']

    for i in range(len(img_paths)):
        image = cv2.imread(img_paths[i])
        image_prepro = utils.crop(image, 200, None)
        image_prepro = utils.resize(image_prepro, img_width, img_height)
        image_prepro = cv2.normalize(image_prepro, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image_prepro = np.expand_dims(image_prepro, 0)
        predicted_steer = model.predict(image_prepro, batch_size=1)[0][0]

        print('Predicted angle:', predicted_steer)
        angle_steer = gt_angles[i]
        speed = speeds[i]
        draw_path_on(image, speed, angle_steer * 100)
        draw_path_on(image, speed, predicted_steer * 100, (0, 255, 0))

        # draw on
        pygame.surfarray.blit_array(camera_surface, image.swapaxes(0, 1))
        camera_surface_2x = pygame.transform.scale2x(camera_surface)
        screen.blit(camera_surface_2x, (0, 0))
        pygame.display.flip()