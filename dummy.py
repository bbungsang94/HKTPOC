import os
import cv2
import shutil
import utilities.config_mapper as ConfigMapper


def save_image(path, image, option=True):
    if option:
        cv2.imwrite(path, image)


config = ConfigMapper.config_copy(
    ConfigMapper.get_config(root='./config')
)
input_config = config['input']
image_list = os.listdir(input_config['path'])

for image_path in image_list:
    full_path = os.path.join(input_config['path'], image_path)
    image_bgr = cv2.imread(full_path, cv2.IMREAD_COLOR)
    (h, w, c) = image_bgr.shape
    if h > w:
        image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)

    save_image(os.path.join(input_config['path'], image_path), image_bgr)
