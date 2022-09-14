import os
import cv2
import shutil
import utilities.config_mapper as ConfigMapper


def save_image(path, image, option=True):
    if option:
        cv2.imwrite(path, image)

def main():
    # region Configuration
    config = ConfigMapper.config_copy(
        ConfigMapper.get_config(root='./config')
    )
    default_option = config['default']
    save_debug = default_option['save']
    # 입력 설정 가져오기
    input_config = config['input']
    proc_config = config['processing']
    proc_config['raw_path'] = os.path.join(input_config['base_path'], proc_config['raw_path'])
    temp_path = os.path.join(input_config['base_path'],
                             config['preprocessing']['raw_path'],
                             config['preprocessing']['input_folder'])
    input_path = os.path.join(proc_config['raw_path'], proc_config['input_folder'])
    image_list = os.listdir(temp_path)
    # 패턴인식용 설정 가져오기
    edge_path = os.path.join(proc_config['raw_path'], proc_config['edge_folder'])
    blur_path = os.path.join(proc_config['raw_path'], proc_config['blur_folder'])
    norm_path = os.path.join(proc_config['raw_path'], proc_config['norm_folder'])

    # 폴더 생성
    if os.path.exists(proc_config['raw_path']):
        shutil.rmtree(proc_config['raw_path'])

    make_folder_list = [proc_config['raw_path'], input_path,
                        edge_path, blur_path, norm_path
                        ]

    for folder in make_folder_list:
        os.mkdir(folder)
    # endregion

    for image_path in image_list:
        full_path = os.path.join(temp_path, image_path)
        image_bgr = cv2.imread(full_path, cv2.IMREAD_COLOR)
        image_b, image_g, image_r = cv2.split(image_bgr)
        # img_blur = cv2.stylization(image_bgr, sigma_s=150, sigma_r=0.5)
        img_g_blur = cv2.GaussianBlur(image_b, (3, 3), sigmaX=0, sigmaY=0)
        img_b_blur = cv2.GaussianBlur(image_g, (3, 3), sigmaX=0, sigmaY=0)
        save_image(os.path.join(input_path, image_path), image_r, save_debug)
        image_edge = cv2.merge([img_b_blur, img_g_blur, image_r])
        # image_edge = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
        save_image(os.path.join(edge_path, image_path), image_edge, save_debug)

        # img_r_blur = cv2.medianBlur(image_r, 75)
        img_g_blur = cv2.medianBlur(image_g, 45)
        img_b_blur = cv2.medianBlur(image_b, 45)
        img_g_blur = cv2.add(img_g_blur, img_b_blur)
        save_image(os.path.join(blur_path, image_path), img_g_blur, save_debug)
        img_norm = cv2.normalize(img_b_blur, None, 0, 255, cv2.NORM_MINMAX)
        save_image(os.path.join(norm_path, image_path), img_norm, save_debug)

if __name__ == '__main__':
    main()


