import os
import cv2
import shutil
import utilities.config_mapper as ConfigMapper

if __name__ == '__main__':
    # region Configuration
    # 설정 가져오기
    config = ConfigMapper.config_copy(
        ConfigMapper.get_config(root='./config')
    )
    print(config)
    # 이미지셋 리스트 가져오기
    input_config = config['input']
    image_list = os.listdir(input_config['path'])
    if os.path.exists(input_config['base_path']):
        shutil.rmtree(input_config['base_path'])

    # 전처리용 폴더 생성
    preproc_config = config['preprocessing']
    preproc_config['raw_path'] = os.path.join(input_config['base_path'], preproc_config['raw_path'])
    lab_path = os.path.join(preproc_config['raw_path'], preproc_config['lab_folder'])
    l_path = os.path.join(lab_path, 'L')
    a_path = os.path.join(lab_path, 'A')
    b_path = os.path.join(lab_path, 'B')
    lab_path_set = [l_path, a_path, b_path]

    # 전처리용 필더 폴더 생성
    mf_path = os.path.join(preproc_config['raw_path'], preproc_config['mf_folder'])
    bf_path = os.path.join(preproc_config['raw_path'], preproc_config['bf_folder'])
    comp_path = os.path.join(preproc_config['raw_path'], preproc_config['comp_folder'])

    # 처리용 폴더 생성
    proc_config = config['Processing']
    proc_config['raw_path'] = os.path.join(input_config['base_path'], proc_config['raw_path'])
    proc_input_path = os.path.join(proc_config['raw_path'], proc_config['input_folder'])
    red_path = os.path.join(proc_config['raw_path'], proc_config['red_folder'])

    make_folder_list = [input_config['base_path'], preproc_config['raw_path'], proc_config['raw_path'],
                        lab_path, l_path, a_path, b_path,
                        mf_path, bf_path, comp_path,
                        proc_input_path, red_path
                        ]

    for folder in make_folder_list:
        os.mkdir(folder)
    # endregion

    # 이미지 불러오기
    for image_path in image_list:
        # region Preprocessing
        full_path = os.path.join(input_config['path'], image_path)
        image_bgr = cv2.imread(full_path, cv2.IMREAD_COLOR)

        # light removal 단계
        # 1) RGB to LAB
        image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(image_lab)
        lab_mono_set = (l_channel, a_channel, b_channel)
        for channel, element_path in enumerate(lab_path_set):
            cv2.imwrite(os.path.join(element_path, image_path), lab_mono_set[channel])
        # 2) LAB with median 25 to 100 and bilateral
        image_mf = cv2.medianBlur(l_channel, 75)
        cv2.imwrite(os.path.join(mf_path, image_path), image_mf)
        image_bf = cv2.bilateralFilter(l_channel, 9, 75, 75)
        cv2.imwrite(os.path.join(bf_path, image_path), image_mf)
        inverted_image = cv2.bitwise_not(image_mf)

        # addweighted가 아닌 add를 하면 제일 높은 블럭을 찾을 수 있다.
        image_composite = cv2.addWeighted(l_channel, 0.5, inverted_image, 0.5, 0)
        cv2.imwrite(os.path.join(comp_path, image_path), image_composite)

        # Light removal 완료, 밝은 난반사 없애야 함
        test_image = cv2.merge([image_composite, a_channel, b_channel])
        image_bgr = cv2.cvtColor(test_image, cv2.COLOR_LAB2BGR)
        cv2.imwrite(os.path.join(proc_input_path, image_path), image_bgr)
        _, _, r_channel = cv2.split(image_bgr)
        cv2.imwrite(os.path.join(red_path, image_path), r_channel)
        # endregion

