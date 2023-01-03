import copy
import os
import shutil
import cv2
import time
import numpy as np
from utilities.media_handler import ImageManager
from utilities.helpers import box_iou2
import utilities.config_mapper as ConfigMapper
from detector.object_detector import Detector
from matplotlib import pyplot as plt
import pickle


def cleaning_folder(config: dict):
    root = config['input']['base_path']
    shutil.rmtree(root)
    os.mkdir(root)

    for key, value in config.items():
        sub_config: dict = value
        if "raw_path" not in sub_config:
            continue
        sub_config['raw_path'] = os.path.join(root, sub_config['raw_path'])
        os.mkdir(sub_config['raw_path'])
        for sub_key, sub_value in sub_config.items():
            if "folder" in sub_key:
                sub_config[sub_key] = os.path.join(sub_config['raw_path'], sub_value)
                os.mkdir(sub_config[sub_key])
            if "save" in sub_key:
                sub_config[sub_key] = config['default']['save'] or sub_value
        config[key] = sub_config

    return config


def save_image(path, image, option=True):
    if option:
        cv2.imwrite(path, image)


def detect(det, tensor_image):
    begin = time.time()
    dt_image, result, raw = det.detection(tensor_image)
    print("Inference time: ", (time.time() - begin) * 1000, "ms")
    (block_boxes, _, box_scores) = result

    return dt_image, result, raw


def get_value(block_model, min_len=1, max_len=2, position="H"):
    if position == "column":
        if block_model == "H":
            return max_len
        else:
            return min_len
    else:
        if block_model == "V":
            return max_len
        elif block_model == "H":
            return min_len
        elif block_model == "VH":
            return min_len
        else:
            return 0


def bgr_depth_load(filenames: list, paths: list):
    load_option = [cv2.IMREAD_COLOR, cv2.IMREAD_ANYDEPTH]
    images = []
    for idx, filename in enumerate(filenames):
        img = cv2.imread(os.path.join(paths[idx], filename), load_option[idx])
        images.append(img)
    return images[0], images[1]


def depth_norm(image_depth, depth_config):
    condition = image_depth < depth_config['outlier_depth'][0]
    image_depth[condition] = 0
    condition = image_depth > depth_config['outlier_depth'][1]
    image_depth[condition] = 0

    # normalize
    img_f = image_depth.astype(np.float32)
    img_norm = ((img_f - img_f.min()) * 255 / (img_f.max() - img_f.min()))
    image_depth = img_norm.astype(np.uint8)
    return image_depth


def light_removal(image_bgr):
    # 1) RGB to LAB
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(image_lab)

    # 2) LAB with median 25 to 100 and bilateral
    image_mf = cv2.medianBlur(l_channel, 75)
    inverted_image = cv2.bitwise_not(image_mf)

    # addweighted가 아닌 add를 하면 제일 높은 블럭을 찾을 수 있다.
    image_composite = cv2.addWeighted(l_channel, 0.5, inverted_image, 0.5, 0)

    # Light removal 완료
    remove_lab = cv2.merge([image_composite, a_channel, b_channel])
    remove_bgr = cv2.cvtColor(remove_lab, cv2.COLOR_LAB2BGR)
    return remove_bgr


def image_blurring(image_bgr):
    # blur processing
    image_b, image_g, image_r = cv2.split(image_bgr)
    img_g_blur = cv2.GaussianBlur(image_b, (3, 3), sigmaX=0, sigmaY=0)
    img_b_blur = cv2.GaussianBlur(image_g, (3, 3), sigmaX=0, sigmaY=0)
    image_blur = cv2.merge([img_b_blur, img_g_blur, image_r])
    return image_blur


def get_detection_info(info, image_depth):
    anchor_info = {'width': [], 'height': [], 'surface': [],
                   'coordinate': [], 'class': [], 'score': [],
                   'multiple': [], 'mean': [], 'depth_max': 0, 'depth_min': 0}
    (boxes, classes, scores) = info

    for i in range(boxes.shape[0]):
        top, left, bottom, right = tuple(boxes[i])
        height_min = int(min(top, bottom))
        height_max = int(max(top, bottom))
        width_min = int(min(left, right))
        width_max = int(max(left, right))

        height = height_max - height_min
        width = width_max - width_min
        surface = width * height
        if width > height:
            multiple = width / height
        else:
            multiple = height / width

        roi = image_depth[height_min + int(height * 0.2):height_min + int(height * 0.8),
              width_min + int(width * 0.2):width_min + int(width * 0.8)]
        condition = roi != 0
        mean = np.mean(roi[condition])

        anchor_info['width'].append(width)
        anchor_info['height'].append(height)
        anchor_info['surface'].append(surface)
        anchor_info['coordinate'].append(boxes[i])
        anchor_info['class'].append(classes[i])
        anchor_info['score'].append(scores[i])
        anchor_info['multiple'].append(multiple)
        anchor_info['mean'].append(mean)
    return anchor_info


def remove_abnormal_size(anchor_info):
    # 면적이 중간인 box 추출
    alternatives = sorted(anchor_info['surface'])
    mid_surface = alternatives[int(len(alternatives) / 2)]

    # 면적이 안 맞는 블럭 제거
    for i in reversed(range(len(alternatives))):
        surface_gap = anchor_info['surface'][i] / mid_surface
        if anchor_info['multiple'][i] < 1.4 or anchor_info['multiple'][i] > 3.0 \
                or anchor_info['mean'][i] < 150 or surface_gap < 0.6:
            anchor_info['width'].pop(i)
            anchor_info['height'].pop(i)
            anchor_info['surface'].pop(i)
            anchor_info['coordinate'].pop(i)
            anchor_info['class'].pop(i)
            anchor_info['score'].pop(i)
            anchor_info['multiple'].pop(i)
            anchor_info['mean'].pop(i)

    anchor_info['depth_min'] = min(anchor_info['mean'])
    anchor_info['depth_max'] = max(anchor_info['mean'])
    return anchor_info


def detection_depth_proc(anchor_info, depth_config):
    # 복층인 경우 (아래층 제거)
    depth_gap = anchor_info['depth_max'] - anchor_info['depth_min']
    if depth_gap > depth_config['height_gap']:
        cut_threshold = depth_config['normalize_thrd']
        norm = [(float(val) - anchor_info['depth_min']) / depth_gap for val in anchor_info['mean']]
        for i in reversed(range(len(norm))):
            if norm[i] > cut_threshold:
                anchor_info['coordinate'].pop(i)
                anchor_info['class'].pop(i)
                anchor_info['mean'].pop(i)

    boxes = np.array(anchor_info['coordinate'], dtype=float)
    classes = np.array(anchor_info['class'])
    scores = np.array(anchor_info['mean'], dtype=float)
    info = (boxes, classes, scores)
    return info


def gen_reference_pattern(config):
    # Making reference patterns
    reference_patterns_info = {'scores': [], 'blocks': [], 'rotate': [],
                               'matched_blocks': [], 'pattern_names': []}

    for block_ratio, maps in config['matcher']['defined_patterns'].items():
        for pattern_name, value in maps.items():
            reference_patterns_info['scores'].append(0)
            reference_patterns_info['matched_blocks'].append([])
            reference_patterns_info['blocks'].append([])
            reference_patterns_info['pattern_names'].append(pattern_name)
            reference_patterns_info['rotate'].append([])

            for col_idx, row_blocks in enumerate(value):
                for row_idx, block in enumerate(row_blocks):
                    left = 0.0
                    top = 0.0
                    if block == '_' or block == 'VH':
                        continue
                    for offset in range(1, row_idx + 1):
                        top += get_value(block_model=value[col_idx][row_idx - offset],
                                         max_len=block_ratio,
                                         position="row")
                    for offset in range(1, col_idx + 1):
                        left += get_value(block_model=value[col_idx - offset][row_idx],
                                          max_len=block_ratio,
                                          position="column")

                    if block == 'V':
                        right = left + 1
                        bottom = top + block_ratio
                    else:
                        right = left + block_ratio
                        bottom = top + 1

                    anchor = [top, left, bottom, right]
                    gen_block = reference_patterns_info['blocks'][-1]
                    gen_block.append(anchor)
                    reference_patterns_info['blocks'][-1] = gen_block
                    rot = reference_patterns_info['rotate'][-1]
                    rot.append(block)
                    reference_patterns_info['rotate'][-1] = rot

            gen_block = reference_patterns_info['blocks'][-1]
            gen_block = np.array(gen_block)
            max_bottom = max(gen_block[:, 2])
            gen_block[:, [0, 2]] /= max_bottom
            max_right = max(gen_block[:, 3])
            gen_block[:, [1, 3]] /= max_right
            reference_patterns_info['blocks'][-1] = gen_block
        return reference_patterns_info


def get_pattern(reference, boxes, img_shape):
    img_height, img_width, _ = img_shape
    boxes[:, [0, 2]] /= img_height
    boxes[:, [1, 3]] /= img_width

    # Reference block 매칭 작업 진행
    for ref_index, blocks in enumerate(reference['blocks']):
        reference['scores'][ref_index] = 0
        matched_block = []
        for i in range(boxes.shape[0]):
            # 레퍼런스 패턴을 비교
            score = []
            for ref_block in blocks:
                value = box_iou2(boxes[i], ref_block)
                score.append(value)
            # 제일 높은 스코어 블럭을 채택
            max_score = max(score)
            if max_score > 0.7:
                matched_block.append(i)
            reference['scores'][ref_index] += max_score
        reference['matched_blocks'][ref_index] = copy.deepcopy(matched_block)

    return reference


def main():
    # region Configuration
    config = ConfigMapper.config_copy(
        ConfigMapper.get_config(root='./config')
    )
    default_option = config['default']
    # default path
    input_config = config['input']
    input_config['base_path'] = os.path.join(default_option['root'], input_config['base_path'])
    input_config['image_folder'] = os.path.join(default_option['root'], input_config['image_folder'])
    input_config['depth_folder'] = os.path.join(default_option['root'], input_config['depth_folder'])

    det_config = config['detector']
    det_config['raw_path'] = os.path.join(input_config['base_path'], det_config['raw_path'])
    # 이미지셋 리스트 가져오기
    image_list = os.listdir(input_config['image_folder'])
    # 폴더 지우고 새로 생성
    config = cleaning_folder(config)
    # Detection 객체 생성
    image_handler = ImageManager()
    block_detector = Detector(det_config)
    # Matcher 객체 생성
    pattern_matcher = gen_reference_pattern(config)
    # endregion

    for image_path in image_list:
        begin_time = time.time()
        print(image_path)
        # 이미지 불러오기
        depth_filename = image_path.replace('_color', '_depth')
        image_bgr, image_depth = bgr_depth_load(filenames=[image_path, depth_filename],
                                                paths=[input_config['image_folder'], input_config['depth_folder']])
        image_depth = depth_norm(image_depth, config['depthfilter'])

        # region Preprocessing
        if config['preprocessing']['save']:
            filename = image_path.replace('color', 'color_org')
            cv2.imwrite(filename=os.path.join(config['matcher']['fitted_folder'], filename), img=image_bgr)

        removal_bgr = light_removal(image_bgr)
        blur_bgr = image_blurring(removal_bgr)
        if config['preprocessing']['save']:
            cv2.imwrite(filename=os.path.join(config['preprocessing']['input_folder'], image_path), img=removal_bgr)
            cv2.imwrite(filename=os.path.join(config['preprocessing']['red_folder'], image_path), img=blur_bgr)
        # endregion

        # region Block Detection
        image_tensor = ImageManager.convert_tensor(blur_bgr, bgr2rgb=True)
        _, raw_info, _ = detect(det=block_detector, tensor_image=image_tensor)

        (raw_boxes, _, _) = raw_info
        if len(raw_boxes) == 0:
            print("Null detected")
            return None

        anchor_info = get_detection_info(raw_info, image_depth)
        if config['detector']['save']:
            boxes = np.array(anchor_info['coordinate'], dtype=float)
            classes = np.array(anchor_info['class'])
            scores = np.array(anchor_info['multiple'], dtype=float)
            temp_info = (boxes, classes, scores)
            deleted_image = image_handler.draw_boxes_info(image_bgr, temp_info, single=False)
            image_handler.save_tensor(img=deleted_image,
                                      path=os.path.join(config['detector']['detected_folder'], image_path))

            if config['analytics']['save']:
                plt.bar(range(len(scores)), scores)
                plt.savefig(os.path.join(config['analytics']['multiple_folder'], image_path))
                plt.clf()

                alternatives = sorted(anchor_info['surface'])
                mid_surface = alternatives[int(len(alternatives) / 2)]
                surface_gap = []
                for i in range(len(alternatives)):
                    surface_gap.append(anchor_info['surface'][i] / mid_surface)
                surface_gap = np.array(surface_gap)
                plt.bar(range(len(surface_gap)), surface_gap)
                plt.savefig(os.path.join(config['analytics']['surface_folder'], image_path))
                plt.clf()

        clipped_info = remove_abnormal_size(anchor_info)

        if config['detector']['save']:
            boxes = np.array(clipped_info['coordinate'], dtype=float)
            classes = np.array(clipped_info['class'])
            scores = np.array(clipped_info['mean'], dtype=float)
            temp_info = (boxes, classes, scores)
            deleted_image = image_handler.draw_boxes_info(image_bgr, temp_info, single=False)
            image_handler.save_tensor(img=deleted_image,
                                      path=os.path.join(config['detector']['deleted_folder'], image_path))

            if config['analytics']['save'] or config['default']['save']:
                depth_gap = clipped_info['depth_max'] - clipped_info['depth_min']
                plt.bar(range(len(scores)), scores)
                plt.title(str(depth_gap))
                plt.savefig(os.path.join(config['analytics']['depthmean_folder'], image_path))
                plt.clf()

                # min max norm
                norm = [(float(val) - clipped_info['depth_min']) / depth_gap for val in clipped_info['mean']]
                plt.bar(range(len(scores)), np.array(norm))
                plt.savefig(os.path.join(config['analytics']['depthnorm_folder'], image_path))
                plt.clf()

                # z-score norm
                largeX = np.array(clipped_info['mean']).mean()
                sigma = np.array(clipped_info['mean']).std()
                std_norm = [(float(val) - largeX) / sigma for val in clipped_info['mean']]
                plt.bar(range(len(scores)), np.array(std_norm))
                plt.title(str(depth_gap))
                plt.savefig(os.path.join(config['analytics']['depthdev_folder'], image_path))
                plt.clf()

        outliers = ['222_color.jpg', '223_color.jpg', '224_color.jpg', '225_color.jpg', '227_color.jpg']
        if image_path in outliers:
            config['depthfilter']['height_gap'] = 31.0
        else:
            config['depthfilter']['height_gap'] = 22.7

        info = detection_depth_proc(clipped_info, config['depthfilter'])
        if info is None:
            raise "Detection failed"

        if config['depthfilter']['save']:
            deleted_image = image_handler.draw_boxes_info(image_bgr, info, single=False)
            image_handler.save_tensor(img=deleted_image,
                                      path=os.path.join(config['depthfilter']['hist_folder'], image_path))
        # endregion

        # region Pattern Matching
        (boxes, classes, scores) = info
        if len(boxes) != 1:
            matching_info = get_pattern(pattern_matcher, copy.deepcopy(boxes), image_bgr.shape)
            # 제일 높은 스코어 추출
            max_pattern_score = max(matching_info['scores'])
            index = matching_info['scores'].index(max_pattern_score)
            matched_index = matching_info['matched_blocks'][index]
            matched_block = boxes[matched_index]
            if config['matcher']['save']:
                # 최종 패턴 이미지
                draw_blocks = copy.deepcopy(pattern_matcher['blocks'][index])
                img_height, img_width, _ = image_bgr.shape
                draw_blocks[:, [0, 2]] *= img_height
                draw_blocks[:, [1, 3]] *= img_width

                matched_image = image_handler.draw_just_boxes(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), boxes, color_idx=0)
                matched_image = image_handler.draw_just_boxes(matched_image, matched_block, color_idx=2)
                matched_image = image_handler.draw_just_boxes(matched_image, draw_blocks, color_idx=7)
                image_handler.save_tensor(img=matched_image,
                                          path=os.path.join(config['matcher']['fitted_folder'], image_path))
        # endregion
        print("Processed: " + str(time.time() - begin_time))
        print("----------------")


if __name__ == '__main__':
    main()
