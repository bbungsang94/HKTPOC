import copy
import os
import cv2
import pickle
import shutil
import numpy as np
import utilities.config_mapper as ConfigMapper
from utilities.helpers import box_iou2
from utilities.media_handler import *

image_handler = ImageManager()


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


def main():
    # region Configuration
    config = ConfigMapper.config_copy(
        ConfigMapper.get_config(root='./config')
    )
    print(config)
    default_option = config['default']
    save_debug = default_option['save']

    input_config = config['input']
    mat_config = config['matcher']
    mat_config['raw_path'] = os.path.join(input_config['base_path'], mat_config['raw_path'])
    temp_path = os.path.join(input_config['base_path'],
                             config['detector']['raw_path'],
                             config['detector']['anchor_folder'])

    fitted_path = os.path.join(mat_config['raw_path'], mat_config['fitted_folder'])
    detected_path = os.path.join(mat_config['raw_path'], mat_config['detected_folder'])
    pickle_list = os.listdir(temp_path)

    if os.path.exists(mat_config['raw_path']):
        shutil.rmtree(mat_config['raw_path'])

    make_folder_list = [mat_config['raw_path'],
                        fitted_path, detected_path]
    for folder in make_folder_list:
        os.mkdir(folder)
    # endregion
    # region Making reference patterns
    reference_patterns_info = {'scores': [], 'blocks': [], 'rotate': [], 'matched_blocks': [], 'pattern_names': []}

    for block_ratio, maps in mat_config['defined_patterns'].items():
        if block_ratio == 2:
            n_of_blocks = 6
        else:
            n_of_blocks = 8

        for pattern_name, value in maps.items():
            print(pattern_name)
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

    # endregion
    # region Matching
    for file_name in pickle_list:
        with open(os.path.join(temp_path, file_name), "rb") as fr:
            pack = pickle.load(fr)
        (image_bgr, info, raw_boxes, largest_anchor) = pack
        image_path = file_name.replace(".pickle", ".jpg")
        img_height, img_width, _ = image_bgr.shape
        # 모든 Block 0, 0으로 fitting 작업
        boxes = copy.deepcopy(info[0])
        if len(boxes) < 2:
            continue
        # 모든 Block 좌표를 [0, 1] 로 Normalizing
        boxes[:, [0, 2]] /= img_height
        boxes[:, [1, 3]] /= img_width

        # Reference block 매칭 작업 진행
        for ref_index, blocks in enumerate(reference_patterns_info['blocks']):
            reference_patterns_info['scores'][ref_index] = 0
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
                reference_patterns_info['scores'][ref_index] += max_score
            reference_patterns_info['matched_blocks'][ref_index] = copy.deepcopy(matched_block)

        # 제일 높은 스코어 추출
        max_pattern_score = max(reference_patterns_info['scores'])
        index = reference_patterns_info['scores'].index(max_pattern_score)
        matched_index = reference_patterns_info['matched_blocks'][index]
        matched_block = info[0][matched_index]
        # Draw Image
        if save_debug:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            # 최종 패턴 이미지
            matched_image = image_handler.draw_just_boxes(image_bgr, info[0], color_idx=0)
            matched_image = image_handler.draw_just_boxes(matched_image, matched_block, color_idx=2)
            image_handler.save_tensor(img=matched_image, path=os.path.join(detected_path, image_path))
            # 이미지 쉐잎에 맞는 패턴 이미지 추출
            draw_blocks = copy.deepcopy(reference_patterns_info['blocks'][index])
            draw_blocks[:, [0, 2]] *= img_height
            draw_blocks[:, [1, 3]] *= img_width
            fitted_image = image_handler.draw_just_boxes(matched_image, draw_blocks, color_idx=7)
            image_handler.save_tensor(img=fitted_image, path=os.path.join(fitted_path, image_path))
        # end region


if __name__ == '__main__':
    main()
