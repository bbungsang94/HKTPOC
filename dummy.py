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
        else:
            return 0


if __name__ == '__main__':
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

    fitted_folder = os.path.join(mat_config['raw_path'], mat_config['fitted_folder'])
    detected_path = os.path.join(mat_config['raw_path'], mat_config['detected_folder'])
    pickle_list = os.listdir(temp_path)

    if os.path.exists(mat_config['raw_path']):
        shutil.rmtree(mat_config['raw_path'])

    make_folder_list = [mat_config['raw_path'],
                        fitted_folder, detected_path]
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
                    if block == '_':
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

    # endregion
    # region Matching
    for file_name in pickle_list:
        with open(os.path.join(temp_path, file_name), "rb") as fr:
            pack = pickle.load(fr)
        (image_bgr, info, _, _) = pack
        image_path = file_name.replace(".pickle", ".jpg")

        # Reference block 매칭 작업 진행
        for ref_index, blocks in enumerate(reference_patterns_info['blocks']):
            score = []
            matched_block = []
            pattern_name = reference_patterns_info['pattern_names'][ref_index]
            rotate = reference_patterns_info['rotate'][ref_index]
            # Reference block 마다 Pivot 을 정하여 Score 비교
            for pivot_index, pivot_block in enumerate(blocks):
                sub_matched = []
                pivot_rot = rotate[pivot_index]
                # Detected box 추출
                boxes = info[0]
                for i in range(boxes.shape[0]):
                    # Detected box 정보 추출
                    top, left, bottom, right = tuple(boxes[i])
                    height_min = int(min(top, bottom))
                    height_max = int(max(top, bottom))
                    width_min = int(min(left, right))
                    width_max = int(max(left, right))

                    height = height_max - height_min
                    width = width_max - width_min
                    if width > height:
                        rot = "H"
                        multiple = width / height
                    else:
                        rot = "V"
                        multiple = height / width

                    # 형상이 다르면 pivot의 조건이 안됨
                    if pivot_rot != rot:
                        continue
                    top, left, _, _ = pivot_block

        # end region
