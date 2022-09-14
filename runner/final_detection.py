import copy
import shutil
import time
from utilities.helpers import box_iou2
import utilities.config_mapper as ConfigMapper
from detector.object_detector import Detector
from utilities.media_handler import *
import pickle


def cleaning_folder(config: dict):
    root = config['input']['base_path']
    shutil.rmtree(root)
    os.mkdir(root)

    for key, value in config.items():
        sub_config: dict = value
        if "raw_path" not in sub_config:
            continue
        sub_root = os.path.join(root, sub_config['raw_path'])
        os.mkdir(sub_root)
        for sub_key, sub_value in sub_config.items():
            if "folder" in sub_key:
                leaf_path = os.path.join(sub_root, sub_value)
                os.mkdir(leaf_path)


def save_image(path, image, option=True):
    if option:
        cv2.imwrite(path, image)


def detect(det, tensor_image, visible=False):
    begin = time.time()
    dt_image, result, raw = det.detection(tensor_image)
    print("Inference time: ", (time.time() - begin) * 1000, "ms")
    (block_boxes, _, box_scores) = result

    if visible:
        dt_image = image_handler.draw_boxes_info(tensor_image, result)

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


def main():
    # region Configuration
    config = ConfigMapper.config_copy(
        ConfigMapper.get_config(root='./config')
    )
    cleaning_folder(config)
    print(config)
    default_option = config['default']
    save_debug = default_option['save']
    # default path
    input_config = config['input']
    det_config = config['detector']
    det_config['raw_path'] = os.path.join(input_config['base_path'], det_config['raw_path'])
    block_detector = Detector(det_config)
    # 이미지셋 리스트 가져오기
    image_list = os.listdir(input_config['path'])
    # endregion

    # depth_analize
    test_path = r'D:\MnS\Presentation\Business Papers\2022_HanKookTire\Data\test\_'
    for image_path in image_list:
        begin_time = time.time()
        print(image_path)
        depth_filename = image_path.replace('_color', '_depth')
        full_path = os.path.join(input_config['depth_path'], depth_filename)
        image_depth = cv2.imread(full_path, cv2.IMREAD_ANYDEPTH)
        image_bgr = cv2.imread(os.path.join(input_config['path'], image_path), cv2.IMREAD_COLOR)
        # region Depth-Preprocessing
        # outlier cut depthfilter outlier_depth
        condition = image_depth < config['depthfilter']['outlier_depth'][0]
        image_depth[condition] = 0
        condition = image_depth > config['depthfilter']['outlier_depth'][1]
        image_depth[condition] = 0

        # normalize
        img_f = image_depth.astype(np.float32)
        img_norm = ((img_f - img_f.min()) * 255 / (img_f.max() - img_f.min()))
        image_depth = img_norm.astype(np.uint8)

        if save_debug:
            plt.subplot(3, 1, 1)
            plt.hist(image_depth.ravel(), 256, [0, 256])
            plt.subplot(3, 1, 2)
            plt.imshow(image_depth)
            plt.subplot(3, 1, 3)
            plt.imshow(image_bgr)
            plt.savefig(test_path + image_path)
            plt.clf()
        # endregion
        # region Light removal
        preproc_config = config['preprocessing']
        preproc_config['raw_path'] = os.path.join(input_config['base_path'], preproc_config['raw_path'])
        # light removal 단계
        # 1) RGB to LAB
        image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(image_lab)
        lab_mono_set = (l_channel, a_channel, b_channel)

        lab_path = os.path.join(preproc_config['raw_path'], preproc_config['lab_folder'])
        l_path = os.path.join(lab_path, 'L')
        a_path = os.path.join(lab_path, 'A')
        b_path = os.path.join(lab_path, 'B')
        lab_path_set = [l_path, a_path, b_path]

        for channel, element_path in enumerate(lab_path_set):
            save_image(os.path.join(element_path, image_path), lab_mono_set[channel], save_debug)
        # 2) LAB with median 25 to 100 and bilateral
        mf_path = os.path.join(preproc_config['raw_path'], preproc_config['mf_folder'])
        comp_path = os.path.join(preproc_config['raw_path'], preproc_config['comp_folder'])

        image_mf = cv2.medianBlur(l_channel, 75)
        save_image(os.path.join(mf_path, image_path), image_mf, save_debug)
        inverted_image = cv2.bitwise_not(image_mf)

        # addweighted가 아닌 add를 하면 제일 높은 블럭을 찾을 수 있다.
        image_composite = cv2.addWeighted(l_channel, 0.5, inverted_image, 0.5, 0)
        save_image(os.path.join(comp_path, image_path), image_composite, save_debug)

        # Light removal 완료
        proc_input_path = os.path.join(preproc_config['raw_path'], preproc_config['input_folder'])
        test_image = cv2.merge([image_composite, a_channel, b_channel])
        image_bgr = cv2.cvtColor(test_image, cv2.COLOR_LAB2BGR)
        save_image(os.path.join(proc_input_path, image_path), image_bgr)
        # endregion
        # region Blur
        proc_config = config['processing']
        proc_config['raw_path'] = os.path.join(input_config['base_path'], proc_config['raw_path'])
        input_path = os.path.join(proc_config['raw_path'], proc_config['input_folder'])

        # 패턴인식용 설정 가져오기
        edge_path = os.path.join(proc_config['raw_path'], proc_config['edge_folder'])

        # blur processing
        image_b, image_g, image_r = cv2.split(image_bgr)
        img_g_blur = cv2.GaussianBlur(image_b, (3, 3), sigmaX=0, sigmaY=0)
        img_b_blur = cv2.GaussianBlur(image_g, (3, 3), sigmaX=0, sigmaY=0)
        save_image(os.path.join(input_path, image_path), image_r, save_debug)
        image_edge = cv2.merge([img_b_blur, img_g_blur, image_r])
        save_image(os.path.join(edge_path, image_path), image_edge, save_debug)
        # endregion
        # region Block detection
        image_tensor = ImageManager.convert_tensor(image_edge, bgr2rgb=False)
        detected_image, info, raw_boxes = detect(det=block_detector, tensor_image=image_tensor, visible=save_debug)
        if save_debug:
            detected_path = os.path.join(det_config['raw_path'], det_config['detected_folder'])
            overlay_path = os.path.join(det_config['raw_path'], det_config['overlay_folder'])
            overlay_image = image_handler.draw_boxes_info(image_bgr, info, single=False)
            image_handler.save_tensor(img=detected_image, path=os.path.join(detected_path, image_path))
            image_handler.save_tensor(img=overlay_image, path=os.path.join(overlay_path, image_path))
            del detected_image
        (boxes, classes, scores) = info
        if len(boxes) == 0:
            print("Null detected")
            continue
        # block 정보 수집
        anchor_info = {'width': [], 'height': [], 'surface': [],
                       'coordinate': [], 'class': [], 'score': [],
                       'multiple': [], 'mean': [], 'depth_max': 0, 'depth_min': 0}
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

            roi = image_depth[int(height_min * 1.2):int(height_max*0.8), int(width_min*1.2):int(width_max*0.8)]
            mean = np.mean(roi[:, :])

            anchor_info['width'].append(width)
            anchor_info['height'].append(height)
            anchor_info['surface'].append(surface)
            anchor_info['coordinate'].append(boxes[i])
            anchor_info['class'].append(classes[i])
            anchor_info['score'].append(scores[i])
            anchor_info['multiple'].append(multiple)
            anchor_info['mean'].append(mean)

        # 면적이 중간인 box 추출
        alternatives = sorted(anchor_info['surface'])
        mid_surface = alternatives[int(len(alternatives) / 2)]

        # 면적이 안 맞는 블럭 제거
        for i in reversed(range(boxes.shape[0])):
            surface_gap = anchor_info['surface'][i] / mid_surface
            if surface_gap < 0.6:
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

        # 복층인 경우 (아래층 제거)
        depth_gap = anchor_info['depth_max'] - anchor_info['depth_min']
        print(depth_gap)
        if depth_gap > config['depthfilter']['height_gap']:
            cut_threshold = config['depthfilter']['normalize_thrd']
            norm = [(float(val) - anchor_info['depth_min']) / depth_gap for val in anchor_info['mean']]
            for i in reversed(range(len(norm))):
                if norm[i] > cut_threshold:
                    anchor_info['coordinate'].pop(i)
                    anchor_info['class'].pop(i)
                    anchor_info['score'].pop(i)

        boxes = np.array(anchor_info['coordinate'], dtype=float)
        classes = np.array(anchor_info['class'])
        scores = np.array(anchor_info['score'], dtype=float)

        if True:
            deleted_path = os.path.join(det_config['raw_path'], det_config['deleted_folder'])
            anchor_path = os.path.join(det_config['raw_path'], det_config['anchor_folder'])

            info = (boxes, classes, scores)
            deleted_image = image_handler.draw_boxes_info(image_bgr, info, single=False)
            image_handler.save_tensor(img=deleted_image, path=os.path.join(deleted_path, image_path))
            pickle_name = image_path.replace(".jpg", ".pickle")
            pack = (image_bgr, info, raw_boxes, anchor_info)
            with open(os.path.join(anchor_path, pickle_name), "wb") as fw:
                pickle.dump(pack, fw)
        # endregion
        # region Pattern matching
        mat_config = config['matcher']
        mat_config['raw_path'] = os.path.join(input_config['base_path'], mat_config['raw_path'])
        if len(anchor_info['coordinate']) > 1:
            # Making reference patterns
            reference_patterns_info = {'scores': [], 'blocks': [], 'rotate': [],
                                       'matched_blocks': [], 'pattern_names': []}

            for block_ratio, maps in mat_config['defined_patterns'].items():
                for pattern_name, value in maps.items():
                    if save_debug:
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

            # Matching algorithm
            img_height, img_width, _ = image_bgr.shape
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
            matched_block = boxes[matched_index]
            # Draw Image
            if True:
                detected_path = os.path.join(mat_config['raw_path'], mat_config['detected_folder'])
                fitted_path = os.path.join(mat_config['raw_path'], mat_config['fitted_folder'])
                image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                # 최종 패턴 이미지
                matched_image = image_handler.draw_just_boxes(image_bgr, boxes, color_idx=0)
                matched_image = image_handler.draw_just_boxes(matched_image, matched_block, color_idx=2)
                image_handler.save_tensor(img=matched_image, path=os.path.join(detected_path, image_path))
                # 이미지 쉐잎에 맞는 패턴 이미지 추출
                draw_blocks = copy.deepcopy(reference_patterns_info['blocks'][index])
                draw_blocks[:, [0, 2]] *= img_height
                draw_blocks[:, [1, 3]] *= img_width
                fitted_image = image_handler.draw_just_boxes(matched_image, draw_blocks, color_idx=7)
                image_handler.save_tensor(img=fitted_image, path=os.path.join(fitted_path, image_path))
            # end region
        print("Processed: " + str(time.time()-begin_time))
        print("----------------")
        # endregion


if __name__ == '__main__':
    image_handler = ImageManager()
    main()