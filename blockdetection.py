import time
import shutil
import pickle
from detector.object_detector import Detector
import utilities.config_mapper as ConfigMapper
from utilities.media_handler import *

image_handler = ImageManager()


def detect(det, tensor_image, visible=False):
    begin = time.time()
    dt_image, result, raw = det.detection(tensor_image)
    print("Inference time: ", (time.time() - begin) * 1000, "ms")
    (block_boxes, _, box_scores) = result

    if visible:
        dt_image = image_handler.draw_boxes_info(tensor_image, result)

    return dt_image, result, raw


if __name__ == '__main__':
    # region Configuration
    # 설정 가져오기
    config = ConfigMapper.config_copy(
        ConfigMapper.get_config(root='./config')
    )
    print(config)
    default_option = config['default']
    save_debug = default_option['save']

    input_config = config['input']
    det_config = config['detector']
    det_config['raw_path'] = os.path.join(input_config['base_path'], det_config['raw_path'])
    temp_path = os.path.join(input_config['base_path'],
                             config['processing']['raw_path'],
                             config['processing']['edge_folder'])

    detected_path = os.path.join(det_config['raw_path'], det_config['detected_folder'])
    overlay_path = os.path.join(det_config['raw_path'], det_config['overlay_folder'])
    deleted_path = os.path.join(det_config['raw_path'], det_config['deleted_folder'])
    anchor_path = os.path.join(det_config['raw_path'], det_config['anchor_folder'])
    image_list = os.listdir(temp_path)

    if os.path.exists(det_config['raw_path']):
        shutil.rmtree(det_config['raw_path'])

    make_folder_list = [det_config['raw_path'],
                        detected_path, overlay_path, deleted_path, anchor_path]
    for folder in make_folder_list:
        os.mkdir(folder)
    # endregion

    blur_path = os.path.join(input_config['base_path'],
                             config['processing']['raw_path'],
                             config['processing']['norm_folder'])

    block_detector = Detector(det_config)
    multiple_gap_count = 0
    mean_gap_count = 0
    for image_path in image_list:
        full_path = os.path.join(temp_path, image_path)
        image_bgr = cv2.imread(full_path, cv2.IMREAD_COLOR)
        (h, w, c) = image_bgr.shape
        if h > w:
            image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)

        image_tensor = ImageManager.convert_tensor(image_bgr, bgr2rgb=False)

        detected_image, info, raw_boxes = detect(det=block_detector, tensor_image=image_tensor, visible=save_debug)

        image_blur = image_bgr # cv2.imread(os.path.join(blur_path, image_path))
        overlay_image = image_handler.draw_boxes_info(image_blur, info, single=False)
        if save_debug:
            image_handler.save_tensor(img=detected_image, path=os.path.join(detected_path, image_path))
            image_handler.save_tensor(img=overlay_image, path=os.path.join(overlay_path, image_path))

        (boxes, classes, scores) = info
        # 가장 큰 면적을 가진 블럭을 최상단 블럭으로 인지
        largest_anchor = {'width': 0, 'height': 0, 'surface': 0, 'index': -1, 'multiple': 1.0, 'mean': 0}
        image_blur_mono, _, _ = cv2.split(image_blur)
        for i in range(boxes.shape[0]):
            top, left, bottom, right = tuple(boxes[i])
            height_min = int(min(top, bottom))
            height_max = int(max(top, bottom))
            width_min = int(min(left, right))
            width_max = int(max(left, right))

            height = height_max - height_min
            width = width_max - width_min
            surface = width * height
            if surface > largest_anchor['surface']:
                largest_anchor['width'] = width
                largest_anchor['height'] = height
                largest_anchor['surface'] = surface
                largest_anchor['index'] = i
                if width > height:
                    largest_anchor['multiple'] = width / height
                else:
                    largest_anchor['multiple'] = height / width

                roi = image_blur_mono[height_min:height_max, width_min:width_max]
                largest_anchor['mean'] = np.mean(roi[:, :])

        # 비율이 안 맞는 블럭 (반으로 잘렸거나 밑 층에 있는 경우), 비닐 속 블럭을 인지했을 경우
        allive_idx = []
        for i in range(boxes.shape[0]):
            top, left, bottom, right = tuple(boxes[i])
            height_min = int(min(top, bottom))
            height_max = int(max(top, bottom))
            width_min = int(min(left, right))
            width_max = int(max(left, right))

            height = height_max - height_min
            width = width_max - width_min
            if width > height:
                multiple = width / height
            else:
                multiple = height / width

            multiple_gap = abs(largest_anchor['multiple'] - multiple)

            roi = image_blur_mono[height_min:height_max, width_min:width_max]
            mean = np.mean(roi[:, :])
            mean_gap = abs(largest_anchor['mean'] - mean)
            mean_gap = mean_gap / 255

            if multiple_gap > 0.5:
                multiple_gap_count += 1
                print("Multiple Gap:" + str(multiple_gap))
                print(multiple_gap_count)
            elif mean_gap > 1.0:
                mean_gap_count += 1
                print("Mean Gap:" + str(mean_gap))
                print(multiple_gap_count)
            else:
                allive_idx.append(i)

        boxes = boxes[allive_idx]
        classes = classes[allive_idx]
        scores = scores[allive_idx]
        info = (boxes, classes, scores)

        deleted_image = image_handler.draw_boxes_info(image_blur, info, single=False)
        if save_debug:
            #file_prefix = "{:.2f}".format(largest_anchor['multiple'])
            image_handler.save_tensor(img=deleted_image, path=os.path.join(deleted_path, image_path))
            pickle_name = image_path.replace(".jpg", ".pickle")
            pack = (image_bgr, info, raw_boxes, largest_anchor)
            with open(os.path.join(anchor_path, pickle_name), "wb") as fw:
                pickle.dump(pack, fw)

