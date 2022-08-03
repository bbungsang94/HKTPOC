import copy

import torchvision
import tensorflow as tf
from detector.utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords, xyxy2xywh
from utilities.media_handler import *
import torch
import torch.nn.functional as nnf
import numpy as np
import time
import cv2

from detector.utils.general import xywh2xyxy


class Detector:
    def __init__(self, config):

        self.model_name = config['model_name']
        self.hub_mode = config['hub_mode']
        self.model_handle = config['model_handle']
        self.label_path = config['label_path']
        self.box_min_score = config['box_min_score']
        self.iou_score = config['iou_score']
        self.offset_score = config['offset_score']
        self.max_boxes = config['max_boxes']

        from detector.models.experimental import attempt_load
        self.Detector = attempt_load(self.model_handle, map_location=torch.device('cuda:0'))

        # self.Detector.eval()

    def detection(self, input_image):
        """Determines the locations of the vehicle in the image

                Args:
                    input_image: image(tensor)
                Returns:
                    list of bounding boxes: coordinates [y_up, x_left, y_down, x_right]

                """
        # converted_img = tf.image.convert_image_dtype(image, tf.float32)
        image = input_image.permute(2, 0, 1)
        device = torch.device('cuda:0')
        image = image.float().to(device)
        image /= 255
        if len(image.shape) == 3:
            converted_img = image[None]  # expand for batch dim
        else:
            converted_img = image

        out = nnf.interpolate(converted_img, size=(480, 640), mode='bicubic', align_corners=False)
        pred = self.Detector(out, augment=False, visualize=False)
        pred = non_max_suppression(pred[0], self.box_min_score, self.iou_score)
        det = pred[0]
        gn = torch.tensor([640, 480, 640, 480])
        result = dict()
        result["detection_boxes"] = np.empty([0, 0])
        result["detection_class_labels"] = np.empty([0, 0])
        result["detection_scores"] = np.empty([0, 0])
        result["detection_class"] = np.empty([0, 0], dtype=np.str)

        detection_boxes = []
        detection_class_labels = []
        detection_scores = []
        detection_class = []
        if len(det):
            # Rescale boxes from img_size to img0 size
            det[:, :4] = scale_coords(converted_img.shape[2:], det[:, :4], input_image.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                detection_boxes.append(xywh)
                detection_class_labels.append(int(cls))
                detection_scores.append(float(conf))
                detection_class.append(None)
            result["detection_boxes"] = np.array(detection_boxes)
            result["detection_class_labels"] = np.array(detection_class_labels)
            result["detection_scores"] = np.array(detection_scores)
            result["detection_class"] = np.array(detection_class)

            # print(f'Inferencing and Processing Done. ({time.time() - t0:.3f}s)')
            # result = {key: value.numpy() for key, value in result.items()}
        info = tf.shape(input_image)
        box_info, raw_boxes = self.__post_process(result, info)
        return converted_img, box_info, raw_boxes

    def __post_process(self, result, info):
        boxes = result["detection_boxes"]
        classe_labels = result["detection_class_labels"]
        scores = result["detection_scores"]
        classes = result["detection_class"]

        cls_idx = classe_labels == 0
        box_idx = scores > self.box_min_score
        box_idx = cls_idx & box_idx
        classes[box_idx] = "Block"
        img_shape = np.array(info)

        box_boxes = boxes[box_idx]
        raw_boxes = copy.deepcopy(box_boxes)
        box_boxes = self.get_zboxes(box_boxes, im_width=img_shape[1], im_height=img_shape[0])
        box_classes = classes[box_idx]
        box_scores = scores[box_idx]
        box_info = (box_boxes, box_classes, box_scores)

        return box_info, raw_boxes

    def get_zboxes(self, boxes, im_width, im_height):
        z_boxes = []
        for i in range(min(boxes.shape[0], self.max_boxes)):
            x_center, y_center, width, height = tuple(boxes[i])

            (left, right, top, bottom) = ((x_center - width / 2) * im_width, (x_center + width / 2) * im_width,
                                          (y_center - height / 2) * im_height, (y_center + height / 2) * im_height)
            z_boxes.append([top, left, bottom, right])
        return np.array(z_boxes)
