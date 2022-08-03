import os
import cv2
import torch
import imageio
import numpy as np

import tensorflow as tf
from PIL import ImageColor, ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt


class VideoManger:
    def __init__(self):
        self.__video_list = []
        self.__frame_list = []
        self.__video_object = None
        self.__frame_rate = 0

        self._frame_rate = None
        self._image_width = 0
        self._image_height = 0
        self._frame_count = -1

        self.video_name = ''
        self.input_path = ''

    def scan_video(self, path):
        """ Scanning videos(mp4 and avi) of path

        Args:
            path: video folder
        Returns:
            video list: video filenames [A.avi, B.mp4, ..., N.avi or N.mp4]
        """

        self.__video_list = []
        files = os.listdir(path)
        for file in files:
            if '.mp4' in file or '.avi' in file:
                self.__video_list.append(file)

        return self.__video_list

    def load_video(self, path):
        """ Activate video object of path

                Args:
                    path: a video filename
                Returns:
                    process result: boolean {True, False}
        """
        self.__frame_list.clear()
        self.__video_object = cv2.VideoCapture(path)
        fps = self.__video_object.get(cv2.CAP_PROP_FPS)
        width = int(self.__video_object.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.__video_object.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
        print("Image Width using video.get(cv2.CAP_PROP_FRAME_WIDTH) : {0}".format(width))
        print("Image Height using video.get(cv2.CAP_PROP_FRAME_HEIGHT) : {0}".format(height))

        self._frame_rate = fps
        self._image_width = width
        self._image_height = height

        return fps, (width, height)

    def update_video_property(self, path):
        if self.load_video(path):
            self.__video_object = None
            return True
        return False

    def append(self, img):
        """ for making output video, append image object

                Args:
                    img: an image using opencv2
                Returns:
                    None
        """
        self.__frame_list.append(img)

    def pop(self):
        if self.__video_object is None:
            return 0, None
        else:
            self._frame_count += 1
            rtn, img = self.__video_object.read()
            if rtn:
                return self._frame_count, img
            else:
                return 0, None

    def write_video(self, path):
        """ for making output video, append image object

                Args:
                    path: save path conclude video name
                Returns:
                    process result: boolean {True, False}
        """
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'),
                              self._frame_rate, (self._image_width, self._image_height))
        for i in range(len(self.__frame_list)):
            # writing to an image array
            out.write(self.__frame_list[i])
        out.release()

    def make_image_name(self, frame_count=-1):
        if frame_count == -1:
            frame_count = self._frame_count
        time_val = frame_count / self._frame_rate
        time_str = "{:011.5f}".format(time_val)
        time_str = time_str.replace('.', '-')
        time_str = time_str + '.jpeg'
        return time_str


class PipeliningVideoManager(VideoManger):
    def __init__(self):
        super().__init__()
        self.__OutputVideo = None

    def activate_video_object(self, path, v_info: tuple = None):
        if v_info is not None:
            self._frame_rate = v_info[0]
            self._image_width = v_info[1]
            self._image_height = v_info[2]

        if self._frame_rate is None:
            return False
        else:
            cv2.VideoWriter()
            self.__OutputVideo = cv2.VideoWriter(path,
                                                 cv2.VideoWriter_fourcc(*'DIVX'),
                                                 self._frame_rate,
                                                 (self._image_width, self._image_height))
            return True

    def append(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.__OutputVideo.write(img)

    def release_video_object(self):
        if self.__OutputVideo is not None:
            self.__OutputVideo.release()
            return True
        return False


class ImageManager:
    def __init__(self):
        self.test = True
        self.figure = None
        self.Colors = list(ImageColor.colormap.values())

        try:
            self.Font = ImageFont.truetype(
                "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 25)
        except IOError:
            print("Font not found, using default font.")
            self.Font = ImageFont.load_default()

    @staticmethod
    def load_tensor(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        return img

    @staticmethod
    def load_cv(path):
        img = cv2.imread(path)
        return img

    @staticmethod
    def convert_uint(img):
        converted_img = tf.image.convert_image_dtype(img, tf.uint8)[tf.newaxis, ...]
        return converted_img

    @staticmethod
    def convert_tensor(img: np.array, bgr2rgb=True):
        if bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # converted_img = tf.image.convert_image_dtype(img, tf.uint8)[tf.newaxis, ...]
        converted_img = torch.from_numpy(img)
        return converted_img

    @staticmethod
    def save_image(img: np.array, path):
        cv2.imwrite(path, img)

    @staticmethod
    def save_tensor(img, path):
        imageio.imwrite(path, img)

    @staticmethod
    def get_roi(img: np.array, box):
        # img: np image
        height_min = int(min(box[0], box[2]))
        height_max = int(max(box[0], box[2]))
        width_min = int(min(box[1], box[3]))
        width_max = int(max(box[1], box[3]))

        roi = img[height_min:height_max, width_min:width_max]
        return roi

    def draw_bounding_box_on_image(self, image,
                                   top, left, bottom, right, color,
                                   thickness=4, display_str_list=()):
        """Adds a bounding box to an image."""
        draw = ImageDraw.Draw(image)

        draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                   (left, top)],
                  width=thickness,
                  fill=color)

        # If the total height of the display strings added to the top of the bounding
        # box exceeds the top of the image, stack the strings below the bounding box
        # instead of above.
        display_str_heights = [self.Font.getsize(ds)[1] for ds in display_str_list]
        # Each display_str has a top and bottom margin of 0.05x.
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = bottom + total_display_str_height
        # Reverse list and print from bottom to top.
        for display_str in display_str_list[::-1]:
            text_width, text_height = self.Font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)
            draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                            (left + text_width, text_bottom)],
                           fill=color)
            draw.text((left + margin, text_bottom - text_height - margin),
                      display_str,
                      fill="black",
                      font=self.Font)
            text_bottom -= text_height - 2 * margin

    def draw_boxes(self, image, boxes, classes, scores):
        """Overlay labeled boxes on an image with formatted scores and label names."""
        np_image = image.numpy()
        np_image = np_image[:, :, ::-1]
        image_pil = Image.fromarray(np_image)
        for i in range(boxes.shape[0]):
            display_str = "{}: {}%".format(classes[i], int(100 * scores[i]))
            color = self.Colors[2 % len(self.Colors)]

            top, left, bottom, right = tuple(boxes[i])
            self.draw_bounding_box_on_image(
                image_pil, top, left, bottom, right,
                color, display_str_list=[display_str])
        return np.array(image_pil)

    def draw_just_boxes(self, image, boxes, color_idx=2):
        if isinstance(image, np.ndarray):
            np_image = image
        else:
            np_image = image.numpy()
        image_pil = Image.fromarray(np_image)

        for i in range(boxes.shape[0]):
            color = self.Colors[color_idx % len(self.Colors)]

            top, left, bottom, right = tuple(boxes[i])
            self.draw_bounding_box_on_image(
                image_pil, top, left, bottom, right,
                color)
        return np.array(image_pil)

    def draw_boxes_info(self, image, info, single=False, color_idx=2):
        """Overlay labeled boxes on an image with formatted scores and label names."""
        if isinstance(image, np.ndarray):
            np_image = image
        else:
            np_image = image.numpy()
        if single:
            np_image = np_image[:, :]
        else:
            np_image = np_image[:, :, ::-1]
        image_pil = Image.fromarray(np_image)

        (boxes, classes, scores) = info
        for i in range(boxes.shape[0]):
            display_str = "{}: {}%".format(classes[i], int(100 * scores[i]))
            color = self.Colors[color_idx % len(self.Colors)]

            top, left, bottom, right = tuple(boxes[i])
            self.draw_bounding_box_on_image(
                image_pil, top, left, bottom, right,
                color, display_str_list=[display_str])
        return np.array(image_pil)

    def display_image(self, img):
        self.figure = plt.figure(figsize=(20, 15))
        plt.grid(False)
        plt.imshow(img)


if __name__ == "__main__":
    test = VideoManger()
    video_list = test.scan_video(path='../video')
    test.load_video('../video/' + video_list[0])
