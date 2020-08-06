import os
import random
import tensorflow as tf
from tensorcv.data.dataset import TSVDataset


def _random_crop_and_flip(image, crop_height, crop_width):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    total_crop_height = (height - crop_height)
    crop_top = tf.random_uniform(
        [], maxval=total_crop_height + 1, dtype=tf.int32)
    total_crop_width = (width - crop_width)
    crop_left = tf.random_uniform(
        [], maxval=total_crop_width + 1, dtype=tf.int32)

    cropped = tf.slice(
        image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])

    cropped = tf.image.random_flip_left_right(cropped)
    return cropped


class FashionaiAttributeDataset(TSVDataset):
    def read_lines(self, path, mode):
        params = self.config.dataset_params
        lines = []
        attribute = params['attribute']
        with tf.gfile.Open(path) as f:
            for line in f:
                path, t, label = line.split(',')
                if mode == tf.estimator.ModeKeys.PREDICT:
                    path = os.path.join(params['test_data_folder'], path)
                else:
                    path = os.path.join(params['train_data_folder'], path)
                if 'y' in label:
                    label = str(label.index('y'))
                else:
                    label = '-1'
                if t == attribute:
                    lines.append([path, label])
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        if is_training: 
            random.shuffle(lines)
        return lines

    def parse_fn(self, line, mode):
        image_path = line[0]
        label = tf.string_to_number(line[1], out_type=tf.int32)
        image = tf.read_file(image_path)
        image = tf.image.decode_jpeg(image, self.config.image_channels)
        h, w = self.config.image_height, self.config.image_width
        if mode == tf.estimator.ModeKeys.TRAIN:
            image = tf.image.resize_images(image, (h + 32, w + 32))
            image = _random_crop_and_flip(image, h, w)
        else:
            image = tf.image.resize_images(image, (h, w))
        image.set_shape([
            self.config.image_height, self.config.image_width,
            self.config.image_channels
        ])
        image = tf.cast(image, dtype=tf.float32)

        return image, label

