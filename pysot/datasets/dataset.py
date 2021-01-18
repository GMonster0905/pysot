# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import sys
import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from pysot.utils.bbox import center2corner, Center
from pysot.datasets.anchor_target import AnchorTarget
from pysot.datasets.augmentation import Augmentation
from pysot.core.config import cfg

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class SubDataset(object):
    def __init__(self, name, root, anno, frame_range, num_use, start_idx):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.name = name
        self.root = os.path.join(cur_path, '../../', root)
        self.anno = os.path.join(cur_path, '../../', anno)
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        logger.info("loading " + name)
        with open(self.anno, 'r') as f:
            meta_data = json.load(f)
            # The returned meta_data only contained
            # the bbox annotations
            meta_data = self._filter_zero(meta_data)

        for video in list(meta_data.keys()):
            for track in meta_data[video]:

                frames = meta_data[video][track]
                frames = list(map(int,
                              filter(lambda x: x.isdigit(), frames.keys())))
                frames.sort()
                # Existing keys under track
                # %6d frame idx
                # Adding the 'frames' keys under
                # the track keys
                # Store the frames-keys in one 
                # track in the sorted list format
                meta_data[video][track]['frames'] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]

        for video in list(meta_data.keys()):
            if len(meta_data[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        self.labels = meta_data
        self.num = len(self.labels)
        # self.num = number of videos in this dataset
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.{}.{}.jpg'
        self.pick = self.shuffle()

    def _filter_zero(self, meta_data):
        # setting the loaded json training data
        # and filter out the wrong bbox which W
        # or H <0
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    # then the content in the new_frames
                    # dict only is the bbox
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        # Shuffle in SubDataset
        # Create the shuffled idx list
        # the length is equal to the self.num_use
        # self.num_use = self.num if self.num_use == -1 else self.num_use
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]

    def get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)
        # self.path_format = '{}.{}.{}.jpg'
        image_path = os.path.join(self.root, video,
                                  self.path_format.format(frame, track, 'x'))
        # image_anno is the bbox list of the single frame of single object
        image_anno = self.labels[video][track][frame]
        return image_path, image_anno

    def get_positive_pair(self, index):
        # self.videos = list(meta_data.keys())
        video_name = self.videos[index]
        # self.labels = meta_data
        video = self.labels[video_name]
        # video contain multi-tracks
        track = np.random.choice(list(video.keys()))
        # get the random track(object) keys
        track_info = video[track]
        # get the info of one track
        # including multi-bbox info
        frames = track_info['frames']
        # frames is a sorted dict
        # of frames-keys(bbox index)
        template_frame = np.random.randint(0, len(frames))
        # left >=0
        left = max(template_frame - self.frame_range, 0)
        # right <= len(frames) -1
        right = min(template_frame + self.frame_range, len(frames)-1) + 1
        search_range = frames[left:right]
        template_frame = frames[template_frame]
        search_frame = np.random.choice(search_range)
        # return pair of image_path, image_anno
        # both suffix is x
        return self.get_image_anno(video_name, track, template_frame), \
            self.get_image_anno(video_name, track, search_frame)

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        # frame is in int type
        # get_image_anno will first convert 
        # the frame var to a str
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num


class TrkDataset(Dataset):
    def __init__(self,):
        # To build the track training dataset
        super(TrkDataset, self).__init__()
        # Customization
        # (255  - 127) / 8 + 1 +8 = 25
        desired_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) / \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
        if desired_size != cfg.TRAIN.OUTPUT_SIZE:
            raise Exception('size not match!')

        # create anchor target
        self.anchor_target = AnchorTarget()

        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(
                    name,
                    subdata_cfg.ROOT,
                    subdata_cfg.ANNO,
                    subdata_cfg.FRAME_RANGE,
                    subdata_cfg.NUM_USE,
                    start
                )
            start += sub_dataset.num
            self.num += sub_dataset.num_use

            sub_dataset.log()
            self.all_dataset.append(sub_dataset)

        # data augmentation
        self.template_aug = Augmentation(
                cfg.DATASET.TEMPLATE.SHIFT,
                cfg.DATASET.TEMPLATE.SCALE,
                cfg.DATASET.TEMPLATE.BLUR,
                cfg.DATASET.TEMPLATE.FLIP,
                cfg.DATASET.TEMPLATE.COLOR
            )
        self.search_aug = Augmentation(
                cfg.DATASET.SEARCH.SHIFT,
                cfg.DATASET.SEARCH.SCALE,
                cfg.DATASET.SEARCH.BLUR,
                cfg.DATASET.SEARCH.FLIP,
                cfg.DATASET.SEARCH.COLOR
            )
        # cfg.DATASET.VIDEOS_PER_EPOCH = 600000
        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= cfg.TRAIN.EPOCH
        self.pick = self.shuffle()

    def shuffle(self):
        # Shuffle in TrkDataset
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick
                p += sub_p
            # len(p) = len(all_dataset)*sub_dataset.num_use
            # max(p) = sum(sub_dataset.num)
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[:self.num]

    def _find_dataset(self, index):
        # use the idx to finding the
        # corresponding dataset
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):
        # convert the bbox of a corresponding 
        # image to adapt to the cropped img
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        # convert (cx, cy, w, h) to (x1, y1, x2, y2)
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        index = self.pick[index]
        dataset, index = self._find_dataset(index)
        # index < dataset.num

        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()

        # get one dataset
        if neg:
            # get a negtive pair(frame)
            template = dataset.get_random_target(index)
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            # get a positve pair(frame)
            template, search = dataset.get_positive_pair(index)
        # template[0] is the image path
        # template[1] is the bbox info of the image
        # get image
        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])

        # get adjusted bounding box
        template_box = self._get_bbox(template_image, template[1])
        search_box = self._get_bbox(search_image, search[1])

        # augmentation
        # return the augmented image and the changed bbox
        template, _ = self.template_aug(template_image,
                                        template_box,
                                        cfg.TRAIN.EXEMPLAR_SIZE,
                                        gray=gray)

        search, bbox = self.search_aug(search_image,
                                       search_box,
                                       cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)

        # get labels
        cls, delta, delta_weight, overlap = self.anchor_target(
                bbox, cfg.TRAIN.OUTPUT_SIZE, neg)
        # template in the (H,W,rgb_channel)
        # transposed : (rgb_channel,H,W)
        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = search.transpose((2, 0, 1)).astype(np.float32)
        return {
                'template': template,
                'search': search,
                'label_cls': cls,
                'label_loc': delta,
                'label_loc_weight': delta_weight,
                'bbox': np.array(bbox)
                }
