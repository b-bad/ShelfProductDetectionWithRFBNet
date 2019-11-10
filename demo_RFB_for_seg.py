from __future__ import print_function
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from data import VOCroot
from data import AnnotationTransform, VOCDetection, BaseTransform, VOC_300,VOC_512

from collections import OrderedDict
import torch.utils.data as data
from layers.functions import Detect,PriorBox
# from utils.nms_wrapper import nms
from utils.timer import Timer
import time
import pprint
import cv2
import pandas as pd
import codecs
import scipy.signal as signal
import heapq



def parse_args():
    parser = argparse.ArgumentParser(description='Receptive Field Block Net')

    parser.add_argument('-v', '--version', default='RFB_vgg',
                        help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
    parser.add_argument('-s', '--size', default='300',
                        help='300 or 512 input size.')
    parser.add_argument('-d', '--dataset', default='VOC',
                        help='VOC or COCO version')
    parser.add_argument('-m', '--trained_model', default='weights/RFB300_80_5.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--save_folder', default='test_inter/', type=str,
                        help='Dir to save demo images')
    parser.add_argument('--cuda', default=False, type=bool,
                        help='Use cuda to train model')
    parser.add_argument('--cpu', default=False, type=bool,
                        help='Use cpu nms')
    parser.add_argument('--save_result', default='eavl/', type=str,
                        help='Dir to save results')
    args = parser.parse_args('--version RFB_E_vgg --size 512 --dataset VOC --save_folder images/ '
                             '--trained_model weights/Final_RFB_E_vgg_VOC.pth'.split())
    return args



class ObjectDetector:
    def __init__(self, net, detection, transform, num_classes=21, cuda=False, max_per_image=300, thresh=0.5):
        self.net = net
        self.detection = detection
        self.transform = transform
        self.num_classes = num_classes
        self.max_per_image = max_per_image
        self.cuda = cuda
        self.thresh = thresh

    def predict(self, img):
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]]).cpu().numpy()
        _t = {'im_detect': Timer(), 'misc': Timer()}
        assert img.shape[2] == 3
        with torch.no_grad():
            x = self.transform(img).unsqueeze(0)
        if self.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        out = net(x)  # forward pass
        boxes, scores = self.detection.forward(out, priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image
        boxes *= scale
        _t['misc'].tic()
        all_boxes = [[] for _ in range(num_classes)]

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > self.thresh)[0]
            if len(inds) == 0:
                all_boxes[j] = np.zeros([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            # print(scores[:, j])
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            # keep = nms(c_bboxes,c_scores)

            if USE_softnms:
                from utils.soft_nms import soft_nms
                keep = soft_nms(c_dets, method = 2)
            else:
                keep = nms(c_dets, 0.45, force_cpu=args.cpu)
            keep = keep[:50]
            c_dets = c_dets[keep, :]
            all_boxes[j] = c_dets
        if self.max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][:, -1] for j in range(1, num_classes)])
            if len(image_scores) > self.max_per_image:
                image_thresh = np.sort(image_scores)[-self.max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][:, -1] >= image_thresh)[0]
                    all_boxes[j] = all_boxes[j][keep, :]

        nms_time = _t['misc'].toc()
        # print('net time: ', detect_time)
        # print('post time: ', nms_time)
        return all_boxes



if __name__ == '__main__':

    USE_softnms = True

    starttime = time.time()
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.dataset == 'COCO':
        from data import COCOroot, COCO_300, COCO_512, COCO_mobile_300, AnnotationTransform, COCODetection, \
            detection_collate, BaseTransform, preproc

    FONT = cv2.FONT_HERSHEY_PLAIN
    out_file = 'result_inter/'
    if not os.path.exists(out_file):
        os.mkdir(out_file)

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    if args.dataset == 'VOC':
        cfg = (VOC_300, VOC_512)[args.size == '512']
    else:
        cfg = (COCO_300, COCO_512)[args.size == '512']

    if args.version == 'RFB_vgg':
        from models.RFB_Net_vgg import build_net
    elif args.version == 'RFB_E_vgg':
        from models.RFB_Net_E_vgg import build_net
    elif args.version == 'RFB_mobile':
        from models.RFB_Net_mobile import build_net

        cfg = COCO_mobile_300
    else:
        print('Unkown version!')

    pprint.pprint(cfg)

    priorbox = PriorBox(cfg)
    with torch.no_grad():
        priors = priorbox.forward()
        if args.cuda:
            priors = priors.cuda()

    img_dim = (300, 512)[args.size == '512']

    pascal_classes = np.asarray(['__background__',
                                 'ksf32', 'kkkl', 'wt', 'hfs',
                                 'hryg', 'hsnrm', 'bbz', 'hzdg'])
    count = np.zeros(9)

    num_classes = np.shape(pascal_classes)[0]
    net = build_net('test', img_dim, num_classes)  # initialize detector
    state_dict = torch.load('F:/数据备份/大四上（电脑）/大创1/task_xuedi（pytorch测试用）/weights/RFB_E_vgg_VOC_epoches_250.pth', map_location=lambda storage, loc:storage)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`·
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    print('Finished loading model!')

    # demo_set = os.listdir(args.save_folder)
    demo_set = os.listdir('F:/数据备份/大四上（电脑）/大创1/task_xuedi（pytorch测试用）/test_inter/')

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()

    detector = Detect(num_classes, 0, cfg)
    save_folder = args.save_result
    rgb_means = ((104, 117, 123), (103.94, 116.78, 123.68))[args.version == 'RFB_mobile']

    top_k = 300
    transform = BaseTransform(net.size, rgb_means, (2, 0, 1))
    object_detector = ObjectDetector(net, detector, transform, num_classes, args.cuda)
    
    txt = open("F:/数据备份/大四上（电脑）/大创1/task_xuedi（pytorch测试用）/android_result.txt", "w")

    image_save = []
    '''for image_id in range(len(demo_set)):
        image_per_starttime = time.time()
        print('##### Start display #####')
        # image_file = os.path.join(args.save_folder, demo_set[image_id])
        image_file = os.path.join('F:/数据备份/大四上（电脑）/大创1/task_xuedi（pytorch测试用）/test_inter/', demo_set[image_id])
        print(image_file)
        image = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), cv2.IMREAD_COLOR)'''
    image = cv2.imdecode(np.fromfile('F:/数据备份/大四上（电脑）/大创1/task_xuedi（pytorch测试用）/test_inter/test.jpg', dtype=np.uint8), cv2.IMREAD_COLOR)
    detect_bboxes = object_detector.predict(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    canny = cv2.canny(image, 60, 100)
    ret, thresh = cv2.threshold(canny, 128, 1, cv2.THRESH_BINARY)
    summ = thresh.sum(axis = 1)
    def smooth(a,WSZ):
        # a:原始数据，NumPy 1-D array containing the data to be smoothed
        # 必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化 
        # WSZ: smoothing window size needs, which must be odd number,
        # as in the original MATLAB implementation
        out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
        r = np.arange(1,WSZ-1,2)
        start = np.cumsum(a[:WSZ-1])[::2]/r
        stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
        return np.concatenate((  start , out0, stop  ))

    sm = smooth(summ, 19)
    peaks, _ = signal.find_peaks(-sm, distance=80)
    value = peaks.copy()
    for i in range(len(peaks)):
        value[i] = sm[peaks[i]]

    peaks_find = heapq.nsmallest(8, range(len(value)), value.take)
    i = 0
    while True:
        j = i + 1
        flag = peaks[peaks_find[i]]
        while j < len(peaks_find):
            if abs(flag - peaks[peaks_find[j]]) < max(200, image.shape[0] / 6):
                peaks_find.pop(i)
            else:
                j += 1
        i += 1
        if i >= len(peaks_find):
            break
    place = []
    place.append(0)
    for i in range(len(peaks_find)):
        place.append(peaks_find[i])
    place.append(image.shape[0])

    cnt = np.zeros((len(peaks_find)+1,9))
    for i in range(1, len(peaks_find)+2):
        seg = image[place[i - 1]:place[i],:,:]
        detect_bboxes = object_detector.predict(seg)
        for class_id, class_collection in enumerate(detect_bboxes):
            if class_id == 0:
                continue
            if len(class_collection) > 0:
                for j in range(class_collection.shape[0]):
                    if class_collection[j, -1] > 0.75:
                        cnt[i][class_id] += 1
    
    max_ = np.zeros(9)
    for i in range(9):
        max_[i] = np.max(cnt[:,i])
    for i in range(1, len(peaks_find)+2):
        seg = image[place[i - 1]:place[i],:,:]
        detect_bboxes = object_detector.predict(seg)
        for class_id, class_collection in enumerate(detect_bboxes):
            if class_id == 0:
                continue
            if cnt[i][class_id] <= max(2, max_[class_id]*0.333334):
                continue
            if len(class_collection) > 0:
                c = 0
                for j in range(class_collection.shape[0]):
                    c += 1
                    pt = class_collection[j]
                    cv2.rectangle(image, (int(pt[0])), (int(pt[1]) + place[i-1]),
                                                    (int(pt[2])), (int(pt[3]) + place[i-1]),
                                                    (0, 0, 255), 5)
                    cv2.putText(image, pascal_classes[class_id], (int(pt[0]), int(pt[1])+place[i-1]), FONT, 8, (0, 0, 255), thickness=3)
                    with codecs.open("F:/数据备份/大四上（电脑）/大创1/task_xuedi（pytorch测试用）/android_result.txt",'a',encoding='utf-8') as f:
                        f.write(pascal_classes[class_id]+":"+str(int(pt[0]))+","+str(int(pt[1])+place[i-1])+","+str(int(pt[2]))+","+str(int(pt[3])+place[i-1])+";")
    cv2.imencode('.jpg', image)[1].tofile("test_det.jpg")


    endTime = time.time()
    print('运行时间：', endTime-starttime)

