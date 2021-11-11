# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 19:26:55 2021

@author: Kyle
"""

import json
import os
import cv2
import numpy as np
from shapely.geometry import *

root_path = ".\data"

def batch_rename(path):
    count = 0
    path = root_path + path
    for fname in os.listdir(path):
        new_fname = str(count) + '.json'
        os.rename(os.path.join(path, fname), os.path.join(path, new_fname))
        count = count + 1     

def SBD_order(segs,index):
    xt = [segs[ikpart] for ikpart in range(0, len(segs), 2)]
    yt = [segs[ikpart] for ikpart in range(1, len(segs), 2)]
    xt.sort()
    yt.sort()
    SBD=[]
    SBD.append(xt[0])
    SBD.append(yt[0])
    SBD.append(xt[3])
    SBD.append(yt[3])
    SBD = SBD + segs
    with open('.\data/json/' + index + '.txt', 'a') as f:
        for x, item in enumerate(SBD):
            if x == 11:
                f.write('%s' %item)
            else:
                f.write('%s,' %item)
        f.write('\n')
    return

def Labelme_to_SBD():
    _indexes = sorted([f.split('.')[0]
                       for f in os.listdir(os.path.join(root_path, 'json'))])
    indexes = [line for line in _indexes]
    for index in  indexes:
        file = os.path.join(root_path, 'json/') + index + '.json'
    
        with open(file,encoding="utf-8") as f:
            f = json.load(f)
            for item_shapes in f["shapes"]:
                s = []
                for item_points in item_shapes["points"]:
                    for y in range(2):
                        s.append(item_points[y])
                        if (len(s)==8):
                            SBD_order(s,index)
def convert_to_BDN():
    root_path = ".\data"
    phase = 'train'
    split = 0
    dataset = {
        'licenses': [],
        'info': {},
        'categories': [],
        'images': [],
        'annotations': []
    }
    with open(os.path.join(root_path, 'classes.txt')) as f:
      classes = f.read().strip().split()
    for i, cls in enumerate(classes, 1):
      dataset['categories'].append({
          'id': i,
          'name': cls,
          'supercategory': 'beverage',
          'keypoints': ['mean',
                        'xmin',
                        'x2',
                        'x3',
                        'xmax',
                        'ymin',
                        'y2',
                        'y3',
                        'ymax',
                        'cross']  # only for keypoints
      })
    
    
    def get_category_id(cls):
      for category in dataset['categories']:
        if category['name'] == cls:
          return category['id']
    
    
    _indexes = sorted([f.split('.')[0]
                       for f in os.listdir(os.path.join(root_path, 'json/vaildation_json'))])
    
    if phase == 'train':
      indexes = [line for line in _indexes if int(
          line) >= split]  # only for this file
    else:
      indexes = [line for line in _indexes if int(line) <= split]
    j = 1
    for index in indexes:
      print('Processing: ' + index)
      im = cv2.imread(os.path.join(root_path, 'images/vaildation_images/') + index + '.jpg')
      height, width, _ = im.shape
      dataset['images'].append({
          'coco_url': '',
          'date_captured': '',
          'file_name': index + '.jpg',
          'flickr_url': '',
          'id': int(index),
          'license': 0,
          'width': width,
          'height': height
      })
      anno_file = os.path.join(root_path, 'json/vaildation_json/') + index + '.txt'
      with open(anno_file) as f:
        lines = [line for line in f.readlines() if line.strip()]
        for i, line in enumerate(lines):
          parts = line.strip().split(',')
          cls = 'text'
          xmin = int(parts[0])
          ymin = int(parts[1])
          xmax = int(parts[2])
          ymax = int(parts[3])
          width = max(0, xmax - xmin + 1)
          height = max(0, ymax - ymin + 1)
          if width == 0 or height == 0:
            continue
          # add seg 
          segs = [int(kkpart) for kkpart in parts[4:]]  # four points
          xt = [segs[ikpart] for ikpart in range(0, len(segs), 2)]
          yt = [segs[ikpart] for ikpart in range(1, len(segs), 2)]
          # cross
          l1 = LineString([(xt[0], yt[0]), (xt[2], yt[2])])
          l2 = LineString([(xt[1], yt[1]), (xt[3], yt[3])])
          p_l1l2 = l1.intersection(l2)
          poly1 = Polygon([(xt[0], yt[0]), (xt[1], yt[1]),
                           (xt[2], yt[2]), (xt[3], yt[3])])
          if not poly1.is_valid:
            print('Not valid polygon found. This bounding box is removing ...')
            continue
          if not p_l1l2.within(poly1):
            print('Not valid intersection found. This bounding box is removing ...')
            continue
          if poly1.area <= 10:
            print('Text region too small. This bounding box is removing ...')
    
          mean_x = np.mean(xt)
          mean_y = np.mean(yt)
          xt_sort = np.sort(xt)
          yt_sort = np.sort(yt)
          xt_argsort = list(np.argsort(xt))
          yt_argsort = list(np.argsort(yt))
          # indexing
          ldx = []
          for ildx in range(4):
            ldx.append(yt_argsort.index(xt_argsort[ildx]))
          all_types = [[1,2,3,4],[1,2,4,3],[1,3,2,4],[1,3,4,2],[1,4,2,3],[1,4,3,2],\
                        [2,1,3,4],[2,1,4,3],[2,3,1,4],[2,3,4,1],[2,4,1,3],[2,4,3,1],\
                        [3,1,2,4],[3,1,4,2],[3,2,1,4],[3,2,4,1],[3,4,1,2],[3,4,2,1],\
                        [4,1,2,3],[4,1,3,2],[4,2,1,3],[4,2,3,1],[4,3,1,2],[4,3,2,1]]
          all_types = [[all_types[iat][0]-1,all_types[iat][1]-1,all_types[iat][2]-1,all_types[iat][3]-1] for iat in range(24)]
          match_type = all_types.index(ldx)
    
          half_x = (xt_sort + mean_x) / 2
          half_y = (yt_sort + mean_y) / 2
    
          # add key_point
          keypoints = []
          keypoints.append(mean_x)
          keypoints.append(mean_y)
          keypoints.append(2)
          for i in range(4):
            keypoints.append(half_x[i])
            keypoints.append(mean_y)
            keypoints.append(2)
          for i in range(4):
            keypoints.append(mean_x)
            keypoints.append(half_y[i])
            keypoints.append(2)
          try:
            keypoints.append(int(p_l1l2.x))
            keypoints.append(int(p_l1l2.y))
            keypoints.append(2)
          except Exception as e:
            print(e)
            # print('EIntersection found. This bounding is removing ...')
            continue
          ''' indexing gt
          
          '''
          dataset['annotations'].append({
              'area': width * height,
              'bbox': [xmin, ymin, width, height],
              'category_id': get_category_id(cls),
              'id': j,
              'image_id': int(index),
              'iscrowd': 0,
              'segmentation': [segs],
              # 'segmentation_shrink': [segs_shrink],
              'keypoints': keypoints,
              'match_type': match_type
          })
          j += 1
    folder = os.path.join(root_path, 'annotations')
    if not os.path.exists(folder):
      os.makedirs(folder)
    json_name = os.path.join(root_path, 'annotations/{}.json'.format(phase))
    with open(json_name, 'w') as f:
      json.dump(dataset, f)
