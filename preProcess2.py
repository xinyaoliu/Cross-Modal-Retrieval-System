#!/usr/bin/python
#-*- coding:utf-8 -*-

"""
make classification dataset.

cropping annotated region and output to a directory.

dir_out/
  1/
    xxxx.jpg
    xxxx.jpg
  2/
    xxxx.jpg
  ...
"""

import os
import cv2
import argparse
import numpy as np

from pycocotools.coco import COCO


dataDir = "/home/liuxinyao/dataset/coco/2014/images//val2014"
annFile = "/home/liuxinyao/dataset/coco/2014/annotations/instances_val2014.json"

# dataDir = "/home/liuxinyao/dataset/coco/2014/images/train2014"
# annFile = "/home/liuxinyao/dataset/coco/2014/annotations/instances_train2014.json"

def get_image_path(coco, image_id):
    img = coco.loadImgs([image_id])[0]
    return os.path.join(dataDir, img["file_name"])    

def outputCategoryImage(coco, cat_id, out_dir):
    """
    output images which have specified category id

    Args:
      out_dir : output directory
    """
    annIds = coco.getAnnIds(catIds=[cat_id], areaRng=[5000,np.inf],iscrowd=False)
    anns = coco.loadAnns(annIds)    

    """
{u'segmentation': [[abbreviate]], u'area': 11941.99525, u'iscrowd': 0, u'image_id': 449560, u'bbox': [176.77, 345.81, 103.23, 165.16], u'category_id': 18, u'id': 8526}
    """
    
    for ann in anns:
        cur_in = get_image_path(coco, ann["image_id"])
        # print (cur_in)
        image = cv2.imread(cur_in)

        # bbox = np.array(ann["bbox"]).astype(int)
        # region = image[0:224, 0:224]

        cur_out = os.path.join(out_dir, str(ann["id"]) + ".jpg")

        cv2.imwrite(cur_out, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", dest="dir_out", required=True)
    args = parser.parse_args()
    dir_out = args.dir_out

    if not os.path.exists(dir_out):
        os.mkdir(dir_out)

    # ground truth
    coco = COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat["name"] for cat in cats]
    print "COCO categories:\n\n",','.join(nms)

    #counts = {}
    for cat in cats:
        #print "process {} id={}".format(cat["name"], cat["id"])
        cur_id = cat["id"]
        cur_dir = os.path.join(dir_out, str(cur_id))

        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)

        outputCategoryImage(coco, cur_id, cur_dir)
        #counts[cur_id] = cur_counts

    """
    for cur_id, cur_count in counts.items():
        if cur_count > 200:
            print "{}".format(cur_id)
    """

