# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

#注意given a roidb，构建一个minibatch，这说明了rpn中的mini_batch并不是类似于以往中的batch images，
#而是从一张图片上面得到的多个proposals中构建批进行训练rpn网络，其中从所有的proposals中选出iou>0.7的128个proposals作为正样本，
#iou<0.3的proposals中挑选出128个proposals作为负样本进行训练，获取到bounding box regerssion所需要的(dx,dy,dw,dh)，
#注意dx，dy，dw，dh代表了什么和如何训练得到的，需要注意一下，
#求得了这些参数之后，我们用它们做什么呢？


#然后我们结合这些参数和原来的rois以及feature maps，这下我们可以得到proposal feature maps（注意这些都是foreground），
#这个过程在https://zhuanlan.zhihu.com/p/31426458有详细介绍，称为Proposal，

#由于得到的proposal feature maps的大小不一，而fast rcnn网络中由于有全连接层的存在，因此需要固定的feature maps，
#因此在将这些feature maps送入训练之前需要调整成统一大小，这儿采用的方式是ROI pooling操作，更详细的资料可以参考https://zhuanlan.zhihu.com/p/30343330

#然后进行训练fast-RCNN网络


#https://blog.csdn.net/zziahgf/article/details/78695868
#for more information you can look this page:https://blog.csdn.net/sloanqin/article/details/51611747
#传入get_minibatch中的roidb其实是[roidb[i]]，即第几张图片中的{}（含有5个key的dict)数据所组成的含有一个元素的list
def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  #现在我们要从里面进行取样，num_image=1
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  # the thing you should notice is im_bloc is just one pic data
  im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

  blobs = {'data': im_blob}

  #look at this place,singe batch only,that is to say,mini-batch you just choose one picture to train rpn network
  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"
  
  # gt boxes: (x1, y1, x2, y2, cls)
  if cfg.TRAIN.USE_ALL_GT:
    # Include all ground truth boxes
    # choose all gt_classes is not bg,that is to say,you choose all foreground flags to train the rpn net!
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
    gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
  #here you get all gt_boxes which shape is (len(gt_inds),5),so we can train
  gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
  blobs['gt_boxes'] = gt_boxes
  blobs['im_info'] = np.array(
    [im_blob.shape[1], im_blob.shape[2], im_scales[0]],
    dtype=np.float32)

  return blobs

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  for i in range(num_images):
    im = cv2.imread(roidb[i]['image'])
    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    
    #prep_im_for_blob,this is preprocess data for train,the thing you must notice is the im is just one pic
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  # yeah,you konw,the blob is just one,im_list_to_blob,because the precessed_ims is just one
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales
