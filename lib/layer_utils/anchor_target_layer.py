# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from model.config import cfg
import numpy as np
import numpy.random as npr
from utils.cython_bbox import bbox_overlaps
from model.bbox_transform import bbox_transform

## in this place we judge anchors' label,that is to say,we get a mini_batch's anchor with label(256)
## rpn_cls_score's shape is (1,57,38,18)
def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
  """Same as the anchor target layer in original Fast/er RCNN """
  A = num_anchors
  total_anchors = all_anchors.shape[0]
  K = total_anchors / num_anchors

  # allow boxes to sit over the edge by a small amount
  _allowed_border = 0

  # map of shape (..., H, W)
  ## get the shape of height and width
  height, width = rpn_cls_score.shape[1:3]

  # only keep anchors inside the image
  ## inds_inside represents the anchors in the picture
  ## 这样的话我们就可以删除掉很多在不在图片中的大量anchors，可以得到2400左右的anchors
  inds_inside = np.where(
    (all_anchors[:, 0] >= -_allowed_border) &
    (all_anchors[:, 1] >= -_allowed_border) &
    (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
    (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
  )[0]

  # keep only inside anchors
  anchors = all_anchors[inds_inside, :]

  # label: 1 is positive, 0 is negative, -1 is dont care
  ## now we judge the label is positive or negative by iou
  #labels shape are (< 1*58*37*9)
  labels = np.empty((len(inds_inside),), dtype=np.float32)
  labels.fill(-1)

  # overlaps between the anchors and the gt boxes
  # overlaps (ex, gt)
  ## after looking at cython.bbox的bbox_overlap之后，我们可以简单的发现，
  ## 这里做的处理非常巧妙：
  ## (58*37*9,4) shape的bbox_anchors,
  ## (len(gt_boxes),4) shape的gt_boxes，
  ## bbox_overlaps的返回值是(58*37*9,len(gt_boxes))，也就是说，对每一个anchors都计算了它距离其他gt_boxes的重叠度
  overlaps = bbox_overlaps(
    np.ascontiguousarray(anchors, dtype=np.float),
    np.ascontiguousarray(gt_boxes, dtype=np.float))
  ## 通过argmax我们选择针对每个bbox_anchor，与它重叠度最高的作为它的预回归位置
  argmax_overlaps = overlaps.argmax(axis=1)
  max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
  gt_argmax_overlaps = overlaps.argmax(axis=0)
  gt_max_overlaps = overlaps[gt_argmax_overlaps,
                             np.arange(overlaps.shape[1])]
  gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

  if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
    # assign bg labels first so that positive labels can clobber them
    # first set the negatives
    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

  ## the two steria we choose for positive anchors 
  #1) fg label: for each gt, anchor with highest overlap
  labels[gt_argmax_overlaps] = 1

  #2)fg label: above threshold IOU
  labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

  if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
    # assign bg labels last so that negative labels can clobber positives
    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

  # subsample positive labels if we have too many
  
  ## positive anchors' count,now it is 256*0.5=128
  num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
  fg_inds = np.where(labels == 1)[0]
  ## if positive anchors is too many,we drop others rois! 
  if len(fg_inds) > num_fg:
    disable_inds = npr.choice(
      fg_inds, size=(len(fg_inds) - num_fg), replace=False)
    labels[disable_inds] = -1
  ## but if positive anchors is less than 128,how do we do to satisfy the mini_batch train???
  ## answer is below

  # subsample negative labels if we have too many
  ## if positive anchors is 128,we choose other negative anchors;but if positive anchors are less than 128,we choose other negatives to fit this!!!
  num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
  bg_inds = np.where(labels == 0)[0]
  ## if the negative anchors are too many,we drop others!!!
  if len(bg_inds) > num_bg:
    disable_inds = npr.choice(
      bg_inds, size=(len(bg_inds) - num_bg), replace=False)
    labels[disable_inds] = -1

  
  ## compute (tx,ty,tw,th)
  bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
  bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

  bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
  # only the positive ones have regression targets
  bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

  bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
  if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
    # uniform weighting of examples (given non-uniform sampling)
    num_examples = np.sum(labels >= 0)
    positive_weights = np.ones((1, 4)) * 1.0 / num_examples
    negative_weights = np.ones((1, 4)) * 1.0 / num_examples
  else:
    assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
            (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
    positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                        np.sum(labels == 1))
    negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                        np.sum(labels == 0))
  bbox_outside_weights[labels == 1, :] = positive_weights
  bbox_outside_weights[labels == 0, :] = negative_weights

  # map up to original set of anchors
  ## 为了将shape恢复为（58*37*9,）和（58*37*9,4）,
  ## 我们采用了_unmap操作，用于进行cls_loss和bbox_loss
  ## 注意labels填充的是-1,
  ## 而bbox_targets填充的是0，代表如果不在图片中的话，anchor不用进行回归，只对在图片中的anchors进行回归！！！
  labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
  bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
  bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
  bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

  # labels
  labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
  labels = labels.reshape((1, 1, A * height, width))
  rpn_labels = labels

  # bbox_targets
  bbox_targets = bbox_targets \
    .reshape((1, height, width, A * 4))

  rpn_bbox_targets = bbox_targets
  # bbox_inside_weights
  bbox_inside_weights = bbox_inside_weights \
    .reshape((1, height, width, A * 4))

  rpn_bbox_inside_weights = bbox_inside_weights

  # bbox_outside_weights
  bbox_outside_weights = bbox_outside_weights \
    .reshape((1, height, width, A * 4))

  rpn_bbox_outside_weights = bbox_outside_weights
  return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
  """ Unmap a subset of item (data) back to the original set of items (of
  size count) """
  if len(data.shape) == 1:
    ret = np.empty((count,), dtype=np.float32)
    ret.fill(fill)
    ret[inds] = data
  else:
    ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
    ret.fill(fill)
    ret[inds, :] = data
  return ret


def _compute_targets(ex_rois, gt_rois):
  """Compute bounding-box regression targets for an image."""

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 5

  return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
