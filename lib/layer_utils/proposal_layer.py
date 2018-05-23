# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes, bbox_transform_inv_tf, clip_boxes_tf
from model.nms_wrapper import nms

def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  """A simplified version compared to fast/er RCNN
     For details please see the technical report
  """
  if type(cfg_key) == bytes:
      cfg_key = cfg_key.decode('utf-8')
  #12000 anchors pre nms algorithm are applied!
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  #2000 anchors are generated from one pic which are generated after nms
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  #the threshold is set to 0.7 to delete abundant anchors!
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
  scores = scores.reshape((-1, 1))
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  #
  proposals = clip_boxes(proposals, im_info[:2])

  # Pick the top region proposals
  order = scores.ravel().argsort()[::-1]
  if pre_nms_topN > 0:
    order = order[:pre_nms_topN]
  proposals = proposals[order, :]
  scores = scores[order]

  # Non-maximal suppression
  keep = nms(np.hstack((proposals, scores)), nms_thresh)

  # Pick th top region proposals after NMS
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep]

  # Only support single image as input
  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
  blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

  return blob, scores


def proposal_layer_tf(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  if type(cfg_key) == bytes:
    cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  scores = tf.reshape(scores, shape=(-1,))
  rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))
  
  ## to anchors we apply bbox_transform operation,and we make can calculate (tx*,ty*,tw*,th*),
  ## and return the pred four coordinates(x1,y1,x2,y2)
  proposals = bbox_transform_inv_tf(anchors, rpn_bbox_pred)
  proposals = clip_boxes_tf(proposals, im_info[:2])

  ## because the proposals are too much to choose,so we apply non-maximal suppression,and we can get about 2000 proposals in training
  ## apply non-maximal suppression to the proposals
  # Non-maximal suppression
  indices = tf.image.non_max_suppression(proposals, scores, max_output_size=post_nms_topN, iou_threshold=nms_thresh)

  ## now we get the proposals after nms operation,and we can get proposals scores and bbox_preds,
  ## scores are predicted scores,not label
  boxes = tf.gather(proposals, indices)
  boxes = tf.to_float(boxes)
  scores = tf.gather(scores, indices)
  scores = tf.reshape(scores, shape=(-1, 1))

  # Only support single image as input
  batch_inds = tf.zeros((tf.shape(indices)[0], 1), dtype=tf.float32)
  blob = tf.concat([batch_inds, boxes], 1)

  return blob, scores


