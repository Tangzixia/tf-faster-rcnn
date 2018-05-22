# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import time

class RoIDataLayer(object):
  """Fast R-CNN data layer used for training."""

  def __init__(self, roidb, num_classes, random=False):
    """Set the roidb to be used by this layer during training."""
    self._roidb = roidb
    self._num_classes = num_classes
    # Also set a random flag
    self._random = random
    self._shuffle_roidb_inds()

  def _shuffle_roidb_inds(self):
    """Randomly permute the training roidb."""
    # If the random flag is set, 
    # then the database is shuffled according to system time
    # Useful for the validation set
    if self._random:
      st0 = np.random.get_state()
      millis = int(round(time.time() * 1000)) % 4294967295
      np.random.seed(millis)
    
    if cfg.TRAIN.ASPECT_GROUPING:
      #获取图片的宽高
      widths = np.array([r['width'] for r in self._roidb])
      heights = np.array([r['height'] for r in self._roidb])
      horz = (widths >= heights)
      vert = np.logical_not(horz)
      horz_inds = np.where(horz)[0]
      vert_inds = np.where(vert)[0]
      #因为前面的horz和vert互为logical_not，因此进行np.where操作之后两者的和仍为5011
      #因为hstack，所以inds的shape变为(10022，)
      inds = np.hstack((
          np.random.permutation(horz_inds),
          np.random.permutation(vert_inds)))
      #转化inds的shape为(5011,2)
      inds = np.reshape(inds, (-1, 2))
      #随机扰乱，shape为(5011,2)
      row_perm = np.random.permutation(np.arange(inds.shape[0]))
      #reshape之后inds的shape为(10022,)
      inds = np.reshape(inds[row_perm, :], (-1,))
      #self._perm的shape为(10022,)
      self._perm = inds
    else:
      #self._perm的shape为（5011，）
      self._perm = np.random.permutation(np.arange(len(self._roidb)))
    # Restore the random state
    if self._random:
      np.random.set_state(st0)
      
    self._cur = 0

  def _get_next_minibatch_inds(self):
    """Return the roidb indices for the next minibatch."""
    #判断当前的标签有没有超过边界
    if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
      self._shuffle_roidb_inds()
    #获取db_inds，即下一次batch的标签，获取一张图片
    db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
    #同时让当前标签前移
    self._cur += cfg.TRAIN.IMS_PER_BATCH

    return db_inds

  def _get_next_minibatch(self):
    """Return the blobs to be used for the next minibatch.

    If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
    separate process and made available through self._blob_queue.
    """
    db_inds = self._get_next_minibatch_inds()
    minibatch_db = [self._roidb[i] for i in db_inds]
    #注意minibatch_db中只有一个self._roidb[i],例如i为1000，则self._roidb[1000]
    #而roidb[1000]中是一个有5个key的dict，（boxes,max_overlap,...）等等，所以训练rpn网络的minibatch都是从一张图片中进行选择
    #下面我们来看具体操作！！！
    return get_minibatch(minibatch_db, self._num_classes)
      
  def forward(self):
    """Get blobs and copy them into this layer's top blob vector."""
    blobs = self._get_next_minibatch()
    return blobs
