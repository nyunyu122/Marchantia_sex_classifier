# -*- coding: utf-8 -*-
"""utils.ipynb
"""
import numpy as np

import torch

import saliency.core as saliency




def min_max_normalization(array_2d: np.ndarray):
  """:input: np.ndarray.shape = (W, H). The grayscale mask.
  """
  normalized = (array_2d - np.min(array_2d))/(np.max(array_2d)-np.min(array_2d))
  return np.float32(normalized)

def calculate_xrai(input, label, xrai_object, call_model_function, 
                    xrai_batch_size=20):
    '''
    ::parameters::
    :input:np.ndarray.shape=(3,244,244)), inputs_test[idx]
    :label:int, 0 or 1 in Marchantia models. The label is converted to call_model_args.
    :call_model_function:
    :xrai:
    :batch_size:20. batch size in computing xrai's IG.
    ::return::
    :xrai_attributions:np.ndarray.shape=(W,H) of np.float32, grayscale mask of the xrai attributions. The mask is min-max-normalized.
    '''
    # xrai_attributionの時点では正規化されていない。min_max_normalizationをかける。
    class_idx_str = 'class_idx_str'
    call_model_args = {class_idx_str: label}
    xrai_attributions = xrai_object.GetMask(input,
                                            call_model_function=call_model_function,
                                            call_model_args=call_model_args,
                                            batch_size=xrai_batch_size)
    xrai_attributions = min_max_normalization(xrai_attributions)
    return xrai_attributions
