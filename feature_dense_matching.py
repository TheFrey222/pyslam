"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import config
config.cfg.set_lib('densematching',prepend=True)

import sys
import os
import cv2 
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.cm as cm
from pathlib import Path
import numpy as np

from threading import RLock
from utils_sys import Printer, is_opencv_version_greater_equal

from DenseMatching.model_selection import model_type, pre_trained_model_types, select_model
from DenseMatching.datasets.util import pad_to_same_shape
from DenseMatching.admin.stats import DotDict 

torch.set_grad_enabled(False)

kVerbose = True   


class DenseMatchingOptions:
    def __init__(self, do_cuda=True):
        # default options from demo_single_pair.ipynb
        self.model = 'PDCNet_plus'
        self.pre_trained_model = 'megadepth'
        flipping_condition = False 
        self.global_optim_iter = 3
        self.local_optim_iter = 7 
        self.estimate_uncertainty = True 
        self.confident_mask_type = "proba_interval_1_above_10"
        self.path_to_pre_trained_models = config.cfg.root_folder + '/thirdparty/DenseMatching/pre_trained_models/'
        self.reference_image = config.cfg.root_folder + "/images/frame0000.jpg"

        if self.model not in model_type:
            raise ValueError('The model that you chose is not valid: {}'.format(self.model))
        if self.pre_trained_model not in pre_trained_model_types:
            raise ValueError('The pre-trained model type that you chose is not valid: {}'.format(self.pre_trained_model))


        # inference parameters for PDC-Net
        network_type = self.model  # will only use these arguments if the network_type is 'PDCNet' or 'PDCNet_plus'
        choices_for_multi_stage_types = ['d', 'h', 'ms']
        multi_stage_type = 'h'
        if multi_stage_type not in choices_for_multi_stage_types:
            raise ValueError('The inference mode that you chose is not valid: {}'.format(multi_stage_type))

        confidence_map_R =1.0
        ransac_thresh = 1.0
        mask_type = 'proba_interval_1_above_10'  # for internal homo estimation
        homography_visibility_mask = True
        scaling_factors = [0.5, 0.6, 0.88, 1, 1.33, 1.66, 2]
        compute_cyclic_consistency_error = True  # here to compare multiple uncertainty 

        # usually from argparse
        self.args = DotDict({'network_type': network_type, 'multi_stage_type': multi_stage_type, 'confidence_map_R': confidence_map_R, 
                        'ransac_thresh': ransac_thresh, 'mask_type': mask_type, 
                        'homography_visibility_mask': homography_visibility_mask, 'scaling_factors': scaling_factors, 
                        'compute_cyclic_consistency_error': compute_cyclic_consistency_error})

        use_cuda = torch.cuda.is_available() & do_cuda
        device = torch.device('cuda' if use_cuda else 'cpu')
        print('SuperPoint using ', device)        
        self.cuda=use_cuda 


# convert matrix of pts into list of keypoints
# N.B.: pts are - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
def convert_densematching_to_keypoints(pts, size=1): 
    kps = []
    if pts is not None: 
        # convert matrix [Nx2] of pts into list of keypoints  
        if is_opencv_version_greater_equal(4,5,3):
            kps = [ cv2.KeyPoint(p[0], p[1], size=size, response=p[2]) for p in pts ]            
        else: 
            kps = [ cv2.KeyPoint(p[0], p[1], _size=size, _response=p[2]) for p in pts ]                      
    return kps         


def transpose_des(des):
    if des is not None: 
        return des.T 
    else: 
        return None 


# interface for pySLAM 
class DenseMatchingFeature2D: 
    def __init__(self, do_cuda=True): 
        self.lock = RLock()
        self.opts = DenseMatchingOptions(do_cuda)
        print(self.opts)
        
        print('SuperPointFeature2D')
        print('==> Loading pre-trained network.')
        # This class runs the SuperPoint network and processes its outputs.
        # define network and load network weights
        self.network, self.estimate_uncertainty = select_model(self.opts.model, 
                                                               self.opts.pre_trained_model, 
                                                               self.opts.args, 
                                                               self.opts.global_optim_iter, 
                                                               self.opts.local_optim_iter, 
                                                               path_to_pre_trained_models=self.opts.path_to_pre_trained_models) 
        print('==> Successfully loaded pre-trained network.')

        self.pts = []
        self.kps = []        
        self.des = []
        self.heatmap = [] 
        self.query_image = None
        self.keypoint_size = 20  # just a representative size for visualization and in order to convert extracted points to cv2.KeyPoint 
          
    # compute both keypoints and descriptors       
    def detectAndCompute(self, frame, mask=None):  # mask is a fake input 
        with self.lock: 
            if type(self.opts.reference_image) == str:
                self.opts.reference_image = frame
            self.query_image, self.reference_image = pad_to_same_shape(frame, self.opts.reference_image)

            query_image_ = torch.from_numpy(self.query_image).permute(2, 0, 1).unsqueeze(0)
            reference_image_ = torch.from_numpy(self.reference_image).permute(2, 0, 1).unsqueeze(0)

            pred = self.network.get_matches_and_confidence(target_img=query_image_, 
                                                           source_img=reference_image_, 
                                                           confident_mask_type=self.opts.confident_mask_type)

            self.pts = pred['kp_source']
            mkpts_ref = pred['kp_target']
            confidence_values = pred['confidence_value']

            print('Found {} confident matches'.format(len(self.pts)))

            sort_index = np.argsort(np.array(confidence_values)).tolist()[::-1]  # from highest to smallest
            confidence_values = np.array(confidence_values)[sort_index]
            self.pts = np.array(self.pts)[sort_index]
            mkpts_ref = np.array(mkpts_ref)[sort_index]

            if len(self.pts) < 5:
                self.pts = np.empty([0, 2], dtype=np.float32)
                mkpts_ref = np.empty([0, 2], dtype=np.float32)
                confidence_values = np.empty([0], dtype=np.float32)
                
            # plot top 10000
            k_top = 10000
            self.pts = self.pts[:k_top]
            mkpts_ref = mkpts_ref[:k_top]
            confidence_values = confidence_values[:k_top]

            # N.B.: pts are - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
            Printer.cyan('pts: ', self.pts)
            Printer.cyan('type: ', type(self.pts))
            Printer.cyan('shape: ', self.pts.shape)
            Printer.cyan('conf: ', confidence_values)
            Printer.cyan('type: ', type(confidence_values))
            Printer.cyan('shape: ', confidence_values.shape)
            
            np.append(self.pts, confidence_values, axis=1)
            
            Printer.cyan('pts: ', self.pts)
            Printer.cyan('type: ', type(self.pts))
            Printer.cyan('shape: ', self.pts.shape)
            
            try:
                self.kps = convert_densematching_to_keypoints(self.pts.T, size=self.keypoint_size)
            except:
                return
                
            if kVerbose:
                print('detector: DenseMatching, #features: ', len(self.kps), ', frame res: ', frame.shape[0:2])      
            return self.kps, transpose_des(self.des)                 
            
    # return keypoints if available otherwise call detectAndCompute()    
    def detect(self, frame, mask=None):  # mask is a fake input  
        with self.lock:         
            #if self.frame is not frame:
            self.detectAndCompute(frame)        
            return self.kps
    
    # return descriptors if available otherwise call detectAndCompute()  
    def compute(self, frame, kps=None, mask=None): # kps is a fake input, mask is a fake input
        with self.lock: 
            if self.frame is not frame:
                Printer.orange('WARNING: SUPERPOINT is recomputing both kps and des on last input frame', frame.shape)
                self.detectAndCompute(frame)
            return self.kps, transpose_des(self.des)