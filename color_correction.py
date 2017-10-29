#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
A color correction method that balances the color histograms of an image based
on a user defined threshold ("percent_correct"), and then adjusts the gamma
and applies a bilateral filter.
"""
# Now create a function to fix the color profiles
def fix_color(image, percentile_correction):
    import numpy as np
    from skimage.exposure import adjust_gamma
    import cv2
    import math

    ##############################################################
    ### Simple Color Balance function ported from:             ###
    ### https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc ###
    ##############################################################
    
    def apply_mask(matrix, mask, fill_value):
        masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
        return masked.filled()

    def apply_threshold(matrix, low_value, high_value):
        low_mask = matrix < low_value
        matrix = apply_mask(matrix, low_mask, low_value)

        high_mask = matrix > high_value
        matrix = apply_mask(matrix, high_mask, high_value)

        return matrix

    def simplest_cb(image, percentile_correction):
        # make sure it's an RBG image
        assert image.shape[2] == 3
        
        # make sure the percentile correction is between 0 and 100
        assert percentile_correction > 0 and percentile_correction < 100

        # split the percentile correction into two
        half_percent = percentile_correction / 200.0

        # split the image into Red Green and Blue channels
        channels = cv2.split(image)

        out_channels = []
        for channel in channels:
            assert len(channel.shape) == 2
            # find the low and high precentile values (based on the 
            # input percentile)
            height, width = channel.shape
            vec_size = width * height
            flat = channel.reshape(vec_size)

            assert len(flat.shape) == 1

            flat = np.sort(flat)

            n_cols = flat.shape[0]

            low_val  = flat[int(math.floor(n_cols * half_percent))]
            high_val = flat[int(math.ceil( n_cols * (1.0 - half_percent)))]

            # saturate below the low percentile and above the high percentile
            thresholded = apply_threshold(channel, low_val, high_val)
            # scale the channel
            normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
            out_channels.append(normalized)

        return cv2.merge(out_channels)
    
    ########################################################################
    ### Balance the image color, adjust gamma, and then filter the image ###
    ########################################################################
    
    # implement the simple color balance
    new_img = simplest_cb(image, percentile_correction)

    # adjust the gamma to help boost the color profile
    new_img = adjust_gamma(new_img, gamma=0.75, gain=1)
    
    # use a bilateral filter, which is effective at noise removal 
    # while preserving edges
    new_img = cv2.bilateralFilter(new_img, 5, 75, 75)
    
    return new_img



