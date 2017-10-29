#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
A function to resize an image based on a given width
"""

def resize_image(image, new_basewidth):
    from cv2 import resize, INTER_AREA

    # turn the user defined new image width into a percent of the current width
    basewidth_percent = (new_basewidth/
                         float(image.shape[0]))

    # determine the new height based on the new width
    new_height = int((float(image.shape[1])*
                      float(basewidth_percent)))

    # resize the image bsed on the new width and height
    resized_image = resize(image,
                       dsize=(new_height,new_basewidth),
                       interpolation = INTER_AREA)
    
    return resized_image
