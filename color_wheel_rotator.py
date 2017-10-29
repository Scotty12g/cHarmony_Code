#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
A function to take a Red Green Blue color profile, rotate a virtual
colorwheel, and return the corresponding Red Green Blue color profile at that
degree rotation.
"""
# function altered from original code at:
# https://stackoverflow.com/questions/14095849/calculating-the-analogous-color-with-python

def rotate_colors((red, green, blue), degreee_rotation):

    from colorsys import rgb_to_hls, hls_to_rgb
    
    # turn the degree rotation into a percent rotation
    degreee_rotation = degreee_rotation/360.0
    
    # turn the red green and blue values into percent of maximum (255)
    red, green, blue = map(lambda x: x/255., [red, green, blue])

    # convert Red Green Blue profiles into hue lightness saturation profiles
    color_hue, color_lightness, color_saturation = rgb_to_hls(red,
                                                              green,
                                                              blue)
    
    # Rotate the color by the specified degree in either direction and store
    # the new values in 'hues'
    hues = [(color_hue+degree) % 1 for degree in (-degreee_rotation, 
            degreee_rotation)]

    # convert new hue lightness saturation profoles back to Red Green Blue
    # profiles
    rotated_color = [map(lambda x: int(round(x*255)), hls_to_rgb(hue, 
                   color_lightness, color_saturation)) for hue in hues]

    return rotated_color