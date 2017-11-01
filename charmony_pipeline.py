#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
An image data pipeline that takes a user defined method of color matching
('complement' (180deg) or 'triad' (120deg)) and two images (item of clothing,
closet). The pipeline returns the color of the item of clothing, the
corresponding matching colors, and the closet image with the matching colors
highlighted and non-mtching colors greyed out.
"""

def charmony_run(color_matching_method, clothing_image_path, closet_image_path):

    # Import the packages we'll need
    import numpy as np
    
    from PIL import Image
    from skimage.segmentation import slic
    from cv2 import COLOR_BGR2RGB, imread, cvtColor
    
    from sklearn.neighbors import KNeighborsClassifier

    from pickler import Pickler
    from color_correction import fix_color
    from color_wheel_rotator import rotate_colors
    from mode_function import get_mode
    from resize_image import resize_image
 
    ################################################################
    ### IMPORTING USER DEFINED VALUES AND PREPARING LOOKUP DICTS ###
    ################################################################
    
    # unpickling the model
    color_detector_pickled = open('color_detector.pkl', 'r')
    color_detector = Pickler.load_pickle(color_detector_pickled)
    color_detector_pickled.close()
    
    # define the Red Green Blue profiles centers for each possible color
    color_dict = {'red':(255, 0, 0),
                  'yellow' : (255, 255, 0),
                  'green' : (0, 255, 0),
                  'cyan' : (0, 255, 255),
                  'blue' : (0, 0, 255),
                  'magenta' : (255, 0, 255)}
    
        
    # import user defined "How to match the color": 'complement' (180deg) or 
    # 'triad' (120deg)
    how_to_match_colors = color_matching_method
    
    # define what degree rotation corresponds to each method of matching colors
    match_rotation_dict = {'complement': 180,
                              'triad': 120}
    
    # import the image path of the clothing you'd like to match with
    image_to_match_to_path = clothing_image_path

    # import the image path of the mix of clothing to match to
    image_of_possible_matches_path = closet_image_path


    
    ###################################
    ### LOADING AND CLEANING IMAGES ###
    ###################################
    
    # load image from image_to_match_to_path and convert to RBG from BGR
    image_to_match_to = imread(image_to_match_to_path)
    
    image_to_match_to = cvtColor(image_to_match_to, 
                                     COLOR_BGR2RGB)

    # Reduce the image size by defining a width, and correct the color balance
    # by defining the low and high precentile value
    image_to_match_to = resize_image(image_to_match_to,
                                     new_basewidth = 600)

    image_to_match_to = fix_color(image_to_match_to,
                                  percentile_correction = 10)

    # load image from image_of_possible_matches_path and 
    # convert to RBG from BGR
    image_of_possible_matches = imread(image_of_possible_matches_path)
    
    image_of_possible_matches = cvtColor(image_of_possible_matches, 
                                             COLOR_BGR2RGB)

    # Reduce the image size by defining a width, and correct the color balance
    # by defining the low and high precentile value
    image_of_possible_matches = resize_image(image_of_possible_matches,
                                             new_basewidth = 600)
    
    image_of_possible_matches=fix_color(image_of_possible_matches,
                                        percentile_correction = 10)
    

    ##########################################################
    ### DETNERMINING THE COLOR OF THE CLOTHING TO MATCH TO ###
    ##########################################################
    
    # Determine the pixel boundaries of image_to_match_to
    max_dim1 = image_to_match_to.shape[0]
    max_dim2 = image_to_match_to.shape[1]
    max_dim1_percent = max_dim1/100.0
    max_dim2_percent = max_dim2/100.0

    # Use the aformentioned boundaries, and select 400 random points from 
    # the center of the image
    dim1_coords = (np.random.choice(np.arange(max_dim1_percent*40,
                                        max_dim1_percent*60,1),400)).astype(int)
    dim2_coords = (np.random.choice(np.arange(max_dim2_percent*40,
                                        max_dim2_percent*60,1),400)).astype(int)
    
    
    # Itterate through each rrandom point and predict the color
    coords_colors = []
    for coord in np.arange(len(dim1_coords)):
        coord_color = color_detector.predict(
                np.array(image_to_match_to[dim1_coords[coord],
                                           dim2_coords[coord],:]).reshape(1,-1))
    
        coords_colors.append(coord_color)

    coords_colors = np.vstack(coords_colors).flatten()

    # find the most frequent color among all the random points, and assign that
    # to image_to_match_to_color
    most_frequent_color = get_mode(coords_colors)
    image_to_match_to_color = most_frequent_color


    ###################################################
    ### DETERMINING THE APPROPRIATE MATCHING COLOR  ###
    ### Note: only supports matching to one color   ###
    ###################################################
    
    # If image_to_match_to_color is a neutral color, return all possible
    # non-neutral colors
    if image_to_match_to_color in ['black', 'brown', 'grey', 'white']:
        matching_colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta']
        
     # else, use a function to rotate the color wheel, based on the current
     # Red Green Blue profile (looked up in color_dict) and the degree rotaiton
     # from how_to_match_colors (looked up in match_rotation_dict)
    else:
        color_rbg_profile = color_dict[image_to_match_to_color]
        matching_rotation = match_rotation_dict[how_to_match_colors]
        matching_color_list = rotate_colors(color_rbg_profile,
                                            matching_rotation)
        
        # ensure the matching_color_list outputs (lists formatted as 
        # Red Green Blue profiles) are all unique
        unique_matching_color_values = list()
        for sublist in matching_color_list:
            if sublist not in unique_matching_color_values:
                unique_matching_color_values.append(sublist)

        # convert tuples in Red Green Blue color format back to color
        # names, and store the color names in matching_colors
        matching_colors = [color for color in color_dict if color_dict[color] 
        in tuple(tuple(rbg) for rbg in unique_matching_color_values)]
 
    
    #################################################################
    ### SEGMENT THE IMAGE OF MULTIPLE GARMENTS IN A CLOSET IMAGE  ###
    #################################################################

    ## Use Simple Linear Itterative Clustering to segment the closet image
    segmented_possible_matches = slic(image_of_possible_matches,
                         n_segments=350,
                         compactness=10,
                         sigma=1)
    
    # Get the unique segment labels
    unique_segments = np.unique(segmented_possible_matches)
    
    # randomly select 1/5 of the pixel data within each segment to summarize 
    # color detection over. First, create a list to hold each segments' data
    unique_segments_summary = []
    
    for segment in unique_segments:
        # get the pixle by pixel Red Green Blue data for a segment
        segment_pixels_data = image_of_possible_matches[
                segmented_possible_matches == segment]
        
        # randomly choose 1/5 of the pixel data for that segment
        pixel_count = len(segment_pixels_data)
        pixel_subset = np.random.choice(np.arange(pixel_count),
                                        (pixel_count/5),
                                        replace=False)
        segment_pixels_data_subset = [segment_pixels_data[pixel] for pixel in 
                                      pixel_subset]
        
        # place the subset data into unique_segments_summary
        unique_segments_summary.append(segment_pixels_data_subset)

    #################################################################
    ### DETERMINE THE COLOR FOR EACH SEGMENT IN THE CLOSET IMAGE  ###
    #################################################################
    
    # Using the subset of pixel data for each segment, itterate over each 
    # segment and predict its color. Keep track of each segment's color in
    # the list unique_segments_colors
    unique_segments_colors=[]
    
    for segment_data in unique_segments_summary:
        # if the segment data has only one dimension then...
        if len(np.array(segment_data).shape)==1:
            # if there is no data in it, just return no data
            if np.array(segment_data).size == 0:
                segment_data
            # if there is data in it, reshape the array and detect the color
            else:
                colors_seen=color_detector.predict(np.array(
                        segment_data).reshape(1, -1))
        # if the data is greater than 1 dimension, detect the color as usual
        else:
            colors_seen=color_detector.predict(segment_data)

        # find the most fre3quent color detected within the segment, and record
        # the color in unique_segments_colors
        most_frequent_color = get_mode(colors_seen)
        unique_segments_colors.append(most_frequent_color)

    ########################################################################
    ### FIND THE SEGMENTS IN THE CLOSET THAT MATCH THE CLOTHING IN COLOR ###
    ########################################################################
    
    # create a boolean to identify which segemnts are color matches
    which_segments_to_highlight = np.reshape([(val in matching_colors) for 
                                         val in unique_segments_colors],
                                         np.array(unique_segments_colors).shape)
    
    # create a list of the matching segments, that will eventually be used
    # to highlight the correct portions of the image
    unique_segments_to_highlight = unique_segments[which_segments_to_highlight]


    ###########################################################
    ### HIGHLIGHT THE MATCHING SEGMENTS IN THE CLOSET IMAGE ###
    ###########################################################
    
    # Create a blank mask the same size as the image_of_possible_matches, where
    # each pixel is black (red = 0, green = 0, blue = 0)
    highlighting_mask = np.zeros(image_of_possible_matches.shape,
                                 dtype=np.uint8)

    # for each y and x pixel coordinate in the image, if the corresponding
    # segment is going to be highlighted, change the corresponding mask pixel 
    # color to white (red = 255, green = 255, blue = 255)
    for y in np.arange(segmented_possible_matches.shape[0]):
        for x in np.arange(segmented_possible_matches.shape[1]):
            if segmented_possible_matches[y,x] in unique_segments_to_highlight:
                highlighting_mask[y,x,0] = 255
                highlighting_mask[y,x,1] = 255
                highlighting_mask[y,x,2] = 255
    
    
    # ensure that image_of_possible_matches is a 'uint8' data type
    image_of_possible_matches=Image.fromarray(np.uint8(
            image_of_possible_matches))
    
    # turn the mask into a 'uint8' data type
    highlighting_mask=Image.fromarray(np.uint8(highlighting_mask))
    
    # convert image_of_possible_matches and the mask to red, green, blue, alpha
    # images (RBGA), so that we can control the alpha levels to highlight with 
    orignal_image_alphas = image_of_possible_matches.convert("RGBA")
    mask_alpha_levels = highlighting_mask.convert("RGBA")
    
    # extract the RBGA data from the RBGA mask
    mask_alpha_levels_data = mask_alpha_levels.getdata()
    
    # for each pixel in the mask...
    mask_alpha_levels_newData = []
    for pixel in mask_alpha_levels_data:
        # if the RBG profile is white (i.e., should be highlighted), then turn 
        # the mask's alpha level to 0 (clear) in that place
        if pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
            mask_alpha_levels_newData.append((255, 255, 255, 0))
        # if the RBG profile is black (i.e., should not be highlighted), then
        # turn the mask's alpha level to 255 (opaque/shaded) in that place
        else:
            mask_alpha_levels_newData.append((0, 0, 0, 255))

    # update the data in the RBGA mask object with the new alpha values
    mask_alpha_levels.putdata(mask_alpha_levels_newData)
    
    # apply the mask alpha values to the original alpha values
    orignal_image_alphas.paste(mask_alpha_levels, mask = mask_alpha_levels)
    
    # convert the alpha data into an RBGA image
    orignal_image_alphas = orignal_image_alphas.convert("RGBA")
    
    # blend the alpha mask image with the original image_of_possible_matches
    # with an interpolation alpha factor of 85%
    image_of_highlighted_matches = Image.blend(
            image_of_possible_matches.convert("RGBA"), 
            orignal_image_alphas,
            alpha = 0.85)
    
    # convert the image back to an RBG format
    image_of_highlighted_matches=image_of_highlighted_matches.convert("RGB")

    ###########################################################################
    ### RETURN THE COLOR MATCHED TO, MATCHING COLORS, AND HIGHLIGHTED IMAGE ###
    ###########################################################################

    return image_to_match_to_color, matching_colors, image_of_highlighted_matches
