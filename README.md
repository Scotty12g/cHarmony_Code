# cHarmony Color Matching Code

## Overview
These python files are the basis of an image data pipeline that searches for matching garments in a users wardrobe. The pipeline inputs are


## How to run the code

* import ‘charmony_run’ from the ‘charmony_pipeline.py’ file

* Execute ‘charmony_run’ with the following inputs

	* The color matching method, based on color theory color wheel rotation
		* ’complement' (180 degree rotation)
		* ‘triad' (120 degree rotation)

	* File path to an image of the garment to match to

	* File path to an image of a closet with possible matching garments

* The executed script returns:

	* The color of the garment you wish to match to

	* A list of the colors that match the garment

	* The image of the closet with matching items highlighted and non-matching items greyed out

## File Overview

**charmony_pipeline.py**: The main image data pipeline for the user to call and execute.

Pipeline process:
* Import user inputs of how to match clothing, and the file paths to the two images (see description above in “how to run the code”). 

* Load a pre-trained K-Nearest Neighbor color classifier (‘color_detector.pkl’) via the unpickling script in ‘picker.py’.

* Load images, reduce their size (with ‘resize_image.py’) and correct image color profiles via functions in ‘color_correction.py’.

* Identify the color of the garment to match to by projecting 400 random points onto the center of the image, identifying the color at each one, and calculating the most frequent color (with the ‘mode_function.py’ file).

* Determine which color(s) match the first garment via color theory calculations (i.e., rotating a simulated color wheel with ‘color_wheel_rotator.py’) and the user defined method (‘complement’ vs. ‘triad’).

* Segment the image of the closet with Simple Linear Iterative Clustering

* Identify the color of 1/5 of the pixies in each segment (with the K-Nearest Neighbor classifier) and label that segment with the most frequent color found (with ‘mode_function.py’).

* Highlight the segments that are labeled with the matching color determined earlier, and grey-out the segments with non-matching colors.

**color_correction.py**: A color correction method that balances the color histograms of an image based on a user defined threshold ("percent_correct"), and then adjusts the gamma and applies a bilateral filter.

**color_detector.pkl**: A pre-trained K-Nearest Neighbor color classifier.

**color_wheel_rotator.py**: A function to take a Red Green Blue color profile, rotate a virtual colorwheel, and return the corresponding Red Green Blue color profile at that degree rotation.

**mode_function.py**: A function to get the most frequent (mode) value from a list.

**pickler.py**: A method to store (pickle) and load (unpickle) files.

**resize_image.py**: A function to resize an image based on a given width.
