# CSC320 Fall 2022
# Assignment 3
# (c) Kyros Kutulakos
#
#
# UPLOADING THIS CODE TO GITHUB OR OTHER CODE-SHARING SITES IS
# STRICTLY FORBIDDEN.
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY FORBIDDEN. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY.
#
# THE ABOVE STATEMENTS MUST ACCOMPANY ALL VERSIONS OF THIS CODE,
# WHETHER ORIGINAL OR MODIFIED.

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

import numpy as np
import cv2 as cv

# File psi.py define the psi class. You will need to
# take a close look at the methods provided in this class
# as they will be needed for your implementation
from . import psi

# File copyutils.py contains a set of utility functions
# for copying into an array the image pixels contained in
# a patch. These utilities may make your code a lot simpler
# to write, without having to loop over individual image pixels, etc.
from . import copyutils


#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################

# If you need to import any additional packages
# place them here. Note that the reference
# implementation does not use any such packages

#########################################


#########################################
#
# Computing the Patch Confidence C(p)
#
# Input arguments:
#    psiHatP:
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    confidenceImage:
#         An OpenCV image of type uint8 that contains a confidence
#         value for every pixel in image I whose color is already known.
#         Instead of storing confidences as floats in the range [0,1],
#         you should assume confidences are represented as variables of type
#         uint8, taking values between 0 and 255.
#
# Return value:
#         A scalar containing the confidence computed for the patch center
#

def computeC(psiHatP=None, filledImage=None, confidenceImage=None):
    assert confidenceImage is not None
    assert filledImage is not None
    assert psiHatP is not None

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################

    # Replace this dummy value with your own code
    C = 1
    w = psiHatP.radius()  # radius of the patch
    # here we need the filled color of image and whether or not it is empty
    filled = copyutils.getWindow(filledImage, (psiHatP.row(),
                                               psiHatP.col()), w)[0]
    # confidence
    con, checker = \
        copyutils.getWindow(confidenceImage, (psiHatP.row(), psiHatP.col()), w)
    # from the paper, we need to compute sum of confidence from the source region
    # intersect patch region and divide it by area of filled patch region for average
    # since we assume confidence of unfilled is zero:
    C = np.sum(np.multiply(filled, con / 255)) / np.count_nonzero(checker)
    # average will only take non empty pixel into consideration

    #########################################


    return C


#########################################
#
# Computing the max Gradient of a patch on the fill front
#
# Input arguments:
#    psiHatP:
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    inpaintedImage:
#         A color OpenCV image of type uint8 that contains the
#         image I, ie. the image being inpainted
#
# Return values:
#         Dy: The component of the gradient that lies along the
#             y axis (ie. the vertical axis).
#         Dx: The component of the gradient that lies along the
#             x axis (ie. the horizontal axis).
#

def computeGradient(psiHatP=None, inpaintedImage=None, filledImage=None):
    assert inpaintedImage is not None
    assert filledImage is not None
    assert psiHatP is not None

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################

    # Replace these dummy values with your own code
    Dy = 1
    Dx = 0
    #
    # from https://techtutorialsx.com/2018/06/02/python-opencv-converting-an
    # -image-to-gray-scale/
    gray = cv.cvtColor(inpaintedImage, cv.COLOR_BGR2GRAY)
    # convert to grayscale image
    w = psiHatP.radius()  # radius of the patch
    gray_array = copyutils.getWindow(gray, (psiHatP.row(), psiHatP.col()), w)[0]
    # 2D matrix of size (2w+1)x(2w+1) will be returned
    # compute gradient from
    # https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html
    # dx = 1
    # I don't actually know why CV_64F instead of CV_8U
    # but 64F seems to fit the reference solution
    gradx = cv.Sobel(gray_array, cv.CV_64F, 1, 0, ksize=5)
    # dy = 1
    grady = cv.Sobel(gray_array, cv.CV_64F, 0, 1, ksize=5)
    # determine filled and not filled pixels
    fill_array, check = copyutils.getWindow(filledImage/255, (psiHatP.row(),
                                                          psiHatP.col()), w)
    # make sure all pixels out boundary is not covered
    checker = np.multiply(fill_array, check)
    # we come up the gradient with unfilled eliminated
    x_gradient = np.multiply(gradx, checker)
    y_gradient = np.multiply(grady, checker)
    sum_array = np.add(x_gradient ** 2, y_gradient ** 2)
    # https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
    # we will find index with max value
    i = np.unravel_index(np.argmax(sum_array), sum_array.shape)
    Dx = x_gradient[i]
    Dy = y_gradient[i]
    l = np.sqrt(Dx**2 + Dy**2) # magnitude to normalize
    #########################################



    return Dy/l, Dx/l


#########################################
#
# Computing the normal to the fill front at the patch center
#
# Input arguments:
#    psiHatP:
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    fillFront:
#         An OpenCV image of type uint8 that whose intensity is 255
#         for all pixels that are currently on the fill front and 0
#         at all other pixels
#
# Return values:
#         Ny: The component of the normal that lies along the
#             y axis (ie. the vertical axis).
#         Nx: The component of the normal that lies along the
#             x axis (ie. the horizontal axis).
#
# Note: if the fill front consists of exactly one pixel (ie. the
#       pixel at the patch center), the fill front is degenerate
#       and has no well-defined normal. In that case, you should
#       set Nx=None and Ny=None
#

def computeNormal(psiHatP=None, filledImage=None, fillFront=None):
    assert filledImage is not None
    assert fillFront is not None
    assert psiHatP is not None

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################

    # Replace these dummy values with your own code
    Ny = 0
    Nx = 1
    w = psiHatP.radius()  # radius of the patch
    front, checker = copyutils.getWindow(fillFront,
                                         (psiHatP.row(), psiHatP.col()), w)
    # if front only has one pixel
    # count non-zero since only pixel in fill front will be non zero
    if np.count_nonzero(front) <= 1:
        return None, None
    # firstly, we will find points near the center
    hei, wid = filledImage.shape
    y0 = max(0, w - 3)
    y1 = min(hei - 1, w + 3)
    x0 = max(0, w - 3)
    x1 = min(wid - 1, w + 3)
    # we define the box
    arr = []



    while y0 <= y1:
        x0 = max(0, w - 3)
        while x0 <= x1:
            if front[y0, x0] != 0:
                # if that point in the patch and not zero
                arr.append([y0, x0])
                # then append all qualified points to the array
            x0 += 1
        y0 += 1



    if len(arr) <= 0:  # if there is only one point within the range
        return None, None
    else:
        # https://python.hotexamples.com/examples/cv2/-/fitLine/python-fitline-function-examples.html
        vx, vy, x, y = cv.fitLine(np.array(arr), cv.DIST_L2, 0, 0.01, 0.01)
        # since cv.fitLine use MSE, so point order doesn't matter
        # and from the picture, normal looks fine
        Nx = -vy
        Ny = vx
        # mag = np.sqrt(vx**2 + vy**2)




    #########################################

    return Ny, Nx
