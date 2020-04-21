#!/usr/bin/env python
"""This is a library of functions for performing color-based image segmentation of an image."""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import isnan


def image_patch_outside(img, rectangle):
    '''returns the region outside a rectangle specified by rectangle '''
    imgpatch = img.copy()
    x = rectangle[0][0]
    y = rectangle[0][1]
    w = rectangle[0][2]
    h = rectangle[0][3]
    box = np.array([[y, x, y + h, x + w]])

    if box[0][0] < 0:
        box[0][0] = 0
    if box[0][1] < 0:
        box[0][1] = 0
    if box[0][2] > img.shape[0]:
        box[0][2] = img.shape[0]
    if box[0][3] > img.shape[1]:
        box[0][3] = img.shape[1]

    imgpatch = np.delete(imgpatch, range(x, x + w), 1)

    return imgpatch


def image_patch(img, x, y, w, h):
    """ Returns a region of interest of img specified by box """
    #check box against the boundaries of the image
    box = np.array([[y, x, y + h, x + w]])
    if box[0][0] < 0:
        box[0][0] = 0
    if box[0][1] < 0:
        box[0][1] = 0
    if box[0][2] > img.shape[0]:
        box[0][2] = img.shape[0]
    if box[0][3] > img.shape[1]:
        box[0][3] = img.shape[1]

    return img[box[0][0]:box[0][2], box[0][1]:box[0][3], :]


def pixel_classify(p):
    """ Classify a pixel as background or foreground accoriding to a set of predefined rules """
    #This implementation is a stub. You should implement your own rules here.
    return 1.0


def image_classify(img):
    """ Classify each pixel in an image, and create a black-and-white mask """
    img_segmented = img.copy()
    for r in xrange(0, img.shape[0]):
        for c in xrange(0, img.shape[1]):
            p = img[r, c, :]
            if pixel_classify(p) < 0:
                img_segmented[r, c, :] = 0
            else:
                img_segmented[r, c, :] = 255
    return img_segmented


def image_line_vertical(img, x):
    """ Adds a green 3px vertical line to the image """
    #MODIFIED: so that the line goes across the whole image
    cv2.line(img, (x, 0), (x, img.shape[0]), (0, 255, 0), 3)
    return img


def image_rectangle(img, rectangle):
    """ Adds a green rectangle to the image where RECTANGLE = [X,Y,W,H]"""
    x = rectangle[0][0]
    y = rectangle[0][1]
    w = rectangle[0][2]
    h = rectangle[0][3]
    img = cv2.rectangle(img, (x + w, y + h), (x, y), (0, 255, 0))

    return img


def image_one_to_three_channels(img):
    """ Transforms an image from two channels to three channels """
    #First promote array to three dimensions, then repeat along third dimension
    img_three = np.tile(img.reshape(img.shape[0], img.shape[1], 1), (1, 1, 3))
    return img_three


def test():
    #load sample image
    img = cv2.imread('../data/BU_logo.png', cv2.IMREAD_COLOR)
    #show sample region
    img_patch = image_patch(img, 50, 20, 20, 20)
    #run classifier to segment image
    img_segmented = image_classify(img)
    #add a line at 10px from the left edge
    img_segmented_line = image_line_vertical(img, 10)
    #show results
    """
    cv2.imshow('patch', img_patch)
    cv2.imshow('segmented', img_segmented_line)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    """


def segmentation_prepare_dataset():
    '''The function prepares the image to be classified. It crops the positive and negative
    regions from the original images and display to screen the rectangle corresponding to
    these regions, than saves the cropped images in the corresponding files'''

    #load the original images
    train = cv2.imread('../scripts/line-color-train.jpg')
    cross = cv2.imread('../scripts/line-cross.jpg')
    test = cv2.imread('../scripts/line-test.jpg')

    #import the rectangle coordinates from  photoshop [x,y,w,h]
    positive_train = np.array([[212, 0, 134, 480]])
    positive_cross = np.array([[256, 0, 183, 480]])
    positive_test = np.array([[272, 0, 74, 480]])

    #crop the images
    train_positive = image_patch(train, positive_train[0][0],
                                 positive_train[0][1], positive_train[0][2],
                                 positive_train[0][3])
    cross_positive = image_patch(cross, positive_cross[0][0],
                                 positive_cross[0][1], positive_cross[0][2],
                                 positive_cross[0][3])
    test_positive = image_patch(test, positive_test[0][0], positive_test[0][1],
                                positive_test[0][2], positive_test[0][3])

    train_negative = image_patch_outside(train, positive_train)
    cross_negative = image_patch_outside(cross, positive_cross)
    test_negative = image_patch_outside(test, positive_test)

    #convert the color profile of the images to hsv
    train_positive = cv2.cvtColor(train_positive, cv2.COLOR_BGR2HSV)
    cross_positive = cv2.cvtColor(cross_positive, cv2.COLOR_BGR2HSV)
    test_positive = cv2.cvtColor(test_positive, cv2.COLOR_BGR2HSV)
    train_negative = cv2.cvtColor(train_negative, cv2.COLOR_BGR2HSV)
    cross_negative = cv2.cvtColor(cross_negative, cv2.COLOR_BGR2HSV)
    test_negative = cv2.cvtColor(test_negative, cv2.COLOR_BGR2HSV)

    #saves the new set of image
    cv2.imwrite('./train-positive.png', train_positive)
    cv2.imwrite('./cross-positive.png', cross_positive)
    cv2.imwrite('./test-positive.png', test_positive)
    cv2.imwrite('./train-negative.png', train_negative)
    cv2.imwrite('./cross-negative.png', cross_negative)
    cv2.imwrite('./test-negative.png', test_negative)

    #in the end we will show the original images with a rectangle that highlights the cropped region
    """
    cv2.imshow('CROPPED REGION', image_rectangle(train, positive_train))
    cv2.waitKey()
    cv2.imshow('CROPPED REGION', image_rectangle(cross, positive_cross))
    cv2.waitKey()
    cv2.imshow('CROPPED REGION', image_rectangle(test, positive_test))
    cv2.waitKey()
    cv2.imshow('CROPPED REGION', train_negative)
    cv2.waitKey()
    cv2.imshow('CROPPED REGION', cross_negative)
    cv2.waitKey()

    cv2.imshow('CROPPED REGION', test_negative)
    cv2.waitKey()
    """


def classifier_parameters():
    #lb = np.array([[105], [150], [160]])
    #ub = np.array([[110], [255], [255]])
    lb = (105, 60, 38)
    ub = (110, 255, 255)
    return lb, ub


def segmentation_statistics(filename_positive, filename_negative):
    """The function shows the mask made with the thresold set in the function above, where the positives values are white and negative are black
    takes in input two images"""
    #read the images
    imgpos = cv2.imread(filename_positive)
    imgneg = cv2.imread(filename_negative)
    #checks if any uploading error has occured
    if np.shape(imgpos) == None or np.shape(imgneg) == None:
        print('Reading error')
    else:
        pass

    #import the boundary
    lb, ub = classifier_parameters()
    #creates the masks
    maskpos = cv2.inRange(imgpos, lb, ub)
    maskneg = cv2.inRange(imgneg, lb, ub)
    cv2.imshow('maskpos', maskpos)
    cv2.waitKey()
    cv2.imshow('maskneg', maskneg)
    cv2.waitKey()
    positive_positive = np.count_nonzero(maskpos == 255)
    positive_negative = np.count_nonzero(maskpos == 0)
    negative_positive = np.count_nonzero(maskneg == 255)
    negative_negative = np.count_nonzero(maskneg == 0)

    stats = np.array([[positive_positive], [positive_negative],
                      [negative_positive], [negative_negative]])

    true_positive = (np.shape(imgpos)[0]) * (np.shape(imgpos)[1])
    true_false = (np.shape(imgneg)[0]) * (np.shape(imgneg)[1])
    print(stats, true_positive, true_false)
    precision = float(100 * (true_positive) /
                      (true_positive + negative_positive))
    recall = float(100 * (true_positive) / (true_positive + positive_negative))

    print('For the image %s the precision is %d  and the recall is %d ' %
          (filename_positive[2:], precision, recall))

    return precision, recall


def image_centroid_horizontal(img):

    imgc = img.copy()
    imgc = cv2.cvtColor(imgc, cv2.COLOR_BGR2HSV)

    x = int(np.median(np.where(imgc == 255)[1]))
    if isnan(x) == True:
        print('Centroid computation failed')
    else:
        pass

    imgc = cv2.cvtColor(imgc, cv2.COLOR_HSV2BGR)

    img_with_line = image_line_vertical(imgc, x)

    #cv2.imshow('line', imgc)
    #cv2.waitKey()
    return x


def image_centroid_test():
    #load test image
    img = cv2.imread('../scripts/line-test.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #make segmented image
    lb, ub = classifier_parameters()
    img_seg = cv2.inRange(img, lb, ub)
    #compute centroid
    x = image_centroid_horizontal(img)
    #make img color
    color = image_one_to_three_channels(img_seg)
    #add line on color img
    line = image_line_vertical(color, x)
    #show images
    cv2.imshow('test_original', img)
    cv2.waitKey()
    cv2.imshow('test_segmented', line)
    cv2.waitKey()


if __name__ == '__main__':
    test()
    segmentation_prepare_dataset()
    lb, ub = classifier_parameters()
    img1 = cv2.inRange(
        cv2.cvtColor(cv2.imread('../scripts/line-color-train.jpg'),
                     cv2.COLOR_BGR2HSV), lb, ub)
    img2 = cv2.inRange(
        cv2.cvtColor(cv2.imread('../scripts/line-cross.jpg'),
                     cv2.COLOR_BGR2HSV), lb, ub)
    # cv2.imshow('Line train segmented', img1)
    # cv2.waitKey()
    #cv2.imshow('Line cross segmented', img2)
    # cv2.waitKey()
    segmentation_statistics('./train-positive.png', './train-negative.png')
    #segmentation_statistics('./cross-positive.png', './cross-negative.png')
    #image_centroid_test()
