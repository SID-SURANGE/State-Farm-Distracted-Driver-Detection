# -*- coding: utf-8 -*-

# IMPORTS
import cv2
import os


# CREATE TEST AND TRAIN DATA -------------------------------------------------------------------------------------------
# TRAIN DATA
def create_training_data(directory, classes):
    training_data = []

    # OS MODULE TO JOIN AND SEARCH FOR IMAGE FOLDER LOCATION
    for category in classes:
        path = os.path.join(directory, category)
        class_num = classes.index(category)

        # IMAGE SIZE DESCALING(using resize()) TO 240*240 AND APPENDING DATA TO training_data
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_img = cv2.resize(img_array, (240, 240))
            training_data.append([
                new_img, class_num])
    return training_data
# eof


# TESTING DATA - Around 79k test images for 10 classes------------------------------------------------------------------
# IMAGES WILL BE APPENDED TO testing_data LIST
def create_testing_data(test_directory):
    testing_data = []

    for img in os.listdir(test_directory):
        img_array = cv2.imread(os.path.join(test_directory, img), cv2.IMREAD_GRAYSCALE)
        new_img = cv2.resize(img_array, (240, 240))
        testing_data.append([img, new_img])

    return testing_data

# eof
