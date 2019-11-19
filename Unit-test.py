# -*- coding: utf-8 -*-

import imports
import os
import cv2
import numpy as np
import keras.backend as K
import glob

# setting the directory path for Test image-----------------------------------------------------------------------------
dir_path = os.path.dirname(os.path.realpath(__file__))
STATIC_FOLDER = dir_path + '/Flask/static'
IMG_FOLDER = dir_path + '/Test_img/'
print('dir-path', dir_path)


# Create a dictionary for class values and labels-----------------------------------------------------------------------
indices = {0: 'safe driving', 1: 'texting-right', 2: 'talking on the phone - right', 3: 'texting - left',
           4: 'talking on the phone - left', 5: 'operating the radio', 6: 'drinking', 7: ' reaching behind',
           8: 'hair and makeup', 9: 'talking to passenger'}


# LOAD THE SAVED MODEL--------------------------------------------------------------------------------------------------
json_file = open(STATIC_FOLDER + '/' + 'Modelarc.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = imports.model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(STATIC_FOLDER + '/' + "Modelweight.h5")
print("Loaded model from disk")


# FUNCTION THAT MAKES PREDICTION ON NEW IMAGES
def test(timg):
    # TRANSFORM AND SCALE THE IMAGE TO SIZE 240*240
    img_array = cv2.imread(IMG_FOLDER + timg, cv2.IMREAD_GRAYSCALE)
    new_img = cv2.resize(img_array, (240, 240))
    x = np.array(new_img).reshape(-1, 240, 240, 1)
    print('image-shape', x.shape)

    # MAKE PREDICTIONS AND DETERMINE CLASS VALUE
    predicted = model.predict(x)
    print('predicted   ', predicted)
#    print('diff is {}'.format(predicted[0] - predicted[1]))
    predicted_class = np.argmax(predicted)
    print('pred-class {}'.format(predicted_class))

# IMPLEMENT GRAD CAM TO VISUALIZE WHAT MODEL INTERPRETS-----------------------------------------------------------------
    class_output = model.output[:, predicted_class]
    print('class_output {}'.format(class_output))
    last_conv_layer = model.get_layer("conv2d_6")
    print('lcl {}'.format(last_conv_layer))
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    print('pooled grad {}'.format(pooled_grads))
    print('Model inpit {}'.format(model.input))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])

    for i in range(128):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    img = cv2.imread(IMG_FOLDER + timg)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
#    cv2.imshow("Original", img)
#    cv2.imshow("GradCam", superimposed_img)
#    spath = 'C:\\Users\\sidsu\\PycharmProjects\\Capstone1\\superimposed_img' +  + '.jpg'
    cv2.imwrite('C:\\Users\\sidsu\\PycharmProjects\\Capstone1\\superimposed_img.jpg', superimposed_img)

    # RETURN CLASS LABEL------------------------------------------------------------------------------------------------
    label = indices[predicted_class]
    print('label is {}'.format(label))
    return predicted_class, label
# END OF FUNCTION


# CALLING THE FUNCTION WITH IMAGE TO BE TESTED
cvalue, label1 = test('c81.jpg')
print('The driver is distracted by {} whose label value is {}'.format(label1, cvalue))


# CALLING PREDICT FOR MULTIPLE IMAGES
# p1 = "Test_img\\*.*"
# path = glob.glob(p1)
# for img in path:
#    cvalue, label1 = test(img)
#    print('The driver is distracted by {} whose label value is {}'.format(label1, cvalue))

# mkl-fft==1.0.6
# mkl-random==1.0.1.1
# mkl-service==2.0.2
