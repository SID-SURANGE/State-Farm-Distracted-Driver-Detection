# coding = utf-8

import os
import cv2
import imports
from flask import Flask, render_template, request
from flask import send_from_directory
import numpy as np
import tensorflow as tf

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + "/uploads"
# STATIC_FOLDER = dir_path + "/static"
print("dir-path", dir_path)
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

# READ MODEL FROM FILE--------------------------------------------------------------------------------------------------
json_file = open(STATIC_FOLDER + "/" + "Model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
model = imports.model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(STATIC_FOLDER + "/" + "Model-weights.h5")
graph = tf.get_default_graph()

print("Loaded model from disk")
# ----------------------------------------------------------------------------------------------------------------------


# call model to predict an image
def api(full_path):
    print("b4")
#   img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
    img_array = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    new_img = cv2.resize(img_array, (240, 240))
    x = np.array(new_img).reshape(-1, 240, 240, 1)
    print("image-shape", x.shape)

    global graph
    with graph.as_default():
        predicted = model.predict(x)

    print("predicted   ", predicted)
    return predicted
# eof----------------------------------------

# home page-------------------------------------------------------------------------------------------------------------
@app.route("/")
def home():
    print("b1")
    return render_template("index.html")


# proccesing uploaded file and predict it ------------------------------------------------------------------------------
@app.route("/upload", methods=["POST", "GET"])
def upload_file():
    print("b2")
    if request.method == "GET":
        return render_template("index.html")
    else:
        file = request.files["image"]
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)

        indices = {0: "Safe driving", 1: "Texting-right", 2: "Talking on the phone - right", 3: "Texting - left",
                   4: "Talking on the phone - left", 5: "Operating the radio", 6: "Drinking", 7: " Reaching behind",
                   8: "Hair and makeup", 9: "Talking to passenger"}

        result = api(full_name)

        predicted_class = np.argmax(result)
        print("pred-class", predicted_class)
#        accuracy = round(result[0][predicted_class] * 100, 2)
        label = indices[predicted_class]

    return render_template("pred.html", image_file_name=file.filename, label=label) #, accuracy=accuracy)

# eof-------------------------------------------------------------------------------------------------------------------
@app.route("/uploads/<filename>")
def send_file(filename):
    print("b3")
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    print("b")
    app.debug = True
    app.run(debug=True)
    app.debug = True

#  <h3><b>Accuracy : <span style="color: green;">{{ accuracy }} %</span></b></h3>
#    body {
#    background-image: url("https://media.giphy.com/media/lDlZoFVhfF2nu/giphy.gif");
#    background-position: center;
#    background-size: cover;
#    }
# ..