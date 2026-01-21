import os
from flask import Flask, render_template, request
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import tensorflow as tf
#import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = os.path.abspath("static")
MODEL = None

@app.route('/', methods=['GET', "POST"])  # route to display the home page
def upload_predict():
    if request.method == "POST":
        image_file = request.files["user-img"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            return render_template("index.html", prediction=1)
    return render_template("index.html", prediction=0)


if __name__=="__main__":
     app.run(debug=True)
