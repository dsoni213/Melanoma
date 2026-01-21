import tensorflow as tf 
classifierLoad = tf.keras.models.load_model('model.h5') # load the model here

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('Harsh.jpg', target_size = (200,200))  # load the sample image here
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifierLoad.predict(test_image)
if result[0][1] == 1:
    print("Below 50% .... You may consult the doctor")
elif result[0][0] == 1:
    print("Below 10% .... Its beginning stage")
elif result[0][2] == 1:
    print("Below 80% .... Take medecine regularly")
elif result[0][3] == 1:
    print("Above 80% .... You are in risk, its final stage - Surgery Needed")   # this are results
