__author__ = 'Roberta Evangelista'
__email__ = 'roberta.evangelista@posteo.de'

"""Modified from the LinkedIn Learning course: 
Building Deep Learning Applications with Keras 2.0, by Adam Geitgey """

import numpy as np
from keras.preprocessing import image
from keras.applications import resnet50

# Load Keras' ResNet50 model (need internet connection)
model = resnet50.ResNet50()

# Load the image file, resizing it to 224x224 pixels (required by this model, correspond to the num. inputs to NN)
img = image.load_img('plaza.jpg', target_size=(224,224))

# Convert the image to a numpy array (for each pixel, 3dim are the 3 RGB color codes)
x = image.img_to_array(img)

# Add a forth dimension since Keras expects a list of images
x = np.expand_dims(x, axis=0)

# Scale the input image to the range used in the trained network
x = resnet50.preprocess_input(x)

# Run the image through the deep neural network to make a prediction. ResNet50 returns a 1000 dim array saying how
# likely it is that the picture belong to one of the 1000 classes the NN was trained on
predictions = model.predict(x)

# Look up the names of the predicted classes. Index zero is the results for the first image.
predicted_classes = resnet50.decode_predictions(predictions, top=9)

print("This is an image of:")

for imagenet_id, name, likelihood in predicted_classes[0]:
    print(" - {}: {:2f} likelihood".format(name, likelihood))

