import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, time, ssl

# Fetching the data from open_ml.
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

# print(pd.Series(y).value_counts())

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

nClasses = len(classes)

# Splitting the data and scaling it.

X_Train, X_Test, Y_Train, Y_Test = train_test_split(
    X, y, random_state=9, train_size=7500, test_size=2500
)

X_Train_Scale = X_Train / 255.0
X_Test_Scale = X_Test / 255.0

logisticModel = LogisticRegression(solver="saga", multi_class="multinomial").fit(
    X_Train_Scale, Y_Train
)

y_predicted = logisticModel.predict(X_Test_Scale)

accuracy = accuracy_score(Y_Test, y_predicted)

print("The accuracy of the model is", accuracy)


# Starting the camera -

cam = cv2.VideoCapture(0)

while True:
    try:
        ret, frame = cam.read()

        # Converting the frame to greyscale to avoid colour issues -
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Drawing the box in the center of the video to detect the image -
        height, width = grey.shape

        # Deciding the coordinates for the rectangle -
        upper_left = (int(width / 2 - 56), int(height / 2 - 56))
        bottom_right = (int(width / 2 + 56), int(height / 2 + 56))

        cv2.rectangle(grey, upper_left, bottom_right, (0, 255, 0), 2)

        roi = grey[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]

        # Converting cv2 image to PIL format which allows lightweight image processing tools -
        im_pil = Image.fromarray(roi)

        # Converting greyscale image to 'L' format wherein each pixel is represented by a single value from 0 to 255.
        im_pil = im_pil.convert("L")

        # Resizing the image -
        image_resized = im_pil.resize((28, 28), Image.ANTIALIAS)
        image_resized_inverted = PIL.ImageOps.invert(image_resized)
        pixel_filter = 20

        min_pixel = np.percentile(image_resized_inverted, pixel_filter)
        image_resized_inverted_scaled = np.clip(
            image_resized_inverted - min_pixel, 0, 255
        )

        maximum_pixel = np.max(image_resized_inverted)
        image_resized_inverted_scaled = (
            np.asarray(image_resized_inverted_scaled) / maximum_pixel
        )
        test_sample = np.array(image_resized_inverted_scaled).reshape(1, 784)
        test_pred = logisticModel.predict()

        print("Predicted class is {}".format(test_pred))

        cv2.imshow(frame, grey)

        if cv2.waitKey(1):
            break
    except Exception as e:
        print(e)

cam.release()
cv2.destroyAllWindows()
