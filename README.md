# fruit_recognition
Fruit recognition app using CNN, built with Python and TensorFlow.

NOTE: Notebook file was made in Google Colab.
If you want to run it on Jupyter Notebook, you have to make some small changes like removing '!' from the start of some of the commands.

Application works on my own dataset made with 6 fruits and 300 photos (50 photos for each fruit).
The convolutional neural network is built in TensorFlow and uses neural net VGG19 as a base with added Dense and Dropout layers.
It has 98% accuracy on training set and 94% on test set. Loss on validation set is 0.04.

To try the app on your computer you have to run mycamera.py. It works on a default camera in a laptop.
After you press 'spacebar' the app recognizes fruit in front of the camera. Press 'q' to quit.

App works best with clean, white background and nothing besides the fruit in hand in front of the camera.
