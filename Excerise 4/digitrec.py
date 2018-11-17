# Importing the Keras libraries and packages
from keras.models import load_model
model = load_model('models/mnistModel.h5')
from PIL import Image
import numpy as np
import os
import urllib.request
import gzip
import matplotlib.pyplot as plt
import shutil
from skimage.io import imsave

choice = True
while choice:
    print("""
    1.Test image from MNIST DataSet Test Images
    2.Test a Image file(png)
    4.Exit/Quit
    """)
    choice = input("What would you like to do? ")
    if choice=="1":
      print("\nEnter a Image number between 0 to 9999 from Mnist Test Images")
    elif choice=="2":
      print("\n Enter image file")
    elif choice=="4":
      print("\n Goodbye") 
      choice = None
    else:
       print("\n Not Valid Choice Try again")