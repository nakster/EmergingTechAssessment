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

# This is for the first option 
def DownloadFiles():
    # First make a folder which will store the the downloads
    path = 'data/'
    # If the file does not exist then make a new file 
    # This makes sure that a file is made even when it doesnt exist
    if not os.path.exists(path):
        os.makedirs(path)
        
    # This will store all the rest of urls that i need to download
    # The Test Images /Test Labels
    urls = ['http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']

    # We can go through a for loop and download the files
    for url in urls:
        # We can then split each url to just the file name 
        file = url.split('/')[-1]
        # print(file)
        
        #Now in the for loop check if the file exists 
        #if the file that im downloading already exists
        # Then it will not download it 
        if os.path.exists(path+file):
            print('The File Youre trying to download already exists!', file)
        else:
            #if the file does not exist the it will download the file
            print('The',file, 'Is Downloading')
            urllib.request.urlretrieve (url, path+file)
    print('Done Downloading')

DownloadFiles()

choice = True
while choice:
    print("""
    1.Test image from MNIST DataSet Test Images
    2.Test a Image file(png)
    3.Exit/Quit
    """)
    choice = input("What would you like to do? ")

    if choice=="1":
        print("\nEnter a Image number between 0 to 9999 from Mnist Test Images")
    elif choice=="2":
        print("\n Enter image file")
    elif choice=="3":
        print("\n Goodbye") 
        choice = None
    else:
        print("\n Not Valid Choice Try again")





