# Importing the Keras libraries and packages
from keras.models import load_model
model = load_model('models/mnistModel.h5')
from PIL import Image
import numpy as np
import os
import urllib.request
import gzip
import shutil

# First make a folder which will store the the downloads
path = 'data/'
#initialise array
ndArray = {}

# This is for the first option 
def DownloadFiles():
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

    # This here shows how many files exist in the directory
    # It should have 2 different files in the folder 
    # get a list of all the files in the folder 'data'
    files = os.listdir(path)

    #the for loop goes through each file and extracts it 
    for file in files:
        #checks if the file ends in .gz 
        if file.endswith('.gz'):
            #this reads the file with gzip 
            with gzip.open(path+file, 'rb') as In:
            #removes the .gz file
                with open(path+file.split('.')[0], 'wb') as out:
                    #shutil copies the contents from In to out
                    shutil.copyfileobj(In, out)

    for file in files:
        if file.endswith('.gz'):
            os.remove(path+file)   
        else:
            print('All files have been Removed')

def saveToArray():
    # This here shows how many files exist in the directory
    # It should have 4 different files in the folder 
    # get a list of all the files in the folder 'data/data'
    files = os.listdir(path)

    #go through a loop and add the files to the ndarray
    for file in files:
        #if the extracted file matches then proceed
        if file.endswith('ubyte'):
            #print('Reading the file', file)
            #open the file if the it ends with ubyte and read 
            with open (path+file,'rb') as f:
                #read the file 
                data = f.read() 
                # find out the magic number of the file
                magic = int.from_bytes(data[0:4], byteorder='big')
                # find out the size of the images 
                size = int.from_bytes(data[4:8], byteorder='big')
                
                # this is the size of Test images and labels 
                if (size==10000):
                    #here we will know if the file is a test image/label 
                    trainOrTest = 'test'
                # this is the size for training labels and images 
                elif (size == 60000):
                    #here we will know if the file is a Training image/label 
                    trainOrTest = 'train'
                # This checks the magic number 2051 which is for image files 
                if (magic == 2051):
                    imgOrLAbel = 'images'
                    #This gets the nummber of rows 
                    rows = int.from_bytes(data[8:12], byteorder='big')
                    #this gets the number of columns 
                    cols = int.from_bytes(data[12:16], byteorder='big')
                    # read values as ints # start from 16 as pixels being from byte 16
                    parsed = np.frombuffer(data,dtype = np.uint8, offset = 16) 
                    # we will reshape the length, 28 x 28 
                    parsed = parsed.reshape(size,rows,cols)  
                # this checks the magic number 2049 which is for labels 
                elif (magic == 2049):
                    imgOrLAbel = 'labels'
                    # read values as ints
                    parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
                    #reshape 
                    parsed = parsed.reshape(size)
                #save each file as a array with their key 
                ndArray[trainOrTest+'_'+imgOrLAbel] = parsed
        else:
            print('No File Found')
    print('Done')



# DownloadFiles() # uncomment if you want to download the files again
saveToArray()

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
        unserInput = input()

        print('The Label of the image is: ', ndArray['test_labels'][int(unserInput)]) # it has the same label as the image 
        # This prints out the first image # int(unserInput) = unser input 
        # find the unserInput image from the ndArray 
        image = ndArray['test_images'][int(unserInput),:,:]
        # reshapes the image for prediction
        image = image.reshape(1,28,28,1)
        # Predicting the Test set results
        pred = model.predict(image)
        correct_indices = np.nonzero(pred)
        print("The program predicts image number to be:", correct_indices[-1])
        
    elif choice=="2":
        # I made the images in gimp 100px * 100px 
        # the background needs to be black and the number in white 
        # if not it wil not work
        print("\n Enter image file (0 to 9 just the number)")
        unserInput = input()

        # the label is the name of the image in this case 
        print("\nThe label of the Image is", unserInput)
        #here the image is converted to grayscale and then numpy array
        img = Image.open('images/' + unserInput + '.png').convert("L")
        img = img.resize((28,28))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,28,28,1)
        
        # Predicting the Test set results
        pred = model.predict(im2arr)
        print(pred)
        correct_indices = np.nonzero(pred)
        print(correct_indices)
        print("The program predicts image number to be:", correct_indices[-1])

    elif choice=="3":
        print("\n Goodbye") 
        choice = None
    else:
        print("\n Not Valid Choice Try again")