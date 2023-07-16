#!/usr/bin/env python
# coding: utf-8

# In[2]:


from commonfunctions import *
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import utlis
from sklearn import datasets, svm, metrics
import scipy
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
os.system("clear")
from skimage.transform import rescale,resize,downscale_local_mean
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
import numpy as np
import random 

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt 

from sklearn import datasets 
from sklearn import svm 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.metrics import classification_report, plot_confusion_matrix


import math
import os
import pytesseract

import cv2 #version 3.2.0
from skimage.feature import hog
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from xlwt import Workbook


# In[3]:


# Functions [show, scann]

def _imshow(img):
    cv2.imshow('image', img)
    plt.show()
    # specify a wait key from keyboard
    k = cv2.waitKey(0) & 0xFF

    if k == 27: #esc in keyboard
        cv2.destroyAllWindows() #close the window   

    elif k == ord('s'): #if order is s save the image
        cv2.imwrite('Test.png', img) #write image in your pc     
        cv2.destroyAllWindows() # close the window 

def _scannar(img):
    heightImg = img.shape[1]
    widthImg  = img.shape[0]
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    # imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(img, (3, 3), 1) # ADD GAUSSIAN BLUR
    imgBlur = img

    imgsobel = sobel(imgBlur)
    imgThreshold = cv2.Canny(imgBlur,100,150) # APPLY CANNY BLUR
    kernel = np.ones((3, 3))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION



    ## FIND ALL COUNTOURS
    imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS
    imgThreshold = cv2.cvtColor(imgThreshold,cv2.COLOR_BAYER_BG2BGR)


    # FIND THE BIGGEST COUNTOUR
    biggest, maxArea = utlis.biggestContour(contours) # FIND THE BIGGEST CONTOUR

    if biggest.size != 0:
        biggest=utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
        imgBigContour = utlis.drawRectangle(imgBigContour,biggest,2)
        pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        #REMOVE 20 PIXELS FORM EACH SIDE
        imgWarpColored=imgWarpColored[:imgWarpColored.shape[0] +150, :imgWarpColored.shape[1] +150 ]

        # APPLY ADAPTIVE THRESHOLD
        # imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpColored, 255, 1, 1, 7, 2)
        img = cv2.adaptiveThreshold(imgWarpColored,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,85,12)

        # imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        # imgAdaptiveThre=cv2.medianBlur(imgWarpColored,3)

    # thres = 130
    # img_bin = np.copy(imgAdaptiveThre)
    # img_bin[imgAdaptiveThre < thres] = 0
    # img_bin[imgAdaptiveThre >= thres] = 255
    # img = img_bin
    # _imshow(img)
    return img


# In[4]:


#symbols dataset
def Readcheck(directory):
    fnames = os.listdir(directory)
    check = []
    for fn in fnames:
        path = os.path.join(directory, fn)
        gray_scale_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        check.append(gray_scale_image)

    return check

def ReadQ(directory):
    fnames = os.listdir(directory)
    Q = []
    for fn in fnames:
        path = os.path.join(directory, fn)
        gray_scale_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        Q.append(gray_scale_image)

    return Q

def ReadEmpty(directory):
    fnames = os.listdir(directory)
    empty = []
    for fn in fnames:
        path = os.path.join(directory, fn)
        gray_scale_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        empty.append(gray_scale_image)

    return empty

def ReadSquare(directory):
    fnames = os.listdir(directory)
    square = []
    for fn in fnames:
        path = os.path.join(directory, fn)
        gray_scale_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        square.append(gray_scale_image)

    return square

def Readver1(directory):
    fnames = os.listdir(directory)
    ver1 = []
    for fn in fnames:
        path = os.path.join(directory, fn)
        gray_scale_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        ver1.append(gray_scale_image)

    return ver1

def Readver2(directory):
    fnames = os.listdir(directory)
    ver2 = []
    for fn in fnames:
        path = os.path.join(directory, fn)
        gray_scale_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        ver2.append(gray_scale_image)

    return ver2

def Readver3(directory):
    fnames = os.listdir(directory)
    ver3 = []
    for fn in fnames:
        path = os.path.join(directory, fn)
        gray_scale_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        ver3.append(gray_scale_image)

    return ver3

def Readver4(directory):
    fnames = os.listdir(directory)
    ver4 = []
    for fn in fnames:
        path = os.path.join(directory, fn)
        gray_scale_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        ver4.append(gray_scale_image)

    return ver4

def Readver5(directory):
    fnames = os.listdir(directory)
    ver5 = []
    for fn in fnames:
        path = os.path.join(directory, fn)
        gray_scale_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        ver5.append(gray_scale_image)

    return ver5

def Readdash(directory):
    fnames = os.listdir(directory)
    dash = []
    for fn in fnames:
        path = os.path.join(directory, fn)
        gray_scale_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        dash.append(gray_scale_image)

    return dash

def Readhor2(directory):
    fnames = os.listdir(directory)
    hor2 = []
    for fn in fnames:
        path = os.path.join(directory, fn)
        gray_scale_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        hor2.append(gray_scale_image)

    return hor2

def Readhor3(directory):
    fnames = os.listdir(directory)
    hor3 = []
    for fn in fnames:
        path = os.path.join(directory, fn)
        gray_scale_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        hor3.append(gray_scale_image)

    return hor3

def Readhor4(directory):
    fnames = os.listdir(directory)
    hor4 = []
    for fn in fnames:
        path = os.path.join(directory, fn)
        gray_scale_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        hor4.append(gray_scale_image)

    return hor4

check= []
Q=[]
square= []
empty = []
dash=[]
hor2 = []
hor3 =[]
hor4 = []
ver1=[]
ver2=[]
ver3=[]
ver4=[]
ver5=[]

check= Readcheck(r"D:\Farah\college\Image processing\module1\cells\check")
Q = ReadQ(r"D:\Farah\college\Image processing\module1\cells\q")
square = ReadSquare(r"D:\Farah\college\Image processing\module1\cells\square")
empty = ReadEmpty(r"D:\Farah\college\Image processing\module1\cells\empty")
dash = Readdash(r"D:\Farah\college\Image processing\module1\cells\dash")
hor2= Readhor2(r"D:\Farah\college\Image processing\module1\cells\hor2")
hor3 = Readhor3(r"D:\Farah\college\Image processing\module1\cells\hor3")
hor4 = Readhor4(r"D:\Farah\college\Image processing\module1\cells\hor4")
ver1 = Readver1(r"D:\Farah\college\Image processing\module1\cells\ver1")
ver2 = Readver2(r"D:\Farah\college\Image processing\module1\cells\ver2")
ver3 = Readver3(r"D:\Farah\college\Image processing\module1\cells\ver3")
ver4 = Readver4(r"D:\Farah\college\Image processing\module1\cells\ver4")
ver5 = Readver5(r"D:\Farah\college\Image processing\module1\cells\ver5")


# In[5]:


#Handwritten digits classifier


DIGIT_WIDTH = 10
DIGIT_HEIGHT = 20
IMG_HEIGHT = 28
IMG_WIDTH = 28
CLASS_N = 10 

class KNN_MODEL():
    def __init__(self, k=3):
        self.k = k
        self.model = cv2.ml.KNearest_create()

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.findNearest(samples, self.k)
        return results.ravel()
class SVM_MODEL():
    def __init__(self, num_feats, C=1, gamma=0.1):
        self.model = cv2.ml.SVM_create()
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setKernel(cv2.ml.SVM_RBF)  # SVM_LINEAR, SVM_RBF
        self.model.setC(C)
        self.model.setGamma(gamma)
        self.features = num_feats

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        results = self.model.predict(samples.reshape(-1, self.features))
        return results[1].ravel()
    
def pixels_to_hog_20(img_array):
    hog_featuresData = []
    for img in img_array:
        fd = hog(img,orientations=10,pixels_per_cell=(5, 5),cells_per_block=(1, 1),Visualise=False)
        hog_featuresData.append(fd)
    hog_features = np.array(hog_featuresData, 'float64')
    return np.float32(hog_features)

def get_contour_precedence(contour, cols):
    return contour[1] * cols + contour[0]  # row-wise ordering


def get_digits(contours, hierarchy):
    hierarchy = hierarchy[0]
    bounding_rectangles = [cv2.boundingRect(ctr) for ctr in contours]
    final_bounding_rectangles = []
    # find the most common heirarchy level - that is where our digits's bounding boxes are
    u, indices = np.unique(hierarchy[:, -1], return_inverse=True)
    most_common_heirarchy = u[np.argmax(np.bincount(indices))]

    for r, hr in zip(bounding_rectangles, hierarchy):
        x, y, w, h = r
        # this could vary depending on the image you are trying to predict
        # we are trying to extract ONLY the rectangles with images in it (this is a very simple way to do it)
        # we use heirarchy to extract only the boxes that are in the same global level - to avoid digits inside other digits
        # ex: there could be a bounding box inside every 6,9,8 because of the loops in the number's appearence - we don't want that.
        # read more about it here: https://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
        if ((w * h) > 250) and (10 <= w <= 200) and (10 <= h <= 200) and hr[3] == most_common_heirarchy:
            final_bounding_rectangles.append(r)

    return final_bounding_rectangles

def load_digits_custom(img_file, ):
    train_data = []
    # pd.read_csv('train.csv')
    # train_data=
    train_target = []
    start_class = 1
    im = cv2.imread(img_file)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # plt.imshow(imgray)
    kernel = np.ones((5, 5), np.uint8)

    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    digits_rectangles = get_digits(contours, hierarchy)  # rectangles of bounding the digits in user image

    # sort rectangles accoring to x,y pos so that we can label them
    digits_rectangles.sort(key=lambda x: get_contour_precedence(x, im.shape[1]))

    for index, rect in enumerate(digits_rectangles):
        x, y, w, h = rect
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        im_digit = imgray[y:y + h, x:x + w]
        im_digit = (255 - im_digit)

        im_digit = cv2.resize(im_digit, (IMG_WIDTH, IMG_HEIGHT))
        train_data.append(im_digit)
        train_target.append(start_class % 10)

        if index > 0 and (index + 1) % 10 == 0:
            start_class += 1
    cv2.imwrite("training_box_overlay.png", im)

    return np.array(train_data), np.array(train_target)

def Num_Classifier(img_test):
    TRAIN_MNIST_IMG = 'digits.png'
    TRAIN_USER_IMG = 'custom_train_digits.jpg'
    TEST_USER_IMG = img_test
    # digits, labels = load_digits(TRAIN_MNIST_IMG) #original MNIST data (not good detection)
    digits, labels = load_digits_custom(TRAIN_USER_IMG)  # my handwritten dataset (better than MNIST on my handwritten digits)

    print('train data shape', digits.shape)
    print('test data shape', labels.shape)

    digits, labels = shuffle(digits, labels, random_state=256)
    train_digits_data = pixels_to_hog_20(digits)
    X_train, X_test, y_train, y_test = train_test_split(train_digits_data, labels, test_size=0.33, random_state=42)

    # ------------------training and testing----------------------------------------

    model = KNN_MODEL(k=7)
    model.train(X_train, y_train)
    preds = model.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, preds))

    model = KNN_MODEL(k=7)
    model.train(train_digits_data, labels)
    numbers = proc_user_img(TEST_USER_IMG, model)

    model = SVM_MODEL(num_feats=train_digits_data.shape[1])
    model.train(X_train, y_train)
    preds = model.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, preds))

    model = SVM_MODEL(num_feats=train_digits_data.shape[1])
    model.train(train_digits_data, labels)
    proc_user_img(TEST_USER_IMG, model)
    w = ""
    for i in reversed(numbers):
        w += i + ""

    return w


# In[34]:


######################################
# Read scan image
######################################

# read your file
file = "1.jpg"
img = cv2.imread(file,1)
## resize image if needed
# img = cv2.resize(img,None,fx = 2,fy = 2)
img = cv2.resize(img,(2984,3990))
# img = resize(img,(2984,3990))
_imshow(img)

# convert to gray scale image 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
## applay adaptive threshold for images that have different priattness 
# img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,85,10)

# scan image 
img_sc = _scannar(img) 

# thresholding the image to a binary image
thresh, img_bin = cv2.threshold(img_sc, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# inverting the image
img_sc = img_bin
img_bin = 255-img_bin
_imshow(img_sc)
show_images([img_sc,img_bin])
print(img_sc.shape)


# In[35]:


######################################
# Convert image into colunms 
######################################
# print(img_sc.shape)
# # get 1st col
# crope = 25
# col_1 = img_sc[:,:img_sc.shape[1]//8 + crope]
# col_1_bin = img_bin[:,:img_bin.shape[1]//8 + crope]

# # get 3th col
# col_3 = img_sc[:,img_sc.shape[1]-2565:img_sc.shape[1]-1145]
# col_3_bin = img_bin[:,img_bin.shape[1]-2565:img_bin.shape[1]-1145]

# # get 4th col
# col_4 = img_sc[:,img_sc.shape[1]-1170:img_sc.shape[1]-800]
# col_4_bin = img_bin[:,img_bin.shape[1]-1170:img_bin.shape[1]-800]

# # get 5th col
# col_5 = img_sc[:,img_sc.shape[1]-820:img_sc.shape[1]-450]
# col_5_bin = img_bin[:,img_sc.shape[1]-820:img_sc.shape[1]-450]

# # get 6th col
# col_6 = img_sc[:,img_sc.shape[1]-476:img_sc.shape[1]-50]
# col_6_bin = img_bin[:,img_sc.shape[1]-476:img_sc.shape[1]-50]

# # _imshow(img_bin)
# show_images([col_1,col_3,col_4,col_5,col_6])
# show_images([col_1_bin,col_3_bin,col_4_bin,col_5_bin,col_6_bin])


# In[36]:


######################################
# Get table stracture 
######################################

# countcol(width) of kernel as 100th of total width
kernel_len = np.array(img_sc).shape[1]//100
# Defining a vertical kernel to detect all vertical lines of image
ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
# Defining a horizontal kernel to detect all horizontal lines of image
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len+8, 1))
# A kernel of 2x2
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))

# Use vertical kernel to detect and save the vertical lines in a jpg
image_1 = cv2.erode(img_bin, ver_kernel, iterations=8)
vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=8)

# Use horizontal kernel to detect and save the horizontal lines in a jpg
image_2 = cv2.erode(img_bin, hor_kernel, iterations=8)
horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=20)


# Combine horizontal and vertical lines in a new third image, with both having same weight.
img_vh = cv2.addWeighted(vertical_lines, 0.1, horizontal_lines, 0.1, 0.0)
# Eroding and thesholding the image
img_vh = cv2.erode(~img_vh, kernel, iterations=4)
thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# bitxor = cv2.bitwise_xor(img_sc, img_vh)
# bitnot = cv2.bitwise_not(bitxor)

# Plotting the generated image
show_images([image_1,image_2,img_vh])


# In[37]:


######################################
# Detect and sort contours
######################################

contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


# Sort all the contours by top to bottom.
contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
cont_img = np.copy(img_vh)
cv2.drawContours(cont_img, contours,4, (0, 255, 0), 0)
# show_images([cont_img])
cont_img = img_vh


# In[38]:


# Creating a list of heights for all detected boxes
heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]

# Get mean of heights
mean = np.mean(heights)

# Create list box to store all boxes in
box = []

# Get position (x,y), width and height for every contour and show the contour on image
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if (w < 10000 and w > 200  and  h < 500):
        # image = cv2.rectangle(img_sc, (x, y), (x+w, y+h), (0, 255, 0),1)
        box.append([x, y, w, h])
print(box[5])


# In[39]:


######################################
# Convert image into colunms 
######################################
margin = 40
cols = []

for i in range(5,-1,-1):
    col = img_sc[:,box[i][0]-margin:box[i][0]+box[i][2]+margin]
    cols.append(col)





# In[40]:


col_1 = np.array(cols[0])
col_2 = np.array(cols[1])
col_3 = np.array(cols[2])
col_4 = np.array(cols[3])
col_5 = np.array(cols[4])
col_6 = np.array(cols[5])  

ncol_0 = 255 - col_1
ncol_1 = 255 - col_2
ncol_2 = 255 - col_3
ncol_3 = 255 - col_4
ncol_4 = 255 - col_5
ncol_5 = 255 - col_6

show_images([cols[0],cols[1],cols[2],cols[3],cols[4],cols[5]])
show_images([ncol_0,ncol_1,ncol_2,ncol_3,ncol_4,ncol_5])


# In[47]:


######################################
# Get table stracture 
######################################

tcol = cols[1]
ncol = ncol_0

# countcol(width) of kernel as 100th of total width
kernel_len_col1  = np.array(tcol).shape[1]//50
# Defining a vertical kernel to detect all vertical lines of image
ver_kernel_col1  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_col1 ))
# Defining a horizontal kernel to detect all horizontal lines of image
hor_kernel_col1  = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_col1 +8, 1))
# A kernel of 2x2
kernel_col1  = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))

# Use vertical kernel to detect and save the vertical lines in a jpg
image_1_col1  = cv2.erode(ncol, ver_kernel_col1 , iterations=10)
vertical_lines_col1  = cv2.dilate(image_1_col1, ver_kernel_col1 , iterations=8)

# Use horizontal kernel to detect and save the horizontal lines in a jpg
image_2_col1  = cv2.erode(ncol, hor_kernel, iterations=15)
horizontal_lines_col1  = cv2.dilate(image_2_col1, hor_kernel_col1 , iterations=8)


# Combine horizontal and vertical lines in a new third image, with both having same weight.
img_vh_col1 = cv2.addWeighted(vertical_lines_col1 , 0.5, horizontal_lines_col1 , 0.5, 0.0)
# Eroding and thesholding the image
img_vh_col1 = cv2.erode(~img_vh_col1 , kernel_col1 , iterations=4)
thresh, img_vh_col1 = cv2.threshold(img_vh_col1 , 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# bitxor = cv2.bitwise_xor(img_sc, img_vh)
# bitnot = cv2.bitwise_not(bitxor)

# Plotting the generated image
show_images([image_1_col1 ,image_2_col1 ,img_vh_col1])


# In[48]:


######################################
# Detect and sort contours
######################################

contours_col1, hierarchy_col1 = cv2.findContours(img_vh_col1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


# Sort all the contours by top to bottom.
contours_col1, boundingBoxes = sort_contours(contours_col1, method="top-to-bottom")
cont_img = np.copy(img_vh_col1)
# cv2.drawContours(cont_img, contours_col1,1, (0, 255, 0), 0)
show_images([cont_img])
cont_img = img_vh


# In[49]:


# Creating a list of heights for all detected boxes
heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]

# Get mean of heights
mean = np.mean(heights)

# Create list box to store all boxes in
box = []

# Get position (x,y), width and height for every contour and show the contour on image
for c in contours_col1:
    x, y, w, h = cv2.boundingRect(c)
    if (w < 10000 and w > 200  and  h < 500):
        # image = cv2.rectangle(img_sc, (x, y), (x+w, y+h), (0, 255, 0),1)
        box.append([x, y, w, h])
print(box[5])


# In[50]:


col_cells = []
m = 5
for i in range(len(box)):
    img3 = tcol[box[i][1]-m:box[i][1]+box[i][3],box[i][0]:box[i][0]+box[i][2]]
    col_cells.append(img3)


# In[51]:



show_images([col_cells[1]])
cells_img = col_cells


# In[52]:



cell_list = []
i = 2
img5 = cells_img[i]
_, img5 = cv2.threshold(img5,100,255,cv2.THRESH_BINARY)
show_images([img5])
#pytesseract.image_to_string(img5)


# In[53]:



_, cells_img[0] = cv2.threshold(cells_img[0],70,255,cv2.THRESH_BINARY)
# celltext = pytesseract.image_to_string(cells_img[0])
# cell_list.append(celltext)
# show_images([cells_img[0]])
# print(celltext)

w=0

for i in range(1,len(cells_img)):
    _, cells_img[i] = cv2.threshold(cells_img[i],90,255,cv2.THRESH_BINARY)
    
    #celltext = pytesseract.image_to_string(cells_img[i])
   
    crop = cells_img[i]
    
    
    for x in check:
        if np.all(crop == x):
            w = '5'
            cell_list.append(w)
    for o in Q: 
        if np.all(crop == o):
            w = "?"
            cell_list.append(w)
    #change it in the excel sheet
    for k in square: 
        if np.all(crop == k):
            w = '0' 
            cell_list.append(w)
    for p in empty:
        if np.all(crop == p):
            w = " "
            cell_list.append(w)
    for l in dash:
        if np.all(crop == l):
            w = '0'
            cell_list.append(w)
    for m in hor2:
        if np.all(crop == m):
            w= '-3'
            cell_list.append(w)
    for n in hor3:
        if np.all(crop == n):
            w= '-2'
            cell_list.append(w)
    for z in hor4:
        if (crop == z):
            w= '-1'
            cell_list.append(w)
    for a in ver1:
        if np.all(crop == a):
            w= '1'
            cell_list.append(w)
    for s in ver2:
        if np.all(crop == s):
            w= '2'
            cell_list.append(w)
          
    for d in ver3:
        if np.all(crop == d):
            w = '3'
            cell_list.append(w)
            
    for e in ver4:
        if np.all(crop == e):
            w= '4'
            cell_list.append(w)
    for r in ver5:
        if np.all(crop == r):
            w= '5'
            cell_list.append(w)
    
    #cell_list.append(celltext)
    show_images([crop])
    print(w)


# In[87]:


print(cell_list[1])


# In[89]:


################Excel###################

arr = np.array(cell_list)
dataframe = pd.DataFrame(arr)
print(dataframe)
data = dataframe.style.set_properties(align="left")
# Converting it in a excel-file
data.to_excel("output.xlsx",index=None,header="code")


# In[295]:


############# ML ####################
# save cells:
img6 = cells_img[1]
show_images([img6])

directory = '/testing'
output_directory = '/testing'

PRINT_SLICES = False
THRESHOLD_PIXELS_COUNT = 60000
MAX_BOUNDING_BOX_WIDTH = 675
MAX_BOUNDING_BOX_HEIGHT = 50

def detect_left_edge(image):
    h,w = image.shape
    max = 0
    edge = 0
    for x in range(0,100):

        vertical_slice = image[0:h,  x:x+15 ]
        vertical_slice_pixels_count = vertical_slice.sum()

        if( vertical_slice_pixels_count > THRESHOLD_PIXELS_COUNT):
            scipy.misc.imsave(output_directory + '/' + filename, image[0:h, x:675])
            return 0

        if (vertical_slice_pixels_count > max):
            max = vertical_slice_pixels_count
            edge = x * 2
    return edge

# loop over images
for filename in os.listdir(directory):
    if filename.endswith("negate.jpg"):
        input_image = scipy.misc.imread(directory + '/' + filename)
        image_height, image_width = input_image.shape
        max = 0;
        output_image = input_image
        # detect top edge of the image bounding box
        for h in range(0, image_height - MAX_BOUNDING_BOX_HEIGHT):
            temp_image = input_image[h:h + MAX_BOUNDING_BOX_HEIGHT, 0:0 + MAX_BOUNDING_BOX_WIDTH]
            if temp_image.sum() > max:
                max= temp_image.sum()
                output_image = temp_image
        edge = detect_left_edge(output_image)


# In[ ]:



# # Creating two lists to define row and column in which cell is located
# row = []
# column = []
# j = 0

# # Sorting the boxes to their respective row and column
# for i in range(len(box)):
#     if(i == 0):
#         column.append(box[i])
#         previous = box[i]
#     else:
#         if(box[i][1] <= previous[1]+mean/2):
#             column.append(box[i])
#             previous = box[i]

#             if(i == len(box)-1):
#                 row.append(column)
#         else:
#             row.append(column)
#             column = []
#             previous = box[i]
#             column.append(box[i])
# print(column[0][0])
# print(row)
# print(box)
# # img4 = img[column[0][2]:column[0][3],column[0][0]:column[0][1]]


# In[ ]:



# # calculating maximum number of cells
# countcol = 0
# for i in range(len(row)):
#     countcol = len(row[i])
#     if countcol > countcol:
#         countcol = countcol

# # Retrieving the center of each column
# center = [int(row[i][j][0]+row[i][j][2]/2)
#           for j in range(len(row[i])) if row[0]]

# center = np.array(center)
# print(len(center))
# center.sort()
# print(center)
# # Regarding the distance to the columns center, the boxes are arranged in respective order
# img5 = img[:,:252]
# show_images([img5])


# In[ ]:



# finalboxes = []
# for i in range(len(row)):
#     lis = []
#     for k in range(countcol):
#         lis.append([])
#     for j in range(len(row[i])):
#         diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
#         minimum = min(diff)
#         indexing = list(diff).index(minimum)
#         lis[indexing].append(row[i][j])
#     finalboxes.append(lis)

# # finalboxes = box
# # print(finalboxes)
# # print(box)
# print(finalboxes[0])
# print(box[1])


# In[ ]:



# # from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
# outer = []
# for i in range(len(finalboxes)):
#     for j in range(len(finalboxes[i])):
#         inner = ''
#         if(len(finalboxes[i][j]) == 0):
#             outer.append(' ')
#         else:
#             for k in range(len(finalboxes[i][j])):
#                 y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], finalboxes[i][j][k][3]
#                 finalimg = bitnot[x:x+h, y:y+w]
#                 kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
#                 border = cv2.copyMakeBorder(
#                     finalimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
#                 resizing = cv2.resize(border, None, fx=2,
#                                       fy=2, interpolation=cv2.INTER_CUBIC)
#                 dilation = cv2.dilate(resizing, kernel, iterations=1)
#                 erosion = cv2.erode(dilation, kernel, iterations=2)

#                 out = pytesseract.image_to_string(erosion)
#                 if(len(out) == 0):
#                     out = pytesseract.image_to_string(
#                         erosion, config='--psm 3')
#                 inner = inner + " " + out
#             outer.append(inner)

# # Creating a dataframe of the generated OCR list
# arr = np.array(outer)
# dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
# print(dataframe)
# data = dataframe.style.set_properties(align="left")
# # Converting it in a excel-file
# data.to_excel("E:\CUFE\Fall22\IP\Project-2022\GradesAutoFiller-main\GradesAutoFiller-main\Module1\output.xlsx")


# In[ ]:




