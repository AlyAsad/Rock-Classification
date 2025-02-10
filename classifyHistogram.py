import sys
import ctypes
import numpy as np
import cv2
import csv
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.append("helperFiles/MvImport")
from MvCameraControl_class import *



### GLOBAL VARIABLES
coordsX, coordsY = 0.0, 0.0
pressedMouse = False
rectCoords = [[0, 0], [0, 0]]

scaler = StandardScaler()
knn = KNeighborsClassifier(n_neighbors = 3) # using 3 neighbours
predicted = -1









# Initialize Camera SDK
MvCamera().MV_CC_Initialize()

# Enumerate Devices
deviceList = MV_CC_DEVICE_INFO_LIST()
ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)

if ret != 0:
    print(f"Device enumeration failed! Error code: 0x{ret:X}")
    sys.exit()

if deviceList.nDeviceNum == 0:
    print("No camera devices found.")
    sys.exit()

print(f"Found {deviceList.nDeviceNum} device(s).")

# Get First Device
stDeviceList = ctypes.cast(deviceList.pDeviceInfo[0], ctypes.POINTER(MV_CC_DEVICE_INFO)).contents

# Create Camera Object
cam = MvCamera()
ret = cam.MV_CC_CreateHandle(stDeviceList)
if ret != 0:
    print(f"Failed to create handle! Error code: 0x{ret:X}")
    sys.exit()

# Open Device
ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
if ret != 0:
    print(f"Failed to open device! Error code: 0x{ret:X}")
    cam.MV_CC_DestroyHandle()
    sys.exit()

# Set Camera Parameters
cam.MV_CC_SetFloatValue("ExposureTime", 60000.0)  # set exposure time
cam.MV_CC_SetEnumValue("GainAuto", 0)  # disable auto gain

# Start Grabbing
ret = cam.MV_CC_StartGrabbing()
if ret != 0:
    print(f"Failed to start grabbing! Error code: 0x{ret:X}")
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    sys.exit()

print("Camera is grabbing frames... Press ESC to exit.")




def getOpenCVImage():
    # Initialize frame buffer
    stOutFrame = MV_FRAME_OUT()
    ctypes.memset(ctypes.byref(stOutFrame), 0, ctypes.sizeof(stOutFrame))

    ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
    if ret != 0:
        print(f"Failed to get image buffer! Error code: 0x{ret:X}")
        exit()

    # Convert to OpenCV Image
    buf_cache = (ctypes.c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
    ctypes.memmove(ctypes.byref(buf_cache), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)

    width, height = stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight
    scale_factor = min(1920 / width, 1080 / height)

    np_image = np.ctypeslib.as_array(buf_cache).reshape(height, width)
    cv_image = cv2.cvtColor(np_image, cv2.COLOR_BayerBG2BGR)
    cv_image = cv2.resize(cv_image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    cam.MV_CC_FreeImageBuffer(stOutFrame)  # Free buffer after use
    
    return cv_image



def mouse_callback(event, x, y, flags, param):
    global coordsX, coordsY, pressedMouse
    if event == cv2.EVENT_LBUTTONDOWN:  # left mouse button click
        coordsX, coordsY = x, y
        pressedMouse = True




def extractHistogram(imagePath):
    
    # first convert image to HSV
    image = cv2.imread(imagePath)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # now i find the histogram for each h, s, v channel
    h_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 256]).flatten()
    s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256]).flatten()
    
    # flattening the histograms
    features = np.concatenate((h_hist, s_hist, v_hist))
    
    return features


# processes all images in the class_x folders and puts output in rock_data_from_input.csv
def process_dataset():
    
    labels = {"class_1": 1, "class_2": 2, "class_3": 3}
    
    with open("helperFiles/rock_data_from_input.csv", mode = "w", newline = "") as file:
        writer = csv.writer(file)
        
        # for each class and each image
        for (className, label) in labels.items():
            for imageNum in range(1, 16): # change if you change the number of images
                
                imagePath = f"images/{className}/{imageNum}.png"
                print(f"Processing image: {imagePath}")
                features = extractHistogram(imagePath)
                
                writer.writerow(list(features) + [label])
                
                
                
    
    


# trains the knn model using the 768 features in rock_data_from_input.csv
def trainModel():
    
    global scaler, knn
    
    data = pd.read_csv("helperFiles/rock_data_from_input.csv", header=None)
    X = data.iloc[:, :-1].values  # all columns except last one
    y = data.iloc[:, -1].values   # last column is labels
    
    # normalizing values
    X = scaler.fit_transform(X)
    
    # training model
    knn.fit(X, y)
    





def showWithClassification(cv_image):
    
    global coordsX, coordsY, pressedMouse, rectCoords
    global scaler, knn, predicted
    
    if (predicted != -1):
        cv2.rectangle(cv_image, (rectCoords[0][0], rectCoords[1][0]), (rectCoords[0][1], rectCoords[1][1]), (255, 0, 255), 2)
        
        labels_dict = {1: "White rock", 2: "Orange rock", 3: "Black rock"}
        rockName = labels_dict[predicted]
        cv2.putText(cv_image, f"Predicted: {predicted}) {rockName}", (coordsX - 100, coordsY - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    cv2.imshow("Click on object for classification", cv_image) 
    cv2.setMouseCallback("Click on object for classification", mouse_callback, param=cv_image)
    
    # if mouse pressed, get details
    if pressedMouse == False:
        return
    
    # else pressedMouse == True
    pressedMouse = False
    
    ## getting histogram of HSV
    
    h, w, _ = cv_image.shape
    half_window = 50  # distance from clicked pixel, increasing will increase number of pixels checked
    
    rectCoords[0][0] = max(coordsX - half_window, 0) # x start
    rectCoords[0][1] = min(coordsX + half_window, w - 1) # x end
    rectCoords[1][0] = max(coordsY - half_window, 0) # y start
    rectCoords[1][1] = min(coordsY + half_window, h - 1) # y end

    # extract the region's colors
    region = cv_image[rectCoords[1][0]:rectCoords[1][1], rectCoords[0][0]:rectCoords[0][1]]
    
    #region to HSV
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # compute histogram features
    h_hist = cv2.calcHist([hsv_region], [0], None, [180], [0, 256]).flatten()
    s_hist = cv2.calcHist([hsv_region], [1], None, [256], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv_region], [2], None, [256], [0, 256]).flatten()

    # flatten
    features = np.concatenate((h_hist, s_hist, v_hist)).reshape(1, -1)

    # normalize
    features = scaler.transform(features)

    # predict
    predicted = knn.predict(features)[0]
    




# use only when you change the class images in the images folder, ive already processed the current ones
#process_dataset()





trainModel()

# main loop
while True:
    
    # getting the image from camera
    cv_image = getOpenCVImage()

    # classifying
    showWithClassification(cv_image)
    
    # press ESC to exit
    if cv2.waitKey(1) == 27:
        break


# stop grabbing & release resources
cam.MV_CC_StopGrabbing()
cam.MV_CC_CloseDevice()
cam.MV_CC_DestroyHandle()
cv2.destroyAllWindows()
print("Camera resources released.")
