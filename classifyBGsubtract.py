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
background = None

scaler = StandardScaler()
knn = KNeighborsClassifier(n_neighbors = 3) # using 3 neighbours









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
cam.MV_CC_SetFloatValue("ExposureTime", 20000.0)  # set exposure time
cam.MV_CC_SetEnumValue("GainAuto", 1)  # enable auto gain

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
    


def capture_background():
    global background
    print("Capturing background, please ensure no objects are in view...")
    background = getOpenCVImage()
    print("Background captured.")



def apply_background_subtraction(frame):
    global background
    
    threshold_value = cv2.getTrackbarPos("Threshold", "Trackbars")
    kernel_size = cv2.getTrackbarPos("Kernel", "Trackbars")

    # converting to gray for better accuracy
    gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fg_mask = cv2.absdiff(gray_background, gray_frame)
    
    _, thresh = cv2.threshold(fg_mask, threshold_value, 255, cv2.THRESH_BINARY)
    
    cv2.namedWindow("After thresholding", cv2.WINDOW_NORMAL)
    cv2.imshow("After thresholding", thresh)

    # applying morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
    
    cv2.namedWindow("After morph", cv2.WINDOW_NORMAL)
    cv2.imshow("After morph", clean_mask)

    return clean_mask




def showWithClassification(frame):
    
    mask = apply_background_subtraction(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_contour_area = cv2.getTrackbarPos("Contour", "Trackbars")

    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:  # ignore small noise
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    
    cv2.namedWindow("Final", cv2.WINDOW_NORMAL)
    cv2.imshow("Final", frame)

    
    
    




# use only when you change the class images in the images folder, ive already processed the current ones
#process_dataset()





capture_background()
trainModel()

cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Threshold", "Trackbars", 50, 255, lambda x : None)
cv2.createTrackbar("Kernel", "Trackbars", 5, 20, lambda x : None)
cv2.createTrackbar("Contour", "Trackbars", 500, 5000, lambda x : None)


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
