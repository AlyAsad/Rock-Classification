import sys
import ctypes
import numpy as np
import cv2
import csv
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


sys.path.append("helperFiles/MvImport")
from MvCameraControl_class import *



### GLOBAL VARIABLES
edge = 100
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
cam.MV_CC_SetFloatValue("ExposureTime", 20000.0)  # eet exposure time
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



def showVideo(cv_image):
    cv2.imshow("Live video", cv_image) 
    


def showEdges(cv_image):
    global edge
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, edge, edge)
    cv2.imshow("Edges", edges)
    edge -= 1

    if(edge < 30):
        edge = 100



def mouse_callback(event, x, y, flags, param):
    global coordsX, coordsY, pressedMouse
    if event == cv2.EVENT_LBUTTONDOWN:  # left mouse button click
        coordsX, coordsY = x, y
        pressedMouse = True


def saveToCSV(value, label):
    
    with open("helperFiles/rock_data_Avg.csv", mode = "a", newline = "") as file:
        writer = csv.writer(file)
        writer.writerow([value[0], value[1], value[2], label])
    
    print("saved")
    


def showWithTraining(cv_image):
    global coordsX, coordsY, pressedMouse, rectCoords
    
    cv2.rectangle(cv_image, (rectCoords[0][0], rectCoords[1][0]), (rectCoords[0][1], rectCoords[1][1]), (255, 0, 255), 2)
    cv2.imshow("Click on object for training", cv_image) 
    cv2.setMouseCallback("Click on object for training", mouse_callback, param=cv_image)
    
    # if mouse pressed, get clicked and average color
    if pressedMouse == False:
        return
    
    # else pressedMouse == True
    pressedMouse = False
    
    # getting average color
    h, w, _ = cv_image.shape
    half_window = 50  # distance from clicked pixel, increasing will increase number of pixels checked
    
    rectCoords[0][0] = max(coordsX - half_window, 0) # x start
    rectCoords[0][1] = min(coordsX + half_window, w - 1) # x end
    rectCoords[1][0] = max(coordsY - half_window, 0) # y start
    rectCoords[1][1] = min(coordsY + half_window, h - 1) # y end

    # extract the region's colors and calculate the average HSV values
    region = cv_image[rectCoords[1][0]:rectCoords[1][1], rectCoords[0][0]:rectCoords[0][1]]
    print(f"Region shape: {region.shape}")
    colorsHSV = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    avgHSV = np.mean(colorsHSV, axis = (0, 1))
    print(f"Average HSV of region: {avgHSV}")
    
    # printing the update
    cv2.rectangle(cv_image, (rectCoords[0][0], rectCoords[1][0]), (rectCoords[0][1], rectCoords[1][1]), (255, 0, 255), 2)
    cv2.imshow("Click on object for training", cv_image)
    cv2.waitKey(10)
    
    # labelling the data
    print("\nEnter label for scanned object:")
    print("1) White rock")
    print("2) Orange rock")
    print("3) Black rock")
    print("Input anything else to discard image")
    label = input("Your input: ")
    
    # so it disables the rectangle when label taking done
    rectCoords = [[0, 0], [0, 0]]
    
    # if need to discard image
    if label not in ["1", "2", "3"]:
        print("discarded")
        return
    
    # now to actually save record to CSV
    saveToCSV(avgHSV, label)
        





def showWithClassification(cv_image):
    
    global coordsX, coordsY, pressedMouse, rectCoords
    global scaler, knn, predicted
    
    rockName = ""
    
    
    if (predicted != -1):
        cv2.rectangle(cv_image, (rectCoords[0][0], rectCoords[1][0]), (rectCoords[0][1], rectCoords[1][1]), (255, 0, 255), 2)
        
        labels_dict = {1: "White rock", 2: "Orange rock", 3: "Black rock"}
        rockName = labels_dict[predicted]
        cv2.putText(cv_image, f"Predicted: {predicted}) {rockName}", (coordsX - 100, coordsY - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
    cv2.imshow("Click on object for classification", cv_image) 
    cv2.setMouseCallback("Click on object for classification", mouse_callback, param=cv_image)
    
    # if mouse pressed, get clicked and average color
    if pressedMouse == False:
        return
    
    # else pressedMouse == True
    pressedMouse = False
    
    # getting average color
    h, w, _ = cv_image.shape
    half_window = 50  # distance from clicked pixel, increasing will increase number of pixels checked
    
    rectCoords[0][0] = max(coordsX - half_window, 0) # x start
    rectCoords[0][1] = min(coordsX + half_window, w - 1) # x end
    rectCoords[1][0] = max(coordsY - half_window, 0) # y start
    rectCoords[1][1] = min(coordsY + half_window, h - 1) # y end

    # extract the region's colors and calculate the average HSV values
    region = cv_image[rectCoords[1][0]:rectCoords[1][1], rectCoords[0][0]:rectCoords[0][1]]
    print(f"Region shape: {region.shape}")
    colorsHSV = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    avgHSV = np.mean(colorsHSV, axis = (0, 1))
    print(f"Average HSV of region: {avgHSV}")
    
    
    # now running the model to classify new data
    data = scaler.transform([avgHSV])
    
    predicted = knn.predict(data)[0]
    
    
    
        
        
        


def trainKNNModel():
    
    global scaler, knn
    
    data = pd.read_csv("helperFiles/rock_data_Avg.csv", header = None)
    X = data.iloc[:, :3].values # first 3 columns are features
    y = data.iloc[:, 3].values # last column is labels
    
    # normalizing values
    X = scaler.fit_transform(X)
    
    # training model
    knn.fit(X, y)
    
    
    



choice = "1"
#UNCOMMENT THE BELOW LINE IF YOU WANT TO ASK FOR TRAINING
#choice = input("Do you want to [0] train or [1] classify: ")



# training the model if classification selected
if choice == "1":
    trainKNNModel()
    
    


# main loop
while True:
    
    # getting the image from camera
    cv_image = getOpenCVImage()

    # showing live video
    #showVideo(cv_image)
    
    
    # edge detection
    #showEdges(cv_image)
    
    
    # train or classify color on mouse click
    
    if choice == "0":
        showWithTraining(cv_image)
    
    elif choice == "1":
        showWithClassification(cv_image)
    
    else:
        print("Wrong input!")
        break


    
    
    
    
    if cv2.waitKey(1) == 27:  # press ESC to exit
        break


# stop grabbing & release resources
cam.MV_CC_StopGrabbing()
cam.MV_CC_CloseDevice()
cam.MV_CC_DestroyHandle()
cv2.destroyAllWindows()
print("Camera resources released.")
