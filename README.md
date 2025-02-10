# Rock Classification


A project I coded while working at Aremak, Turkey.



## What it does
1) Firstly, it connects to an external camera that is attached to the computer. The camera should have the rocks to classify in its view.

2) Then, you can choose whether to use average HSV values of the selected region or histogram values of the selected region for classification (histogram will have more accuracy but higher number of features in KNN model).

3) With classification, it can classify the different types of rocks by using their HSV color values. The `classifyBGsubtract.py` file has the most advanced and accurate code.


## How to use
1) Install the dependencies like numpy, pandas, sklearn.
   
2) Make sure the external camera is connected and setup with proper lighting/exposure.
   
3) Run either:
   * `classifyAvg.py` to classify using average values of the user selected region,
   * `classifyHistogram.py` to classify using a histogram of the user selected region,
   * `classifyBGsubtract.py` to detect and classify objects automatically within the camera frame.

4) For `classifyAvg.py` and `classifyHistogram.py`, a window will pop up, you can click on the region which you want to scan for classification, and it will give the prediction. For `classifyBGsubtract.py`, you need to first clear the viewing frame and press 1 to save it as background. Then, you can put objects into the frame and it will automatically classify them. You can also change some parameters in the pop-up window to make object detection more accurate.


## Equipment/Technologies used:
- 📷 **Camera Model:** Hikrobot MV-CS060-10UC-PRO
- 🔬 **Lens:** MVL-HF0828M-6MPE
- 🏗 **Camera Stand:** Aremak Adjustable Machine Vision Test Stand
- 💡 **Lighting:** Hikrobot Shadowless Ring Light(MV-LGES-116-W)
- 🖥️ **Operating System:** Windows
- 🔧 **Software Tools:** Python, OpenCV, Hikrobot SDK, CSV, Pandas, scikit-learn


## Setup photos

Image of the camera setup:
![Setup Image](images/setup.jpeg)

Image of the rocks used for training and testing:
![Rocks Image](images/rock_samples.jpeg)




## Acknowledgment
🏢 This project was developed during an internship at [Aremak Bilişim Teknolojileri](https://www.aremak.com.tr) under the supervision of Emrah Bala.
