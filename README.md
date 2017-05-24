# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, the goal is to write a software pipeline to detect vehicles and track them in a video.

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.  
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[image1]: ./output_images/car_image.JPG "car"
[image2]: ./output_images/noncar_image.JPG "non car"
[image3]: ./output_images/hog_ch_3.JPG "HOG"
[image4]: ./output_images/detection.JPG "detections"
[image5]: ./output_images/false_positives.JPG "false pos"


#### Preparation for the project

I started with reading in images from the links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train my classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself. 

An example of car(vehicle) and non-car(non-vehicle) image is shown below

![alt text][image1] ![alt text][image2]

This is done in cell 1 of the iPython notebook and we get the following distribution between car and non-car images:

**No.of car images: 8792**

**No.of non car images: 8968**


#### Feature Extraction
Next step in the process was to perform Histogram of Oriented Gradients (HOG) feature extraction on the labeled training set of images retrieved in the above section. Along with the HOG features, I also used binned color features and histogram of colors to and created a combined feature vector using **extract\_features()** function. These are defined in cell 3 of the iPython notebook.

Here is an example of running HOG feature extraction over a car and a non car image.

![alt text][image3]

#### SVM Classifier

The classifier that I used for training on the sample images is an Support Vector Machines (SVM) classifier. This is done in cell 5 of the notebook.

After a few trials on the right parameters and finding out the training accuracy of my SVM classifier to be around 98.87% I reached at the parameter values as shown below.

color\_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

orient = 9  # HOG orientations

pix\_per\_cell = 8 # HOG pixels per cell

cell\_per_\block = 2 # HOG cells per block

hog\_channel = 0 # Can be 0, 1, 2, or "ALL"

spatial\_size = (32, 32) # Spatial binning dimensions

hist\_bins = 32    # Number of histogram bins

spatial\_feat = True # Spatial features on or off

hist\_feat = True # Histogram features on or off

hog\_feat = True # HOG features on or off

y\_start\_stop = [None, None] # Min and max in y to search in slide\_window()


#### Sliding Window
Also included in cell 3 is the **sliding\_window()** function which takes in an image, start and stop positions of the window, size of the window and overlap as parameters. This function creates a list of window positions based on the parameters that were passed. This list will be useful in the next functions.

#### Search Windows
This function also in cell 3, takes the list of window positions and the image along with parameters required for extracting features. It uses this information and for each window, it passes the window of size 64X64 pixels to a function called **single\_image\_features()** which extracts features for that window.

These features are then passed through the trained SVM classifier which outputs if it is a car or a non-car image. All the windows positively detected for cars are then appended and returned as a list from the function.

#### Checking detection of vehicles in test images using the trained SVM classifier

Cell 7 of the notebook verifies if the above functions are working on some sample images and detecting vehicles or not. An example of all windows running on 2 sample images show detection and non detections.

![alt text][image4]

#### Heat Maps & False Positives

During the detection, I also encountered a lot of false positive detections of cars. This resulted in boxes being drawn at non-car locations. In order to reduce false positives, I created a heat map which increases heat on a blank image in areas where there are overlapping windows for positive detections. This will help in reducing false positives by applying a threshold on the values of the heat map images. The thresholded heatmap was then used in drawing boxes that lead to more robust detection of cars VS false positive cars.

An example of heatmap implementation on sample images is shown in cell 8. Shown below is an example of a false positive detection and its corresponding heat map which is darker than true positives. The idea of heat map thresholding is to threshold these darker areas and remove false positive detections.

![alt text][image5]

#### Classes for averaging and smoothing frame-to-frame drawing boxes

While running the above pipeline on the project video, one would notice jumping boxes since my pipeline would detect vertices of heat map drawn boxes jump frame-to-frame. This led me to define classes that would save heatmap history and also box vertices history. These classes are defined in cell 9.

Also defined in cell 9 are functions that help improve the smoothing of boxes along with the above classes. In order to improve smoothing, my 1st method includes averaging the heat map over certain frames of images, in my case which is 3 frames. The 2nd method of improving smoothing is by averaging the vertices of boxes detected by **draw\_labeled\_bboxes\_1()** function. This function utilises the **Vehicle()** class to accumulate history of vertices over frames. This function is in cell 9 as well. 

These include an **add\_heat()** function that adds heat to an image, **remove\_heat()** removes heat when detection of heat maps crosses a threshold of certain number of frames.

#### Find Cars function

This function in cell 10 bundels a lot of above mentioned function into one. This function carries out cropping of unnecessary areas along y axis. Also I am cropping a portion of the left sides of the frames since I noticed there were false positives due to the lanes on the left side opposite lanes. This scheme obviously needs to be automated in an actual system, which kicks in when it detects if the car is driving in the left most lane and is off otherwise.

**find\_cars()** function then applies appropriate color transformation on cropped frames, extracts features and uses the classifier to detect if there are cars in the windows over the image frames. The function then returns a list of boxes that can be used in the main pipeline for video.

#### PIPELINE

Cell 11 contains the main pipeline that works on generating the **results\_video.mp4** file. The pipeline has the following steps:

1. Get a list of boxes for the frame passed to it

2. If boxes are detected in the frame then add heat to those areas of detection

3. Since the pipeline is utilizing the **heat\_it\_up()** class it looks for averaging the heat map over 3 frames in my case.

4. After averaging heat map over certain number of frames, it then applies a threshold on the heat map to reduce false positives.

5. Then apply label [function](https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html) from scipy library which finds out the areas of connected pixels and labels them as return parameters. Which means roughly it tells us the number of cars detected and thier locations in the image frame

6. Then lastly the **draw\_labeled\_bboxes\_1()** function draws boxes on the image frame

#### Final Results

Cell 12 applies the pipeline over all the image frames for the **project\_video.mp4** file. The results are located in this same repository as **results\_video.mp4** and with this YouTube [link](https://youtu.be/O-GPZk4yWsk)

#### Improvements and Future Work

1. Although my boxes detection is not jumping but due to flickering I think I can still improve the averaging functions and therefor improve the smoothness of my boxes highlighting detection.

2. Also the pipeline handles false positives in shadow areas noticeably well. However, when the car transitions to a different color road like the bridge in the video, it fails to detect the vehicle in those frames. I think learnings from previous project on using color spaces well can improve this area too.

