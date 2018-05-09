# **Vehicle Detection Project** 
---

Pongrut Palarpong  
May 11, 2018

---
![cam_calibration](./output_images/cam_calibration.jpg)


The goals / steps of this project are the following:

* The goal is to write a software pipeline to identify vehicles in a video from a front-facing camera on a car. 



## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

The above pipeline using HOG features and a linear SVM is well-known since 2005. Very recently extremely fast neural network based object detectors have emerged which allow object detection faster than realtime. I merely cloned the original darknet repository and applied YOLO to the project video. I only needed to do a minor code modification to allow saving videos directly. The result is quite amazing. As no sliding windows are used the detection is extremely fast. A frame is passed to the network and processed precisely once, hence the name YOLO — “you only look once”.

### Histogram of Oriented Gradients (HOG) vs. Convolutional Neural Network (CNN)

The histogram of gradients (HOG) is a descriptor feature. The HOG algorithm will check every pixel about how much darker the surrounding pixels are and then specify the direction that pixel is getting darker, then counts the occurrences of gradient orientation in localized portions of an image. The HOG result is features that use in support vector machine for the classification task.


![HOG](./figures/HOG.jpg)<br/>
HOG Features Visualization

![car_color_hist](./figures/car_color_hist.jpg)
![notcar_color_hist](./figures/notcar_color_hist.jpg)
Color Histogram Features Visualization

YOLO Real-Time Object Detection apply convolutional neural network architecture to classify an object. CNN architecture suitable for image classification because the image is indeed 2D width and height. CNN can do convolution operation by sweeping the relationship between each part of the image and creating essential filters. This convolution operation makes it easy for CNN to detect objects in multiple locations, difference lightings, or even just some part of objects in an image.

![CNN](https://www.mathworks.com/content/mathworks/www/en/discovery/deep-learning/jcr:content/mainParsys/band_2123350969_copy_1983242569/mainParsys/columns_1635259577/1/image_2128876021_cop.adapt.full.high.svg/1508444613873.svg)
Example: A network with many convolutional layers

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)<br/>
[![](http://img.youtube.com/vi/m24QWDf1TRQ/0.jpg)](http://www.youtube.com/watch?v=m24QWDf1TRQ "Advanced Lane Finding")


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

