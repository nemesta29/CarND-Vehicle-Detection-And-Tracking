
## Proect Writeup
### The following write-up shall explain the thought process involved in making and completeing this project

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]:  ./69.png
[image2]:  ./extra59.png
[image3]: ./hog1.jpg
[image4]: ./2.jpg
[image5]: ./t1.jpg
[image6]: ./t3.jpg
[image7]: ./t4.jpg
[image8]: ./t1.jpg
[image9]: ./h1.jpg
[image10]: ./t3.jpg
[image11]: ./h3.jpg
[image12]: ./t4.jpg
[image13]: ./h4.jpg
[image14]:./t6.jpg
[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup 

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook as the function `get_hog_features`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

Vehicle Class
![alt text][image1]

Non - Vehicle Class
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. I changed the color_spaces from *RGB*, to *HSV*, *HLS* and even experimented with *YUV* and *YCrCb*. I tried individual channels and all channels combined with different combinations of parameters. While using *YUV*, I also found that `transform_sqrt` needed to be set to *False* to preserve the properties of that colorspace

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` and the hog_channel as *Y* channel:


![alt text][image2]

#### 2. How I settled on your final choice of HOG parameters.

I tried various combinations of parameters and colorspaces and displayed HOG outputs for the various scenarios. Some depicted the outlines of cars vividly while others displayed a general mediocrity in the gradient of the whole image. I checked how distincly the HOG defined vehicles, and compared it with detections of non-car elements to ensure a differentiation was possible. Changing various parameters, seeing how the gradients smoothed out, displaying cars in an identifiable form, I chose the *Y* channel of *YCrCb* channel with the parameters as `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`

#### 3.Traing a classifier - Linear SVM.

I trained a linear SVM for classfying images between cars and non-cars, in code cell 6. As a machine learning technique, I had to extract features to help the SVM learn how to differentiate a car from a non-car. I used spatial binning, color histograms and HOG of images in the traing data set, extracting 4932 features per image. The model was then trained with an 80-20 split of data as training and test data, to return an accuracy of 99.46%. A `StandardScaler` from `sklearn.preprocessing` to normalise the feature-set and avoid favouritism in features.

### Sliding Window Search

#### 1. Sliding window search

To detect cars in the video stream, i implemented a sliding window search. It ran in the lower half of the image, as cars would be found only in the lower half of the image, not the sky. I implemented the code for it using the helper functions provided in the classroom material. Additionally, I incremented the width and height by 20 px, after each row-wise pass. That way the higher regions, closer to the horizon, were covered by a small window and the regions closer to the car, were covered by larger windows. This accomodated perspective differences and allowed for smoother detection of cars on the road.

*Note : Due to operational limitations at Infosys(where I do the course), the video processing time was too high for a complete video, so the region was reduced to include only the right half of the image, for faster processing*

![alt text][image4]

#### 2. Examples and optimizing the performance of my classifier

Ultimately I searched on ta single incremental scaling using YCrCb single-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Taking only the first channel of YCrCb allowed for elimination of certain false positives, which helped improve the performance. I also using image scaling, to help convert the difference in reading .png and .jpg images for the training dataset and the actuat images respectively.  Here are some example images:

[alt text][image5]
![alt text][image6]
![alt text][image7]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. An average of the past 15 frames was used with a threshold of 4. The heatmap technique helped eliminate false positives and allowed for a more consistent detection over the frames, removing sudden changes in the bounding box.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are three frames and their corresponding heatmaps:

![alt text][image8]
![alt text][image9]

![alt text][image10]
![alt text][image11]

![alt text][image12]
![alt text][image13]

We can see that the heatmap has been drawn to represent consistent occurence and an average of heat map is used to display the final image with combined accuracy.
![alt text][image14]
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

I took a visual approach, where I analysed the data and it's properties and built a system to suit the present needs. I picked features that would aide in better identification of vehicle or non-vehicle and worked to solve the problem. The following are areas where the pipeline might fail:
* A change of lanes by the vehicle may lead to less accurate detections. This can be resolved by building a pipeline with a more robust engine to work with
* A change in elevation or road design can lead to incorrect region of interest and causing misreads. Such an issue should be addressed through a proper generalization

One possible solution/ area of improvement:
* To replace the machine learning procedure with a CNN or similar architecture that involves deep learning. By doing so, we shall only have to provide data to train the model. No feature extraction or other processing would be required and the system will be more generalised based on training data used.


