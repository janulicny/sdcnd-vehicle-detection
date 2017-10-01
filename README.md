# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[cars]: ./examples/cars.png
[noncars]: ./examples/noncars.png
[hog]: ./examples/hog.png
[roc]: ./examples/roc.png
[learn]: ./examples/learn.png
[w1]: ./examples/w1.png
[w2]: ./examples/w2.png
[w3]: ./examples/w3.png
[w4]: ./examples/w4.png
[w5]: ./examples/w5.png
[w6]: ./examples/w6.png
[w7]: ./examples/w7.png
[test1]: ./examples/test1.png
[test2]: ./examples/test2.png
[test3]: ./examples/test3.png
[test4]: ./examples/test4.png
[diag1]: ./examples/diag1.png
[diag2]: ./examples/diag2.png
[diag3]: ./examples/diag3.png
[diag4]: ./examples/diag4.png
[diag5]: ./examples/diag5.png
[diag6]: ./examples/diag6.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.


I started by reading in all the `vehicle` and `non-vehicle` images. While exploring the dataset, I realized that the size of car and classes is roughly the same. Since in the real world I expect the noncar class to be detected much more often than the car class, I was afraid that the classifier might be prone to detect cars even when there are none. To resolve this, I decide to augment the data set to reduce the overrepresentation of the car class.

I rotated every non car image by 90°, 180° and 270°. On top of that I blurred both car and non car images to prevent the model to learn from noise patterns. This step is contained in code cell 2 of the IPYthon notebook.

After the data augmentation, I was left with this data set:

* Number of car images: 8792
* Number of non car images: 35872 


Here are some examples of the augmented samples from `vehicle` class:

![alt text][cars]

And here of `non-vehicle` class:

![alt text][noncars]

#### 2. Explain how you settled on your final choice of HOG parameters.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) to decide on the best features for the classificator.

After trying out different parameters on smaller subsets of the data set, I was getting inconlusive results, so I just decided to brute-force it and really find out, which parameter combination works best.

I decided to search in the following parameter space:

* Color space: RGB, HSV, LUV, HLS, YUV or YCrCb
* Color channel: 0, 1, 2 or all
* Gradient orientations: 4, 9, 11, 15
* Pixels per cell: 4, 8, 16
* Cells per block: 2, 4, 8

To test all combination would be too intensive, so I decided to split the search into two phases with different parameters frozen.

#### Search for color space and color channel

First, I decided to search for suitable color space and color channels. The other parameters were fixed at following values:

* Gradient orientations: 11
* Pixels per cell: 16
* Cells per block: 2

These values were chosen based on the previous experience, because they limit the feature vector size, while still providing adequate predicting capability.

The classifier (will be described later in this report) was trained on 80% of the augmented dataset and the remaining 20% served as validation set on which classifier's accuracy and precision were evaluated. The result of this first round can be found in the table below.

Color space | Color channel | Accuracy | Precision
---|---|---|---
YUV | ALL | 98.88% | 96.93%
YCrCb | ALL | 98.86% | 96.82%
HSV | ALL | 98.74% | 96.42%
HLS | ALL | 98.75% | 96.36%
LUV | ALL | 98.81% | 96.19%
RGB | ALL | 98.09% | 94.09%
HSV | 2 | 97.26% | 91.93%
HLS | 1 | 97.21% | 91.93%
YUV | 0 | 97.16% | 91.87%
YCrCb | 0 | 97.16% | 91.87%
RGB | 1 | 97.19% | 91.81%
LUV | 0 | 97.17% | 91.76%
YCrCb | 1 | 97.21% | 91.47%
YUV | 2 | 97.22% | 91.47%
RGB | 2 | 97.12% | 91.13%
LUV | 1 | 96.91% | 91.13%
RGB | 0 | 96.78% | 90.96%
HSV | 1 | 95.75% | 88.63%
YUV | 1 | 96.16% | 88.52%
YCrCb | 2 | 96.18% | 88.40%
LUV | 2 | 95.91% | 87.66%
HSV | 0 | 95.57% | 87.09%
HLS | 0 | 95.59% | 86.92%
HLS | 2 | 95.04% | 86.64%

As you can see, the best result was accomplished when using all channels of YUV color space, so I decided to stick with it.

#### Search for the parameters of HOG extract function

In the second round, I kept the YUV color space and all color channels fixed and tested the remaining parameters. The results are recorded in the table below.


Orientations | Pixels per cell | Cells per block | Accuracy | Precision
---|---|---|---|---
15 | 16 | 2 | 99.12% | 97.38%
11 | 16 | 2 | 98.88% | 96.93%
9 | 16 | 2 | 98.88% | 96.53%
4 | 8 | 2 | 98.54% | 95.96%
4 | 16 | 2 | 97.91% | 94.26%
15 | 16 | 4 | 98.21% | 93.92%
15 | 8 | 8 | 98.19% | 93.41%
9 | 16 | 4 | 97.88% | 93.35%
9 | 8 | 8 | 98.11% | 93.12%
11 | 16 | 4 | 97.91% | 93.06%
11 | 8 | 8 | 98.09% | 92.95%
15 | 32 | 2 | 96.68% | 90.62%
4 | 8 | 8 | 97.25% | 90.28%
4 | 16 | 4 | 96.52% | 89.26%
11 | 32 | 2 | 95.60% | 88.06%
9 | 32 | 2 | 95.00% | 86.47%
4 | 32 | 2 | 91.39% | 73.68%

Turns out that the best combination is all channels of YUV color space with 15 gradient orientations, cell size of 16 pixels and 2 cells per block. This results in feature vector of length 1620 (3 channels x 9 block x 4 cells x 15 orientations).

Here is an example of how the HOG features look with these parameters on random image of a car:

![alt text][hog]


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The HOG features were extracted from the data set using functions in code cells 6 and 7, these functions were adopted from the lessons.

The data were then split into training and validation sets with ```sklearn``` function ```train_test_split```, the division was done with 80-20 ratio.

Linear SVM classifier was fitted with the training set.

Here is the learning curve that suggests that the classifier did not overfit, because the accuracy on testing set decreases and on validation set increases with bigger number of samples and eventually they converge:

![alt text][learn]

Here is the ROC curve to investigate the possible TPF/FPR tradeoffs:

![alt text][roc]

The default threshold value in ```predict``` method is 0, but this can be tweaked. In the end, I decided to keep the value at 0, raising the threshold to decrease false positive rate, also dramatically decreased the true positive rate.

The final performance of my classfier is following:

* Validation accuracy =  0.9918 %
* Precision =  0.9768 %


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used slightly modified sliding window function from the lessons (function ```detect_cars``` code cell 17).
I searched following windows:

* 2 rows of window size 64x64 px, stride (16, 16) px between rows 400 and 496
  ![alt text][w1]
* 3 rows of window size 96x96 px, stride (12, 24) px between rows 390 and 510
![alt text][w2]
![alt text][w3]
* 3 rows of window size 128x128 px, stride (32, 32) px between rows 380 and 572
![alt text][w4]
* 2 rows of window size 160x160 px, stride (20, 40) px between rows 400 and 580
![alt text][w5]
![alt text][w6]
* 1 row of window size 192x192 px, stride (48, 48) px between rows 458 and 650
![alt text][w7]

These windows were selected based on location an size of vehicles in the test images. The stride is defined by the cell site of scale*16px. In some cases I wanted finer step in vertical direction. In those cases two images were displayed, because the HOG features had to be generated twice for that sample.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The effort to weed out the false positives started with the selection process for the features, I was optimizing not only for accuracy, but also for precision of the classifier.

For really long time I was getting lots of false positive. After through investigation and process of elimination, I found out that it was caused by my choice of normalization function in ```skimage.hog()```. The default one is *L1*, but a warning pops out that this will be changed in future version to *L2-hys*. I thought, might as well change it right away. I still don't really understand why this had such big impact and will need to look more into it.

To get rid of any remaining false positives, I made use of the heatmap function and thresholding (code cell 16) that was taught in the lessons. After some trial and error, I ended up with threshold value 5. So 5 detections are needed for the pixel to be marked as containing car.

Ultimately I searched on 4 scales using YUV 3-channel HOG feature vector, which provided an acceptable result.  Here are some example images with detected patches in blue and confident detections after threshholding in green:

![alt text][test1]
![alt text][test2]
![alt text][test3]
![alt text][test4]


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I defined 

I recorded the positions of positive detections in each frame of the video using a custom class (code cell 25). I chose to store the data for 10 frames.  From the positive detections from all stored frames I created a heatmap and then thresholded that map to identify vehicle positions. The threshold remained the same at 5.

To account for the decreasing value of the older detections, the old frames are added to the heatmap with following weights:

Frame (from oldest) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10
---|---|---|---|---|---|---|---|---|---|---
Weight |0.17 | 0.20 | 0.25 | 0.30 | 0.37 | 0.45 | 0.55 | 0.67 | 0.82 | 1


I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 


Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames, detected vehicles, corresponding heatmaps and the resulting bounding boxes:

Standard situation - shows that everything works.
![alt text][diag1]

Example of flase positive detections being eliminated by the thresholding.
![alt text][diag2]

Example of no vehicle detections for last 3 frames, yet the resulting bounding box is still constructed from previous frames.
![alt text][diag3]

Advantage of fairly low threshold value is ability to distinguish between two vehicles close to each other in the image.
![alt text][diag4]

The algorithm managed to detect car in incoming traffic.
![alt text][diag5]

Unfortunately, the history of ten frames combined with the quick motion of the vehicles driving in the other direction causes these bounding boxes to linger long after the vehicle is not present.
![alt text][diag6]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This was challenging project, mainly because of many possible parameters to tune - from HOG extraction parameters and sliding window parameters to the thresholding of the heatmap. 

I was stuck for quite some time on the normalization function used by skimage.hog(). I still need to figure out, why exactly did it affect the performance so much.

All in all, I now have much greater appreciation of the neural networks that take care of much of this hyperparameter tuning and feature engineering.

As was shown, in the last section, the model is able to detect cars in oncoming lane, but the detection is really fragile. One idea would be to include Kalman filters into this model to anticipate the next detection of particular vehicle.

I am able to process video about 2 fps, it would be nice to get to better framerates.



