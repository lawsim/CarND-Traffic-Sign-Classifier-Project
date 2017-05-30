[//]: # (Image References)

[labels_bar]: ./images/labels_barchart.png "Labels Bar Chart"
[labels_pie]: ./images/labels_piechart.png "Labels Pie Chart"
[preprocess_original]: ./images/preprocess_original.png "Pre-process original image"
[preprocess_processed]: ./images/preprocess_processed.png "Pre-process processed image"
[generated_image]: ./images/generated_image.png "Generated image"
[dl_sign_1]: ./german-signs-web/1_canstock14957677.jpg "DL Sign 1"
[dl_sign_2]: ./german-signs-web/33_467030179.jpg "DL Sign 2"
[dl_sign_3]: ./german-signs-web/14_1-Iiwrp2CLW7bhOopz7QTB5w.png "DL Sign 3"

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

Here is a link to my [project code](https://github.com/lawsim/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy and pandas to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32 x 32) = 1024
* The number of unique classes/labels in the data set is  43

#### 2. Include an exploratory visualization of the dataset.

Here are visualisations showing the dataset.  They are a bar and a pie chart showing the count of each unique label in the training set.

![alt text][labels_bar]
![alt text][labels_pie]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data.

For processing the data, I first turned the image to grayscale as I believed the color data might not provide significantly different results but would increase training time and could possibly lower accuracy.  I also normalized the image per the recommendation in the notebook.

Before:
![alt text][preprocess_original]
After
![alt text][preprocess_processed]

I decided to look into generating additional data primarily because it was recommended in the assignment!  Some Google'ing later led me to an article (http://benanne.github.io/2014/04/05/galaxy-zoo.html) which described some of the techniques and reasons for data augmentation.

I explored a few different options such as creating various methods by hand, using scikit and eventually settled on using Keras' ImageDataGenerator for augmenting my data as it looked to have a variety of options available.  I elected to generate 5 additional images for each training example and experimented with different parameters until settling on:
* Rotation up to 20 degrees
* width and height shift up to 1/10
* shear angle up to 0.1 radians (~6 degrees)
* zoom up to 0.1

I did not use the ImageDataGenerator in the Keras pipeline itself as seems to be more common as I wanted to work off of the LeNet Tensorflow implementation we started with.  Instead, I appended the data to the original dataset as the program went through the batches.

Here is an example of a generated image:

![alt text][generated_image]


#### 2. Describe what your final model architecture looks like.

I started with the LeNet model defined in the previous assignment.  From there I modified some of the original parameters adding additional depth to convolution layers and adding dropout in between a couple of places in the model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Dropout				| Keep prob of 70% in train set					|
| Max pooling	      	| 2x2 stride, outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x24 					|
| Flatten		      	| To 600					 					|
| Fully connected		| Input of 600, output 120						|
| RELU					|												|
| Fully connected		| Input of 120, output 84						|
| RELU					|												|
| Dropout				| Keep prob of 70% in train set					|
| Fully connected		| Input of 84, output 43						|

#### 3. Describe how you trained your model.

For the model, I landed on a learning rate of 0.001, 15 epochs and a batch size of 128 for my parameters.  I used the same Adam Optimizer from the lab.  I first trained both the augmented and non-augmented data set to see if it performed better (it did).

I seemed to be getting diminishing returns above 10 epochs on some runs but on others it was proving helpful so I left it.  After I achieved results I was happy with I ran against the test set.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

My final model results were:
* validation set accuracy of 0.968
* test set accuracy of 0.950

I ended up building on top of the LeNet architecture from the previous lesson.  I read about a few other architectures which seem to perform better (https://medium.com/towards-data-science/neural-network-architectures-156e5bad51ba) but decided I would rather iterate on the one from the class as it might be more informative to me.

I started by just running the LeNet architecture against the data set.  With this, I couldn't achieve above around 0.88 accuracy.  I adjusted the architecture by adding dropout in a couple of points as well as adjusting the output depth in some of the convolutions.

I believe accuracy could be improved further by implementing some of the models/ideas in the previously linked article but it was functional for the project and I understood it well.

### Test a Model on New Images

#### 1. Choose at least five German traffic signs found on the web and provide them in the report.

I ended up downloading more than five images as at first I was receiving 100% accuracy and I wanted to be sure I didn't somehow stumble upon images already in the set.  In the interest of space I will only link a handful.

![alt text][dl_sign_1] ![alt text][dl_sign_2] ![alt text][dl_sign_3]

I scaled and pre-processed the signs before applying the classifier to them.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection      		| Stop sign   									| 
| No entry     			| No entry 										|
| General caution					| General caution											|
| Speed limit (30km/h) 		| Speed limit (30km/h)				 				|
| Speed limit (30km/h) 			| Speed limit (30km/h)       							|
| Wild animals crossing 			| Wild animals crossing       							|
| Turn right ahead 			| Turn right ahead       							|
| Go straight or right 			| Go straight or right       							|
| Stop 			| Stop       							|
| No entry 			| No entry       							|
| Road work 			| Dangerous curve to the right       							|
| Pedestrians 			| General caution       							|
| Speed limit (70km/h) 			| Speed limit (70km/h)       							|

I had an accuracy of 84% on this data.  It's hard to say if this is favorable or not because the set of data I used is so small.  For the ones that were wrong, perhaps the resolution is so low that it makes it difficult for the classifier to accurately predict the signs.  A larger training set, higher resolution or an improved model may all improve these results.

#### 3. Describe how certain the model is when predicting on each of the new images by looking at the softmax probabilities for each prediction.

The model was far more certain than I expected in all of the cases despite being completely wrong in two guesses.  It was 100% certain of the correct result for:
* Right-of-way at the next intersection
* No entry
* Speed limit (30km/h)
* Speed limit (30km/h)
* Go straight or right
* No entry
* Speed limit (70km/h)

For some of the others it was correct on it was very certain of the correct result. General caution, for instance:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 97.05%      			| General caution   									| 
| 2.91%     				| Traffic signals 										|
| 0.04%					| Pedestrians											|
| 0.00%	      			| Road narrows on the right					 				|
| 0.00%				    | Road work      							|


However, on the two incorrect choices it was very certain of the wrong result.

Road Work:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 90.08%      			| Dangerous curve to the right   									| 
| 6.08%      			| Road work   									| 
| 2.57%      			| Children crossing   									| 
| 1.25%      			| Slippery road   									| 
| 0.01%      			| Dangerous curve to the left   									| 

Pedestrians:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 98.88%      			| General caution   									| 
| 0.66%      			| Speed limit (70km/h)   									| 
| 0.28%      			| Traffic signals   									| 
| 0.09%      			| Road work   									| 
| 0.05%      			| Pedestrians   									| 




