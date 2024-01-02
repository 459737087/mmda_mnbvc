
	

		

			

				
Generalizing Deep Models for Overhead Image Segmentation Through Getis-Ord Gi* Pooling

				
23 Dec 2019

				

					

						

							
Xueqing
Deng
xdeng7@ucmerced.edu

							
U
C
Merced

							
Yi
Zhu

							
Shawn
Newsam
Generalizing Deep Models for Overhead Image Segmentation Through Getis-Ord Gi* Pooling

						

							
23 Dec 2019
FF1FB7B296C20D88FD636D7849E5F6D6
arXiv:1912.10667v1[cs.CV]

			

				

					
GROBID - A machine learning software for extracting information from scholarly documents

			


That most deep learning models are purely data driven is both a strength and a weakness.
Given sufficient training data, the optimal model for a particular problem can be learned.
However, this is usually not the case and so instead the model is either learned from scratch from a limited amount of training data or pre-trained on a different problem and then fine-tuned.
Both of these situations are potentially suboptimal and limit the generalizability of the model.
Inspired by this, we investigate methods to inform or guide deep learning models for geospatial image analysis to increase their performance when a limited amount of training data is available or when they are applied to scenarios other than which they were trained on.
In particular, we exploit the fact that there are certain fundamental rules as to how things are distributed on the surface of the Earth and these rules do not vary substantially between locations.
Based on this, we develop a novel feature pooling method for convolutional neural networks using Getis-Ord G * i analysis from geostatistics.
Experimental results show our proposed pooling function has significantly better generalization performance compared to a standard data-driven approach when applied to overhead image segmentation.

		

		


Introduction
Research in remote sensing has been steadily increasing since it is an important source for Earth observation.
Overhead imagery can easily be acquired using low-cost drones and no longer requires access to expensive high-resolution satellite or airborne platforms.
Since the data provides convenient and large-scale coverage, people are using it for a number of societally important problems such as traffic monitoring 
[21]
[4]
[9]
1
The content in the current sliding window is a cluster of pixels of tree.
We propose to incorporate geospatial knowledge to build a pooling function which can propagate such a spatial cluster during training, while the standard pooling is not able to achieve it.
land cover segmentation 
[17]
[36]
Recently, the analysis of overhead imagery has benefited greatly from deep learning thanks to the significant advancements made by the computer vision community on regular (non-overhead) images.
However, there still often remains challenges when adapting these deep learning techniques to overhead image analysis, such as the limited availability of labeled overhead imagery, the difficulty of the models to generalize between locations, etc.
Annotating overhead imagery is labor intensive so existing datasets are often not large enough to train effective convolutional neural networks (CNNs) from scratch.
A common practice therefore is to fine-tune an ImageNet pre-trained model on a small amount of annotated overhead imagery.
However, the generalization capability of fine-tuned models is limited as models trained on one location may not work well on others.
This is known as the cross-location generalization problem and is not necessarily limited to overhead image analysis as it can also be a challenge for ground-level imagery such as cross-city road scene segmentation 
[10]
Deep models are often overfitting due to their large capacity yet generalization is particularly important for overhead images since they can look quite different due to variations in the seasons, position of the sun, location variation, etc.
For regular image analysis, two widely adopted approaches to overcome these so-called domain gaps include domain adaptation 
[12,
13,
[32]
[33]
[34]
Both approaches have been adapted by the remote sensing community 
[2]
In this paper, we take a different, novel approach to address the domain gap problem.
We exploit the fact that things are not laid out at random on the surface of the Earth and that this structure does not vary substantially between locations.
In particular, we pose the question of how prior knowledge of this structure or, more interestingly, how the fundamental rules of geography might be incorporated into general CNN frameworks.
Inspired by work on physicsguided neural networks 
[15]
We term this geo-constrained pooling strategy Getis-Ord G * i pooling and show that it significantly improves the semantic segmentation of overhead imagery particularly in cross-location scenarios.
To our knowledge, ours is the first work to incorporate geo-spatial knowledge directly into the fundamental mechanisms of CNNs.
A brief overview of our motivation is shown in Figure 
1
Our contributions are summarized as follows:
(1) We propose Getis-Ord G * i pooling, a novel pooling method based on spatial Getis-Ord G * i analysis of CNN feature maps.
Getis-Ord G * i pooling is shown to significantly improve model generalization for overhead image segmentation.
(2) We establish more generally that using geospatial knowledge in the design of CNNs can improve the generalizability of models which provides the simulated process of the data.
Related Work
Semantic segmentation Fully connected neural networks (FCN) were recently proposed to improve the semantic segmentation of non-overhead imagery 
[20]
Various techniques have been proposed to boost their performance, such as atrous convolution 
[6]
[7]
[8]
39]
[26]
[3]
And recently, video is used to scale up training sets by synthesizing new training samples which is able to improve the accuracy of semantic segmentation networks 
[41]
Remote sensing research has been driven largely by adapting advances in regular image analysis to overhead imagery.
In particular, deep learning approaches to overhead image analysis have become a standard practice for a variety of tasks, such as land use/land cover classification 
[17]
[36]
[23]
[9]
More literature can be found in a recent survey 
[40]
And various segmentation networks have been proposed, such relation-augmentation networks 
[24]
[19]
However, these methods only adapt deep learning techniques and networks from regular to overhead images-they do not incorporate geographic structure or knowledge.
Knowledge guided neural networks Analyzing overhead imagery is not just a computer vision problem since principles of the physical world such as geo-spatial relationships can help.
For example, knowing the road map of a city can definitely improve tasks like building extraction or land cover segmentation.
While there are no works directly related to ours, there have been some initial attempts to incorporate geographic knowledge into deep learning 
[5,
38]
Chen et al. 
[5]
They also apply area-based rules during a post-processing step.
Zhang et al. 
[38]
However, these methods simply fuse prior knowledge from other sources.
Our proposed method is novel in that we incorporate geospatial rules into the CNN mechanics.
We show later how this helps regularize the model learning and leads to better generalization.
Pooling functions There are various studies in pooling for image classification as well as segmentation.
L p norm is proposed to extend max pooling where intermediate pooling functions are manually selected between max and average pooling to better fit the distribution of the input data.
[18]
Detail-Preserving Pooling (DPP) 
[27]
Salient pixels are more importance in order to achieve higher visual satisfaction.
Stride convolution is used toreplace all max pooling layers and activation functions in a small classification model that is trained from scratch and achieve better performance 
[30]
However, stride convolutions are common in segmentation tasks.
For example, the DeepLab series of networks 
[7,
8]
To enhance detail preservation in segmentation, a recent polynomial pooling approach is proposed in 
[35]
However, all these pooling methods are based on non-spatial statistics.
We instead incorporate geo-spatial rules/simulation to perform the downsampling.
Methods
In this section, we investigate how geo-spatial knowledge can be incorporated into standard deep CNNs.
We discuss some general rules from geography to describe geo- spatial patterns on the Earth.
Then we propose using Getis-Ord G * i analysis, a common technique for geo-spatial clustering, to encapsulate these rules.
This then informs our pooling function which is very general and can be used in many network architectures.
Getis-Ord G *
i pooling (G-pooling)
We take inspiration from the well-known first law of geography: everything is related to everything else, but near things are more related than distant things 
[31]
While this rule is very general and abstract, it motivates a number of quantitative frameworks that have been shown to improve geospatial data analysis.
For example, it motivates spatial autocorrelation which is the basis for spatial prediction models like kriging.
It also motivates the notion of spatial clustering wherein similar things that are spatially nearby are more significant than isolated things.
Our proposed framework exploits this to introduce a novel feature pooling method which we term Getis-Ord G * i pooling.
Pooling is used to spatially downsample the feature maps in deep CNNs.
In contrast to standard image downsampling methods which seek to preserve the spatial envelope of pixel values, pooling selects feature values that are more significant in some sense.
The most standard pooling method is max pooling in which the maximum feature value in a window is propagated.
Other pooling methods have been proposed.
Average pooling is an obvious choice and is used in 
[14,
37]
Strided convolution 
[16]
However, max pooling remains by far the most common as it has the intuitive appeal of extracting the maximum activation and thus the most prominent features of an image.
However, we postulate that isolated high feature values might not be the most informative and instead develop a method to propagate clustered values.
Specifically, we use a technique from geostatistics termed hotspot analysis to identify clusters of large values and then propagate a representative from these clusters.
Hotspot analysis uses the Getis-Ord G * i 
[25]
These locations are the so-called hotspots.
The Getis-Ord G * i statistic is computed by comparing the local sum of a feature and its neighbors proportionally to the sum of all features in a spatial region.
When the local sum is different from the expected local sum, and when that difference is too large to be the result of random noise, it will lead to a high positive or low negative G * i value that is statistically significant.
We focus on locations with high positive G * i values since we want to propagate activations.
Definition
We now describe our G-pooling algorithm in detail.
Please see Figure 
2
Similar to other pooling methods, we use a stride sliding window to downsample the input.
Given a feature map within the stride window, in order to compute its G * i , we first need to define the weight matrix based on the spatial locations.
We denote the feature values within the sliding window as X = x 1 , x 2 , ..., x n where n is the number of pixels (locations) within the sliding window.
We assume the window is rectangular and compute the G * i statistic at the center of the window.
Let the feature value at the center be x i .
(If the center does not fall on a pixel location then we compute x i as the average of the adjacent values.)
The G * i statistic uses weighed averages where the weights are based on spatial distances.
Let p x (x j ) and p y (x j ) denote the x and y positions of feature value x j in the image plane.
A weight matrix w that measures the Euclidean distance on the image plane between x i and the other locations within the sliding window is then computed as
w i,j = (p x (x i ) − p x (x j )) 2 + (p y (x i ) − p y (x j )) 2 . (1)
The Getis-Ord G * i value at location i is now computed as
G * i = n j=1 w i,j x j − X n j=1 w i,j S [n n j=1 w 2 i,j −( n j=1 wi,j ) 2 ] n−1 .
(2) where X and S are as below,
X = n j=1 x j n ,
(3)
S = n j=1 x 2 j n − ( X) 2 . (
4
)
Spatial clusters can be detected based on the G * i value.
The higher the value, the more significant the cluster is.
However, the G * i value just indicates whether there is a spatial cluster or not.
To achieve our goal of pooling, we need to summarize the local region of the feature map by extracting a representative value.
We use a threshold to do this.
If the computed G * i is greater than or equal to the threshold, a spatial cluster is detected and the value x i is used for pooling, otherwise the maximum in the window is used.
G − pooling(x) = x i if G * i ≥ threshold max(x) if G * i < threshold
(5)
It's noted that G * i is in range [-2.8,2.8]
where a negative value indicates a coldspot which means a spatial scatter and a positive value indicates a hotspot which means a spatial cluster.
The absolute value |G * i | indicates the significance.
For example, a high positive G * i value indicates the feature is more likely to be a spatial cluster.
The output feature map produced by G-pooling is Gpooling(X) which results after sliding the window over the entire input feature map.
The threshold is set to 3 different values in this work, 1.0, 1.5, 2.0.
A higher threshold means the current feature map has less chance to be reported as a spatial cluster and so max pooling will be applied instead.
A lower threshold causes more spatial clusters to be detected and max pooling will be applied less often.
As the threshold ranges from 1.0 to 1.5 to 2.0, fewer spatial clusters/hotspots will be detected.
We find that a threshold of 2.0 results in few hostpots being detected and max pooling mostly to be used.
Network Architecture
A pretrained VGG network 
[29]
VGG has been widely used as a backbone in various semantic segmentation networks such as FCN 
[20]
[26]
[3]
In VGG, the standard max pooling is a 2×2 window size with a stride of 1.
Our proposed Gpooling uses a 4×4 window size with a stride of 4. Therefore, after applying the standard pooling, the size of feature map drops to 1/2, while with our G-pooling it drops to 1/4.
A small window size is not used in our proposed G-pooling since Getis-Ord G * i analysis may not work well in such a small region.
However, we tested the scenario where standard pooling is performed with a 4 × 4 sliding window and the performance is only slightly different from that using the standard 2 × 2 window.
In general, segmentation networks using VGG16 as the backbone have 5 max pooling layers.
So, when we replace max pooling with our proposed G-pooling, there will be two G-pooling and one max pooling layers.
Experiments
Dataset
ISPRS dataset We evaluate our method on two image datasets from the ISPRS 2D Semantic Labeling Challenge 
[1]
These datasets are comprised of very high resolution aerial images over two cities in Germany: Vaihingen and Potsdam.
While Vaihingen is a relatively small village with many detached buildings and small multi-story buildings, Potsdam is a typical historic city with large building blocks, narrow streets and dense settlement structure.
The goal is to perform semantic labeling of the images using six common land cover classes: buildings, impervious surfaces (e.g.
roads), low vegetation, trees, cars and clutter/background.
We report test metrics obtained on the held-out test images.
Vaihingen The Vaihingen dataset has a resolution of 9 cm/pixel with tiles of approximately 2100 × 2100 pixels.
There are 33 images, from which 16 have a public ground truth.
Even though the tiles consist of Infrared-Red-Green (IRRG) images and DSM data extracted from the Lidar point clouds, we use only the IRRG images in our work.
We select five images for validation (IDs: 11, 15, 28, 30 and 34) and the remaining 11 for training, following 
[22,
28]
Potsdam
The Potsdam dataset has a resolution of 5 cm/pixel with tiles of 6000 × 6000 pixels.
There are 38 images, from which 24 have public ground truth.
Similar to Vaihingen, we only use the IRRG images.
We select seven images for validation (IDs: 2 11, 2 12, 4 10, 5 11, 6 7, 7 8 and 7 10) and the remaining 17 for training, again following 
[22,
28]
Experimental settings
Baselines Here, we compare our proposed G-pooling with the standard max-pooling, average-pooling, stride convolution, and the recently proposed P-pooling 
[35]
Max/average pooling is commonly for downsampling in the semantic segmentation networks that have VGG as a backbone.
ResNet 
[11]
Such a network architecture has been adopted by recent studies for semantic segmentation, in particular the DeepLab series 
[6]
[7]
[8]
[39]
Max pooling is removed and instead strided convolution is used to downsample the feature maps while dilated convolution is used to enlarge the receptive fields.
There is also work on detail preserving pooling, for example DDP 
[27]
[35]
We select the most recent one, P-pooling, which outperforms the other detail preserving methods for comparison.
Evaluation Metrics
We have two goals in this work, the model's segmentation accuracy and its generalization performance.
Model accuracy is used to report the performance on the test/validation set using the model trained with training set within one dataset.
Model generalizability is used to report the performance of the test/validation set with another dataset.
In general, the domain gap between train and test/validation set from one dataset is relatively small.
However, cross-dataset testing exists large domain shift problem.
Model accuracy
The commonly used per class intersection over union (IoU) and mean IoU (mIoU) as well as the pixel accuracy are adopted for evaluating segmentation accuracy.
Model generalizability Specifically, we will perform evaluation on the ISPRS Potsdam set with a model trained on the ISPRS Vahingen set (Potsdam→Vaihingen) and reverse the order (Vaihingen→Potsdam).
Pixel accuracy and mIoU are used to report the performance of the generalizability.
Implementation Details
Implementation of G-pooling Models are implemented using the PyTorch framework.
Max-pooling, averagepooling, stride conv are provided as built-in function and P-pooling has open-source code.
We implement our Gpooling in C and use the interface to connect to PyTorch for network training.
We adopt the network architecture of FCN 
[20]
[29]
The details of the FCN using our G-pooling can be found in Section 3.3.
The results in Table 
1
Training settings Since the image tiles are too large to be fed through a deep CNN due to limited GPU memory, we randomly extract image patches of size of 256×256 pixels as the training set.
Following standard practice, we only use horizontal and vertical flipping as data augmentation during training.
For testing, the whole image is split into 256×256 patches with a stride of 256.
Then, the predictions of all patches are concatenated for evaluation.
We train all our models using Stochastic Gradient Descent (SGD) with an initial learning rate of 0.1, a momentum of 0.9, a weight decay of 0.0005 and a batch size of 5.
If the validation loss plateaus for 3 consecutive epochs, we divide the learning rate by 10.
If the validation loss plateaus for 6 consecutive epochs or the learning rate is less than 1e-8, we stop the model training.
We use a single TITAN V GPU for training and testing.
Effectiveness of G-pooling
In this section, we first show that incorporating geospatial knowledge into a pooling function of the standard CNN learning can improve segmentation accuracy.
Then we demonstrate the promising generalization capability of our proposed G-pooling.
The segmentation accuracy on FCN using various pooling functions reported on the test set is shown in Table 
1
For G-pooling, we experiment on 3 different thresholds, which is 1.0, 1.5 and 2.0.
The range of G * i value is [-2.8, 2.8].
As explained in Section 3.2, higher G * i value can cause more uses of max pooling.
If we set the G * i value as 2.8, then the case will be all max pooling.
Qualitative results are shown in Figure 
4
And the quantitative results for eval-uating model accuracy and cross-location generalization is shown in Table 
1 and 2
respectively.
Non-spatial vs geospatial statistics The baselines of pooling functions are usually non-spatial statistics, for example, finding the max/average value.
Our approach provides a geospatial process to simulate how things are related based on spatial location.
Here, we pose the question, "is the knowledge useful to train a deep CNN?".
As we mentioned in Section 3, such a knowledge incorporated method can bring the benefit of improved generalizability.
As shown in Table 
1
Our G-pooling-1.0 and 2.0 is not able to outperform some baselines in the model accuracy testing, which indicates the threshold selection is important.
Some classes of the baselines have higher performance compared to ours.
This is expected since the dataset is relatively small and may be overfitting.
The qualitative results in Figure 
4
In particular, there is less noise inside the objects compared to the other methods.
This demonstrates our proposed G-pooling simulates the geospatial distributions and makes the prediction within the objects more compact.
The effects of threshold is shown in Table 
3
2
We note that the UDA method AdaptSegNet 
[32]
The other methods don't benefit from the unlabeled data.
As shown in Table 
2
For Potsdam→Vaihingen, G-pooling outperforms P-pooling by more than 2%.
For Vaihingen→Potsdam, the improvement is even more significant, at least 3.41%.
When we compare the knowledge incorporation method G-pooling with the domain adaptation method AdaptSegNet, the performance difference is just 0.61% for Potsdam.
The results verify our assumption that incorporating knowledge helps generalizations as well.
And the performance is close to that of domain adaptation which utilizes a great amount of unlabeled data to learn the data distribution.
Even though knowledge incorporation doesn't outperform data-based domain adaptation, these two methods can be combined to provide even better generalization.
Domain adaptation vs knowledge incorporation Table
G-pooling and state-of-the-art methods
In order to verify that our proposed G-pooling is able to improve state-of-the-art segmentation approaches, we select DeepLab 
[6]
[3]
As mentioned above, the models in Section 5 use FCN as the network architecture and VGG-16 as the backbone.
For fair comparison with FCN, VGG-16 is also used as the backbone in DeepLab and Seg-Net.
DeepLab 
[6]
For the baseline DeepLab itself, pool4 and pool5 from the backbone VGG-16 are removed and followed by 
[32]
For the G-pooling version, pool1,pool2 are replaced with G-pooling and we keep pool3.
Thus there are three max pooling layers in the baseline and one G-pooling layer and one max pooling layer in our proposed version.
SegNet uses an encoder-decoder architecture and preserves the max pooling index for unpooling in the decoder.
Similar to Deeplab, there are 5 max pooling layers in total in the encoder of SegNet so pool1,pool2 are replaced with the proposed G pool1 and pool3,pool4 are replaced with G pool2, and pool5 is kept.
This leads us to use a 4 × 4 unpooling window to recover the spatial resolution where the original ones are just 2 × 2. Thus there are two G-pooling and one max pooling layers in our SegNet version.
As can be seen in Table 
4
And the improvement on the generalization test Potsdam→Vaihingen is even more obvious, G-pooling improves mIoU from 38.57 to 40.04.
Similar observations can be made for SegNet and FCN.
For Vaihingen, even though the model accuracy is not as high as the baseline, the difference is small.
The mIoU of our versions of DeepLab, SegNet and FCN is less than 1% lower.
We note that Vaihingen is an easier dataset than Potsdam, since it only includes urban scenes while Potsdam includes both urban and nonurban.
However, the generalizability of our model using G-pooling is much better.
As shown, when testing Potsdam using a model trained on Vaihingen, FCN with G-pooling is able to achieve 23.02% mIoU which is an improvement of 7.54% IoU.
The same observations can be made for DeepLab and SegNet.
Discussion
Incorporating knowledge is not a novel approach for neural networks.
Before deep learning, there was work on rule-based neural networks which required expert knowledge to design the network for specific applications.
Due to the large capacity of deep models, deep learning has become the primary approach to address vision problems.
However, deep learning is a data-driven approach which relies significantly on the amount of training data.
If the model is trained with a large amount of data then it will have good generalization.
But the case is often, particularly in overhead image segmentation, that the dataset is not large enough like it is in ImageNet/Cityscapes.
This causes overfitting.
Early stopping, cross-validation, etc. can help to avoid overfitting.
Still, if domain shift exists between the training and test sets, the deep models do not perform well.
In this work, we propose a knowledge-incorporated approach to reduce overfitting.
We address the question of how to incorporate the knowledge directly into the deep models by proposing a novel pooling method for overhead image segmentation.
But some issues still need discussing as follows.
Scenarios using G-pooling As mentioned in section 3, Gpooling is developed using Getis-Ord G * i analysis which quantifies how the spatial convergence occurs.
This is a simulated process design for geospatial data downsampling.
Thus it's not necessarily appropriate for other image datasets.
This is more general restriction of incorporating of knowledge.
The Getis-Ord G * i provides a method to identify spatial clusters while training.
The effect is similar to conditional random fields/Markov random fields in standard computer vision post-processing methods.
However, it is different from them since the spatial clustering is dynamically changing based on the feature maps and the geospatial location while post-processing methods rely on the prediction of the models.
Local geospatial pattern
We now explain how G-pooling works in deep neural networks.
Getis-Ord G * i analysis is usually used to analyze a global region hotspot detection which describes the geospatial convergence.
As shown in Figure 
3
The spatial size of the G-pooling will be 64 × 64 and 16 × 16 respectively.
And the max-pooling will lead to the size of feature map being reduced by 1/2 while ours it will be by 1/4.
This is because we want to compute G * i over a larger region.
Even though G * i is usually computed over a larger region than in our framework, it still provides captures spatial convergence within a small region.
Also, two G-pooling operations are applied at different scales of feature map and so a larger region in the input image is really considered.
Specifically, the first 4 × 4 pooling window is slid over the 256 × 256 feature map and the output feature map has size 64 × 64.
This is fed through the next conv layers and a second G-pooling is applied.
At this stage, the input feature map is 64 × 64 and so when a 4 × 4 sliding window is now used, a region of 16 × 16 is really considered, which is 1/16 of the whole image.
Limitations There are some limitations of our work.
For example, we didn't investigate the optimal window size for performing Getis-Ord G * i analysis.
We also only consider one kind of spatial pattern, clusters.
And, there might be better places than pooling to incorporate knowledge in CNN architectures.
Conclusion
In this paper, we investigate how geospatial knowledge can be incorporated into deep learning for geospatial image analysis.
We demonstrate that incorporating geospatial rules improves performance.
We realize, though, that ours is just preliminary work into geospatial guided deep learning.
We note the limitations of our approach, for example, that the prior distribution does not provide benefits for classes in which this prior knowledge is not relevant.
Our proposed approach does not show much improvement on the single dataset case especially a small dataset.
ISPRS Vaihingen is a very small dataset which contains around only 500 images of size of 256 × 256.
In the future, we will explore other ways to encode geographic rules so they can be incorporated into deep learning models.
Figure 2 :
2
Figure 2: Given a feature map as an input, max pooling (top right) and the proposed G-pooling (bottom right) create different output downsampled feature map based on the characteristics of spatial cluster.
The feature map within the sliding window (blue dot line) indicates a spatial cluster.
Max pooling takes the max value ignoring the spatial cluster, while our G-pooling takes the interpolated value at the center location.
(White, gray and black represent three values range from low to high.)
Figure 3 :
3
Figure 3: A FCN network architecture with G-pooling.
Figure 4 :
4
Figure 4: Qualitative results of ISPRS Potsdam.
White: road, blue: building, cyan: low vegetation, green: trees, yellow: cars, red: clutter.
Table 1 :
1
Experimental results of FCN using VGG-16 as backbone.
Stride conv, P-pooling and ours G-pooling are used to replaced the standard max/average pooling.
Potsdam
Methods
Roads Buildings Low Veg. Trees Cars mIoU Pixel Acc.
Max
70.62
74.28
65.94
61.36 61.40 66.72
79.55
Average
69.34
74.49
63.94
60.06 60.28 65.62
78.08
Stride
67.22
73.97
63.01
60.09 59.39 64.74
77.54
P-pooling
71.97
75.55
66.80
62.03 62.39 67.75
81.02
G-pooling-1.0 (ours) 68.59
77.39
67.48
55.56 62.18 66.24
79.43
G-pooling-1.5 (ours) 70.06
76.12
67.67
62.12 63.91 67.98
81.63
G-pooling-2.0 (ours) 70.99
74.89
65.34
61.57 60.77 66.71
79.46
Vaihingen
Max
70.63
80.42
51.57
70.12 55.32 65.61
81.88
Average
70.54
79.86
50.49
69.18 54.83 64.98
79.98
Strde conv
68.36
77.65
49.21
67.34 53.29 63.17
79.44
P-pooling
71.06
80.52
51.70
70.93 53.65 65.57
82.44
G-pooling-1.0 (ours) 72.15
79.69
53.28
70.89 53.72 65.95
81.78
G-pooling-1.5 (ours) 71.61
78.74
48.18
68.53 55.64 64.54
80.42
G-pooling-2.0 (ours) 71.09
78.88
50.62
68.32 54.01 64.58
80.75
Table 2 :
2
Cross-location evaluation.
We compare the generalization capability of using G-pooling with domain adaptation method AdaptSegNet which utilize the unlabeled data.
Potsdam → Vaihingen
Roads Buildings Low Veg. Trees Cars mIoU Pixel Acc.
Max-pooling
28.75
51.10
13.48
56.00 25.99 35.06
47.48
stride conv
28.66
50.98
12.76
55.02 24.81 34.45
46.51
P-pooling
32.87
50.43
13.04
55.41 25.60 35.47
48.94
Ours (G-pooling) 37.27
54.53
14.85
54.24 27.35 37.65
55.20
AdaptSegNet
41.54
40.74
21.68
50.45 36.87 38.26
57.73
Vaihingen → Potsdam
Max-pooling
20.36
24.51
19.19
9.71
3.65
15.48
45.32
stride conv
20.65
23.22
16.57
8.73
8.32
15.50
42.28
P-pooling
23.97
27.66
14.03
10.30 12.07 19.61
44.98
Ours (G-pooling) 27.05
29.34
33.57
9.12 16.01 23.02
45.54
AdaptSegNet
40.28
37.97
46.11
15.87 20.16 32.08
50.28
Table 3 :
3
The average percentage of detected spatial clusters per feature map with different threshold.
Threshold
1.0
1.5
2.0
Potsdam
15.87 9.85 7.65
Vaihingen 14.99 10.44 7.91
Table 4 :
4
Experimental results on comparing w/o and w/ proposed G-pooling for the state-of-the-art segmentation networks.
P→V indicates the model trained on Potsdam and test on Vaihingen, and versa the verses.
Potsdam (P)
P→V
Network G-Pooling mIoU
PA
mIoU
PA
×
67.97 81.25 38.57 58.47
DeepLab
68.33 80.67 40.04 63.21
×
69.47 82.53 35.98 53.69
SegNet
70.17 83.27 39.04 56.42
×
66.72 79.55 35.06 47.48
FCN
67.98 81.63 37.65 55.20
Vaihingen (V)
V→P
×
70.80 83.74 18.44 33.96
DeepLab
70.11 83.09 19.26 36.17
×
66.04 81.79 16.77 45.90
SegNet
66.71 82.66 25.64 48.08
×
65.61 81.88 15.48 45.32
FCN
65.95 81.87 23.02 45.54

			


				




	

		
ISPRS 2D Semantic Labeling Challenge

	

		
Beyond RGB: Very High Resolution Urban Remote Sensing with Multimodal Deep Networks

			
N
Audebert

			
B
Saux

			
S
Lefvre

		
ISPRS Journal of Photogrammetry and Remote Sensing

			
2
2018

	

		
SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation

			
V
Badrinarayanan

			
A
Kendall

			
R
Cipolla

		
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)

			
2
4
7
2017

	

		
Comprehensive Survey of Deep Learning in Remote Sensing: Theories, Tools, and Challenges for the Community

			
J
E
Ball

			
D
T
Anderson

			
C
S
Chan

		
Journal of Applied Remote Sensing

			
1
2017

	

		
Knowledge-guided Golf Course Detection using a Convolutional Neural Network Fine-tuned on Temporally Augmented Data

			
J
Chen

			
C
Wang

			
A
Yue

			
J
Chen

			
D
He

			
X
Zhang

		
J. Appl. Remote Sens

			
2
2017

	

		
DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs

			
L
Chen

			
G
Papandreou

			
I
Kokkinos

			
K
Murphy

			
A
Yuille

		
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)

			
2
5
7
2017

	

		
Rethinking Atrous Convolution for Semantic Image Segmentation

			
L
Chen

			
G
Papandreou

			
F
Schroff

			
H
Adam
arXiv:1706.05587

			
2017
2
5
arXiv preprint

	

		
Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

			
L
Chen

			
Y
Zhu

			
G
Papandreou

			
F
Schroff

			
H
Adam

		
European conference on computer vision (ECCV)

			
2018
2
5

	

		
Vehicle Detection in Satellite Images by Hybrid Deep Convolutional Neural Networks

			
X
Chen

			
S
Xiang

			
C
Liu

			
C
Pan

		
IEEE Geoscience and Remote Sensing Letters

			
1
2
2014

	

		
No More Discrimination: Cross City Adaptation of Road Scene Segmenters

			
Y
Chen

			
W
Chen

			
Y
Chen

			
B
Tsai

			
Y
Wang

			
M
Sun

		
International Conference on Computer Vision (ICCV)

			
2017
2

	

		
CyCADA: Cycle-Consistent Adversarial Domain Adaptation

			
K
He

			
X
Zhang

			
S
Ren

			
J
Sun

			
J
Hoffman

			
E
Tzeng

			
T
Park

			
J
Zhu

			
P
Isola

			
K
Saenko

			
A
Efros

			
T
Darrell

		
Proceedings of the IEEE conference on computer vision and pattern recognition
the IEEE conference on computer vision and pattern recognition

			
2016. 2018
2
International Conference on Machine Learning

	

		
FCNs in the Wild: Pixel-Level Adversarial and Constraint-based Adaptation

			
J
Hoffman

			
D
Wang

			
F
Yu

			
T
Darrell
arXiv:1612.02649
2016. 2
arXiv preprint

	

		
Densely Connected Convolutional networks

			
G
Huang

			
Z
Liu

			
L
Van Der Maaten

			
K
Weinberger

		
IEEE Conference on Computer Vision and Pattern Recognition (CVPR)

			
2017
3

	

		

			
A
Karpatne

			
W
Watkins

			
J
Read

			
V
Kumar
arXiv:1710.11431
2017. 2
Physics-Guided Neural Networks (PGNN): An Application in Lake Temperature Modeling
arXiv preprint

	

		
DelugeNets: Deep Networks with Efficient and Flexible Cross-Layer Information Inflows

			
J
Kuen

			
X
Kong

			
G
Wang

			
Y
Tan

		
International Conference on Computer Vision (ICCV)

			
2017
3

	

		
Deep Learning Classification of Land Cover and Crop Types Using Remote Sensing Data

			
N
Kussul

			
M
Lavreniuk

			
S
Skakun

			
A
Shelestov

		
IEEE Geoscience and Remote Sensing Letters

			
1
2
2017

	

		
Generalizing pooling functions in convolutional neural networks: Mixed, gated, and tree

			
C
Lee

			
P
Gallagher

			
Z
Tu

		
Artificial intelligence and statistics

			
2016
2

	

		
Semantic Labeling in Very High Resolution Images via a Self-Cascaded Convolutional Neural Network

			
Y
Liu

			
B
Fan

			
L
Wang

			
J
Bai

			
S
Xiang

			
C
Pan

		
ISPRS Journal of Photogrammetry and Remote Sensing

			
2
2018

	

		
Fully Convolutional Networks for Semantic Segmentation

			
J
Long

			
E
Shelhamer

			
T
Darrell

		
IEEE Conference on Computer Vision and Pattern Recognition (CVPR)

			
2015
5

	

		
Learning Traffic as Images: A Deep Convolutional Neural Network for Large-Scale Transportation Network Speed Prediction

			
X
Ma

			
Z
Dai

			
Z
He

			
J
Ma

			
Y
Wang

			
Y
Wang

		
Sensors

			
1
2017

	

		
High-Resolution Aerial Image Labeling with Convolutional Neural Networks

			
E
Maggiori

			
Y
Tarabalka

			
G
Charpiat

			
P
Alliez

		
IEEE Transactions on Geoscience and Remote Sensing

			
4
2017

	

		
Learning to Detect Roads in High-Resolution Aerial Images

			
V
Mnih

			
G
E
Hinton

		
European Conference on Computer Vision (ECCV)

			
2010
2

	

		
A Relation-Augmented Fully Convolutional Network for Semantic Segmentation in Aerial Scenes

			
L
Mou

			
Y
Hua

			
X
X
Zhu

		
IEEE Conference on Computer Vision and Pattern Recognition (CVPR)

			
2019
2

	

		
Local Spatial Autocorrelation Statistics: Distributional Issues and an Application

			
J
K
Ord

			
Arthur
Getis

		
Geographical Analysis

			
3
1995

	

		
U-net: Convolutional networks for biomedical image segmentation

			
O
Ronneberger

			
P
Fischer

			
T
Brox

		
International Conference on Medical image computing and computer-assisted intervention

			
2015
2
4

	

		
Detailpreserving pooling in deep networks

			
F
Saeedan

			
N
Weber

			
M
Goesele

			
S
Roth

		
IEEE Conference on Computer Vision and Pattern Recognition (CVPR)

			
2018
5

	

		
Fully Convolutional Networks for Dense Semantic Labelling of High-Resolution Aerial Imagery

			
Jamie
Sherrah
arXiv:1606.02585
2016. 4
arXiv preprint

	

		
Very deep convolutional networks for large-scale image recognition

			
K
Simonyan

			
A
Zisserman
arXiv:1409.1556

			
2014
4
5
arXiv preprint

	

		
Striving for simplicity: The all convolutional net

			
J
Springenberg

			
A
Dosovitskiy

			
T
Brox

			
M
Riedmiller

		
International Conference on Learning Representation workshop (ICLR workshop)

			
2015
2

	

		
A Computer Movie Simulating Urban Growth in the Detroit Region

			
W
R
Tobler

		
Economic Geography

			
3
1970

	

		
Learning to Adapt Structured Output Space for Semantic Segmentation

			
Y
Tsai

			
W
Hung

			
S
Schulter

			
K
Sohn

			
M
Yang

			
M
Chandraker

		
IEEE Conference on Computer Vision and Pattern Recognition (CVPR)

			
2018
7

	

		
Domain Adaptation for Structured Output via Discriminative Representations

			
Y
Tsai

			
K
Sohn

			
S
Schulter

			
M
Chandraker

		
International conference on Computer Vision (ICCV)

			
2019
2

	

		
Adversarial Discriminative Domain Adaptation

			
E
Tzeng

			
J
Hoffman

			
K
Saenko

			
T
Darrell

		
IEEE Conference on Computer Vision and Pattern Recognition (CVPR)

			
2017
2

	

		
Building detail-sensitive semantic segmentation networks with polynomial pooling

			
Z
Wei

			
J
Zhang

			
L
Liu

			
F
Zhu

			
F
Shen

			
Y
Zhou

			
S
Liu

			
Y
Sun

			
L
Shao

		
IEEE Conference on Computer Vision and Pattern Recognition (CVPR)

			
2019
2
5

	

		
Learning Building Extraction in Aerial Scenes with Convolutional Networks

			
J
Yuan

		
IEEE Transactions on Pattern Analysis and Machine Intelligence

			
1
2
2017
TPAMI)

	

		
Wide Residual Networks

			
S
Zagoruyko

			
N
Komodakis
arXiv:1605.07146
2016. 3
arXiv preprint

	

		
Airport Detection from Remote Sensing Images using Transferable Convolutional Neural Networks

			
P
Zhang

			
X
Niu

			
Y
Dou

			
F
Xia

		
International Joint Conference on Neural Networks (IJCNN)

			
2016
2

	

		
Pyramid Scene Parsing Network

			
H
Zhao

			
J
Shi

			
X
Qi

			
X
Wang

			
J
Jia

		
IEEE Conference on Computer Vision and Pattern Recognition (CVPR)

			
2017
5

	

		
Deep Learning in Remote Sensing: A Comprehensive Review and List of Resources

			
X
X
Zhu

			
D
Tuia

			
L
Mou

			
G
Xia

			
L
Zhang

			
F
Xu

			
F
Fraundorfer

		
IEEE Geoscience and Remote Sensing Magazine

			
2
2017

	

		
Improving semantic segmentation via video propagation and label relaxation

			
Y
Zhu

			
K
Sapra

			
F
A
Reda

			
K
J
Shih

			
S
Newsam

			
A
Tao

			
B
Catanzaro

		
The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)

			
2019
2
