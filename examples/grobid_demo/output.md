# Generalizing Deep Models for Overhead Image Segmentation Through
- Getis-Ord Gi
- Xueqing Pooling
- None Deng
- Yi Amazon
- Yuxin Tian
- Shawn Newsam

## Abstract
That most deep learning models are purely data driven is both a strength and a weakness.Given sufficient training data, the optimal model for a particular problem can be learned.However, this is usually not the case and so instead the model is either learned from scratch from a limited amount of training data or pre-trained on a different problem and then fine-tuned.Both of these situations are potentially suboptimal and limit the generalizability of the model.Inspired by this, we investigate methods to inform or guide deep learning models for geospatial image analysis to increase their performance when a limited amount of training data is available or when they are applied to scenarios other than which they were trained on.In particular, we exploit the fact that there are certain fundamental rules as to how things are distributed on the surface of the Earth and these rules do not vary substantially between locations.Based on this, we develop a novel feature pooling method for convolutional neural networks using Getis-Ord G * i analysis from geostatistics.Experimental results show our proposed pooling function has significantly better generalization performance compared to a standard data-driven approach when applied to overhead image segmentation.
## 1. Introduction
Research in remote sensing has been steadily increasing since it is an important source for Earth observation.Overhead imagery can easily be acquired using low-cost drones and no longer requires access to expensive high-resolution satellite or airborne platforms.Since the data provides convenient and large-scale coverage, people are using it for a number of societally important problems such as traffic monitoring [[21]], urban planning [[4]], vehicle detection [[9]], Figure 1<ns0:ref xmlns:ns0="http://www.tei-c.org/ns/1.0" type="figure" coords="1,337.94,401.32,3.88,8.64">1</ns0:ref>: Motivation of our work.: Motivation of our work.The content in the current sliding window is a cluster of pixels of tree.We propose to incorporate geospatial knowledge to build a pooling function which can propagate such a spatial cluster during training, while the standard pooling is not able to achieve it.land cover segmentation [[17]], building extraction [[36]], etc.
Recently, the analysis of overhead imagery has benefited greatly from deep learning thanks to the significant advancements made by the computer vision community on regular (non-overhead) images.However, there still often remains challenges when adapting these deep learning techniques to overhead image analysis, such as the limited availability of labeled overhead imagery, the difficulty of the models to generalize between locations, etc.
Annotating overhead imagery is labor intensive so existing datasets are often not large enough to train effective convolutional neural networks (CNNs) from scratch.A common practice therefore is to fine-tune an ImageNet pre-trained model on a small amount of annotated overhead imagery.However, the generalization capability of fine-tuned models is limited as models trained on one location may not work well on others.This is known as the cross-location generalization problem and is not necessarily limited to overhead image analysis as it can also be a challenge for ground-level imagery such as cross-city road scene segmentation [[10]].
					Deep models are often overfitting due to their large capacity yet generalization is particularly important for overhead images since they can look quite different due to variations in the seasons, position of the sun, location variation, etc.For regular image analysis, two widely adopted approaches to overcome these so-called domain gaps include domain adaptation [[12,]
						[13,]
						[[32]]
						[[33]]
						[[34]] and data fusion.Both approaches have been adapted by the remote sensing community [[2]] to improve performance and robustness.
In this paper, we take a different, novel approach to address the domain gap problem.We exploit the fact that things are not laid out at random on the surface of the Earth and that this structure does not vary substantially between locations.In particular, we pose the question of how prior knowledge of this structure or, more interestingly, how the fundamental rules of geography might be incorporated into general CNN frameworks.Inspired by work on physicsguided neural networks [[15]], we develop a framework in which spatial hotspot analysis informs the feature map pooling.We term this geo-constrained pooling strategy Getis-Ord G * i pooling and show that it significantly improves the semantic segmentation of overhead imagery particularly in cross-location scenarios.To our knowledge, ours is the first work to incorporate geo-spatial knowledge directly into the fundamental mechanisms of CNNs.A brief overview of our motivation is shown in Figure 1<ns0:ref xmlns:ns0="http://www.tei-c.org/ns/1.0" type="figure" coords="2,171.42,398.32,3.74,8.64">1</ns0:ref>.
					.
					
Our contributions are summarized as follows:
(1) We propose Getis-Ord G * i pooling, a novel pooling method based on spatial Getis-Ord G * i analysis of CNN feature maps.Getis-Ord G * i pooling is shown to significantly improve model generalization for overhead image segmentation.
(2) We establish more generally that using geospatial knowledge in the design of CNNs can improve the generalizability of models which provides the simulated process of the data.

## 2. Related Work
Semantic segmentation Fully connected neural networks (FCN) were recently proposed to improve the semantic segmentation of non-overhead imagery [[20]].
					Various techniques have been proposed to boost their performance, such as atrous convolution [[6]]
						[[7]]
						[[8]]
						[39]], skip connections [[26]], and preserving max pooling index for unpooling [[3]].
					And recently, video is used to scale up training sets by synthesizing new training samples which is able to improve the accuracy of semantic segmentation networks [[41]].
					Remote sensing research has been driven largely by adapting advances in regular image analysis to overhead imagery.In particular, deep learning approaches to overhead image analysis have become a standard practice for a variety of tasks, such as land use/land cover classification [[17]], building extraction [[36]], road segmentation [[23]], car detection [[9]], etc.More literature can be found in a recent survey [[40]].
					And various segmentation networks have been proposed, such relation-augmentation networks [[24]] and casnet [[19]].
					However, these methods only adapt deep learning techniques and networks from regular to overhead images-they do not incorporate geographic structure or knowledge.Knowledge guided neural networks Analyzing overhead imagery is not just a computer vision problem since principles of the physical world such as geo-spatial relationships can help.For example, knowing the road map of a city can definitely improve tasks like building extraction or land cover segmentation.While there are no works directly related to ours, there have been some initial attempts to incorporate geographic knowledge into deep learning [[5,]
						[38]].
					Chen et al. [[5]] develop a knowledge-guided golf course detection approach using a CNN fine-tuned on temporally augmented data.They also apply area-based rules during a post-processing step.
						[Zhang et al. [38]] propose searching for adjacent parallel line segments as prior spatial information for the fast detection of runways.However, these methods simply fuse prior knowledge from other sources.Our proposed method is novel in that we incorporate geospatial rules into the CNN mechanics.We show later how this helps regularize the model learning and leads to better generalization.Pooling functions There are various studies in pooling for image classification as well as segmentation.L p norm is proposed to extend max pooling where intermediate pooling functions are manually selected between max and average pooling to better fit the distribution of the input data.
						[[18]] generalizes pooling methods by using a learned linear combination of max and average pooling.Detail-Preserving Pooling (DPP) [[27]] learns weighted summations of pixels over different pooling regions.Salient pixels are more importance in order to achieve higher visual satisfaction.Stride convolution is used toreplace all max pooling layers and activation functions in a small classification model that is trained from scratch and achieve better performance [[30]].
					However, stride convolutions are common in segmentation tasks.For example, the DeepLab series of networks [[7,]
						[8]] use stride convolutional layers for feature down-sampling rather than max pooling.To enhance detail preservation in segmentation, a recent polynomial pooling approach is proposed in [[35]].
					However, all these pooling methods are based on non-spatial statistics.We instead incorporate geo-spatial rules/simulation to perform the downsampling.

## 3. Methods
In this section, we investigate how geo-spatial knowledge can be incorporated into standard deep CNNs.We discuss some general rules from geography to describe geo- spatial patterns on the Earth.Then we propose using Getis-Ord G * i analysis, a common technique for geo-spatial clustering, to encapsulate these rules.This then informs our pooling function which is very general and can be used in many network architectures.

## 3.1. Getis-Ord G *
i pooling (G-pooling)
We take inspiration from the well-known first law of geography: everything is related to everything else, but near things are more related than distant things [[31]].
					While this rule is very general and abstract, it motivates a number of quantitative frameworks that have been shown to improve geospatial data analysis.For example, it motivates spatial autocorrelation which is the basis for spatial prediction models like kriging.It also motivates the notion of spatial clustering wherein similar things that are spatially nearby are more significant than isolated things.Our proposed framework exploits this to introduce a novel feature pooling method which we term Getis-Ord G * i pooling.Pooling is used to spatially downsample the feature maps in deep CNNs.In contrast to standard image downsampling methods which seek to preserve the spatial envelope of pixel values, pooling selects feature values that are more significant in some sense.The most standard pooling method is max pooling in which the maximum feature value in a window is propagated.Other pooling methods have been proposed.Average pooling is an obvious choice and is used in [[14,]
						[37]] for image classification.Strided convolution [[16]] has also been used.However, max pooling remains by far the most common as it has the intuitive appeal of extracting the maximum activation and thus the most prominent features of an image.
However, we postulate that isolated high feature values might not be the most informative and instead develop a method to propagate clustered values.Specifically, we use a technique from geostatistics termed hotspot analysis to identify clusters of large values and then propagate a representative from these clusters.Hotspot analysis uses the Getis-Ord G * i [[25]] statistic to find locations that have either high or low values and are surrounded by locations also with high or low values.These locations are the so-called hotspots.The Getis-Ord G * i statistic is computed by comparing the local sum of a feature and its neighbors proportionally to the sum of all features in a spatial region.When the local sum is different from the expected local sum, and when that difference is too large to be the result of random noise, it will lead to a high positive or low negative G * i value that is statistically significant.We focus on locations with high positive G * i values since we want to propagate activations.

## 3.2. Definition
We now describe our G-pooling algorithm in detail.Please see Figure 2<ns0:ref xmlns:ns0="http://www.tei-c.org/ns/1.0" type="figure" coords="3,381.86,392.48,4.98,8.64" target="#fig_0">2</ns0:ref> for reference. for reference.Similar to other pooling methods, we use a stride sliding window to downsample the input.Given a feature map within the stride window, in order to compute its G * i , we first need to define the weight matrix based on the spatial locations.
We denote the feature values within the sliding window as X = x 1 , x 2 , ..., x n where n is the number of pixels (locations) within the sliding window.We assume the window is rectangular and compute the G * i statistic at the center of the window.Let the feature value at the center be x i .(If the center does not fall on a pixel location then we compute x i as the average of the adjacent values.)The G * i statistic uses weighed averages where the weights are based on spatial distances.Let p x (x j ) and p y (x j ) denote the x and y positions of feature value x j in the image plane.A weight matrix w that measures the Euclidean distance on the image plane between x i and the other locations within the sliding window is then computed as
<ns0:formula xmlns:ns0="http://www.tei-c.org/ns/1.0" xml:id="formula_0" coords="3,315.20,633.43,229.91,10.49">w i,j = (p x (x i ) -p x (x j )) 2 + (p y (x i ) -p y (x j )) 2 . (1)</ns0:formula>
				$$
w i,j = (p x (x i ) -p x (x j )) 2 + (p y (x i ) -p y (x j )) 2 . (1)
$$The Getis-Ord G * i value at location i is now computed as
<ns0:formula xmlns:ns0="http://www.tei-c.org/ns/1.0" xml:id="formula_1" coords="3,352.08,680.74,144.83,35.35">G * i = n j=1 w i,j x j -X n j=1 w i,j S [n n j=1 w 2 i,j -( n j=1 wi,j ) 2 ] n-1</ns0:formula>
				$$
G * i = n j=1 w i,j x j -X n j=1 w i,j S [n n j=1 w 2 i,j -( n j=1 wi,j ) 2 ] n-1
$$.
(2) where X and S are as below,
<ns0:formula xmlns:ns0="http://www.tei-c.org/ns/1.0" xml:id="formula_2" coords="4,139.18,239.25,147.18,26.77">X = n j=1 x j n ,<ns0:label>(3)</ns0:label>
				</ns0:formula>
				$$
X = n j=1 x j n ,
$$<ns0:formula xmlns:ns0="http://www.tei-c.org/ns/1.0" xml:id="formula_3" coords="4,115.67,289.60,166.82,26.77">S = n j=1 x 2 j n -( X) 2 . (<ns0:label>4</ns0:label>
				</ns0:formula>
				$$
S = n j=1 x 2 j n -( X) 2 . (
$$<ns0:formula xmlns:ns0="http://www.tei-c.org/ns/1.0" xml:id="formula_4" coords="4,282.49,301.12,3.87,8.64">)</ns0:formula>
				$$
)
$$Spatial clusters can be detected based on the G * i value.The higher the value, the more significant the cluster is.However, the G * i value just indicates whether there is a spatial cluster or not.To achieve our goal of pooling, we need to summarize the local region of the feature map by extracting a representative value.We use a threshold to do this.If the computed G * i is greater than or equal to the threshold, a spatial cluster is detected and the value x i is used for pooling, otherwise the maximum in the window is used.
<ns0:formula xmlns:ns0="http://www.tei-c.org/ns/1.0" xml:id="formula_5" coords="4,61.34,459.09,225.02,26.67">G -pooling(x) = x i if G * i �� threshold max(x) if G * i &lt; threshold<ns0:label>(5)</ns0:label>
				</ns0:formula>
				$$
G -pooling(x) = x i if G * i �� threshold max(x) if G * i < threshold
$$It's noted that G * i is in range [-2.8,2.8]where a negative value indicates a coldspot which means a spatial scatter and a positive value indicates a hotspot which means a spatial cluster.The absolute value |G * i | indicates the significance.For example, a high positive G * i value indicates the feature is more likely to be a spatial cluster.
The output feature map produced by G-pooling is Gpooling(X) which results after sliding the window over the entire input feature map.The threshold is set to 3 different values in this work, 1.0, 1.5, 2.0.A higher threshold means the current feature map has less chance to be reported as a spatial cluster and so max pooling will be applied instead.A lower threshold causes more spatial clusters to be detected and max pooling will be applied less often.As the threshold ranges from 1.0 to 1.5 to 2.0, fewer spatial clusters/hotspots will be detected.We find that a threshold of 2.0 results in few hostpots being detected and max pooling mostly to be used.

## 3.3. Network Architecture
A pretrained VGG network [29] is used in our experiments.VGG has been widely used as a backbone in various semantic segmentation networks such as FCN [[20]], Unet [[26]], and SegNet [[3]].
					In VGG, the standard max pooling is a 2��2 window size with a stride of 1.Our proposed Gpooling uses a 4��4 window size with a stride of 4. Therefore, after applying the standard pooling, the size of feature map drops to 1/2, while with our G-pooling it drops to 1/4.A small window size is not used in our proposed G-pooling since Getis-Ord G * i analysis may not work well in such a small region.However, we tested the scenario where standard pooling is performed with a 4 �� 4 sliding window and the performance is only slightly different from that using the standard 2 �� 2 window.In general, segmentation networks using VGG16 as the backbone have 5 max pooling layers.So, when we replace max pooling with our proposed G-pooling, there will be two G-pooling and one max pooling layers.

## 4. Experiments

## 4.1. Dataset
ISPRS dataset We evaluate our method on two image datasets from the ISPRS 2D Semantic Labeling Challenge [[1]].
					These datasets are comprised of very high resolution aerial images over two cities in Germany: Vaihingen and Potsdam.While Vaihingen is a relatively small village with many detached buildings and small multi-story buildings, Potsdam is a typical historic city with large building blocks, narrow streets and dense settlement structure.The goal is to perform semantic labeling of the images using six common land cover classes: buildings, impervious surfaces (e.g.roads), low vegetation, trees, cars and clutter/background.We report test metrics obtained on the held-out test images.
Vaihingen The Vaihingen dataset has a resolution of 9 cm/pixel with tiles of approximately 2100 �� 2100 pixels.There are 33 images, from which 16 have a public ground truth.Even though the tiles consist of Infrared-Red-Green (IRRG) images and DSM data extracted from the Lidar point clouds, we use only the IRRG images in our work.We select five images for validation (IDs: 11, 15, 28, 30 and 34) and the remaining 11 for training, following [[22,]
						[28]].
					

## Potsdam
The Potsdam dataset has a resolution of 5 cm/pixel with tiles of 6000 �� 6000 pixels.There are 38 images, from which 24 have public ground truth.Similar to Vaihingen, we only use the IRRG images.We select seven images for validation (IDs: 2 11, 2 12, 4 10, 5 11, 6 7, 7 8 and 7 10) and the remaining 17 for training, again following [[22,]
						[28]].
					

## 4.2. Experimental settings
Baselines Here, we compare our proposed G-pooling with the standard max-pooling, average-pooling, stride convolution, and the recently proposed P-pooling [[35]].
					Max/average pooling is commonly for downsampling in the semantic segmentation networks that have VGG as a backbone.ResNet [[11]] is proposed without using any pooling but strided convolution.Such a network architecture has been adopted by recent studies for semantic segmentation, in particular the DeepLab series [[6]]
						[[7]]
						[[8]] and PSPNet [[39]].
					Max pooling is removed and instead strided convolution is used to downsample the feature maps while dilated convolution is used to enlarge the receptive fields.There is also work on detail preserving pooling, for example DDP [[27]] and P-pooling [[35]].
					We select the most recent one, P-pooling, which outperforms the other detail preserving methods for comparison.

## 4.3. Evaluation Metrics
We have two goals in this work, the model's segmentation accuracy and its generalization performance.Model accuracy is used to report the performance on the test/validation set using the model trained with training set within one dataset.Model generalizability is used to report the performance of the test/validation set with another dataset.In general, the domain gap between train and test/validation set from one dataset is relatively small.However, cross-dataset testing exists large domain shift problem.

## Model accuracy
The commonly used per class intersection over union (IoU) and mean IoU (mIoU) as well as the pixel accuracy are adopted for evaluating segmentation accuracy.
Model generalizability Specifically, we will perform evaluation on the ISPRS Potsdam set with a model trained on the ISPRS Vahingen set (Potsdam��Vaihingen) and reverse the order (Vaihingen��Potsdam).Pixel accuracy and mIoU are used to report the performance of the generalizability.

## 4.4. Implementation Details
Implementation of G-pooling Models are implemented using the PyTorch framework.Max-pooling, averagepooling, stride conv are provided as built-in function and P-pooling has open-source code.We implement our Gpooling in C and use the interface to connect to PyTorch for network training.We adopt the network architecture of FCN [[20]] with a backbone of a pretrained VGG-16 [[29]].
					The details of the FCN using our G-pooling can be found in Section 3.3.The results in Table 1<ns0:ref xmlns:ns0="http://www.tei-c.org/ns/1.0" type="table" coords="5,442.69,610.46,4.98,8.64" target="#tab_0">1</ns0:ref> are reported using FCN with a VGG-16 backbone. are reported using FCN with a VGG-16 backbone.
Training settings Since the image tiles are too large to be fed through a deep CNN due to limited GPU memory, we randomly extract image patches of size of 256��256 pixels as the training set.Following standard practice, we only use horizontal and vertical flipping as data augmentation during training.For testing, the whole image is split into 256��256 patches with a stride of 256.Then, the predictions of all patches are concatenated for evaluation.We train all our models using Stochastic Gradient Descent (SGD) with an initial learning rate of 0.1, a momentum of 0.9, a weight decay of 0.0005 and a batch size of 5.If the validation loss plateaus for 3 consecutive epochs, we divide the learning rate by 10.If the validation loss plateaus for 6 consecutive epochs or the learning rate is less than 1e-8, we stop the model training.We use a single TITAN V GPU for training and testing.

## 5. Effectiveness of G-pooling
In this section, we first show that incorporating geospatial knowledge into a pooling function of the standard CNN learning can improve segmentation accuracy.Then we demonstrate the promising generalization capability of our proposed G-pooling.
The segmentation accuracy on FCN using various pooling functions reported on the test set is shown in Table 1<ns0:ref xmlns:ns0="http://www.tei-c.org/ns/1.0" type="table" coords="6,278.89,632.78,3.74,8.64" target="#tab_0">1</ns0:ref>.
					.
					For G-pooling, we experiment on 3 different thresholds, which is 1.0, 1.5 and 2.0.The range of G * i value is [-2.8, 2.8].As explained in Section 3.2, higher G * i value can cause more uses of max pooling.If we set the G * i value as 2.8, then the case will be all max pooling.Qualitative results are shown in Figure 4<ns0:ref xmlns:ns0="http://www.tei-c.org/ns/1.0" type="figure" coords="6,131.15,704.51,3.74,8.64" target="#fig_2">4</ns0:ref>.
					.
					And the quantitative results for eval-uating model accuracy and cross-location generalization is shown in Table 1<ns0:ref xmlns:ns0="http://www.tei-c.org/ns/1.0" type="table" coords="6,371.74,312.13,4.98,8.64" target="#tab_0">1</ns0:ref> and 2 respectively. and 2 respectively.
Non-spatial vs geospatial statistics The baselines of pooling functions are usually non-spatial statistics, for example, finding the max/average value.Our approach provides a geospatial process to simulate how things are related based on spatial location.Here, we pose the question, "is the knowledge useful to train a deep CNN?".As we mentioned in Section 3, such a knowledge incorporated method can bring the benefit of improved generalizability.As shown in Table 1<ns0:ref xmlns:ns0="http://www.tei-c.org/ns/1.0" type="table" coords="6,385.92,424.64,3.74,8.64" target="#tab_0">1</ns0:ref>, for Potsdam, using geospatial knowledge to design the pooling function can bring 1.23% improvement compared to P-pooling., for Potsdam, using geospatial knowledge to design the pooling function can bring 1.23% improvement compared to P-pooling.Our G-pooling-1.0 and 2.0 is not able to outperform some baselines in the model accuracy testing, which indicates the threshold selection is important.Some classes of the baselines have higher performance compared to ours.This is expected since the dataset is relatively small and may be overfitting.The qualitative results in Figure 4<ns0:ref xmlns:ns0="http://www.tei-c.org/ns/1.0" type="figure" coords="6,403.30,520.28,4.98,8.64" target="#fig_2">4</ns0:ref> show our proposed G-pooling has less pepper-and-salt effect. show our proposed G-pooling has less pepper-and-salt effect.In particular, there is less noise inside the objects compared to the other methods.This demonstrates our proposed G-pooling simulates the geospatial distributions and makes the prediction within the objects more compact.The effects of threshold is shown in Table 3<ns0:ref xmlns:ns0="http://www.tei-c.org/ns/1.0" type="table" coords="6,537.64,580.05,3.74,8.64" target="#tab_2">3</ns0:ref>, as described in Section 3, the higher the threshold the less spatial cluster detected., as described in Section 3, the higher the threshold the less spatial cluster detected.
						2<ns0:ref xmlns:ns0="http://www.tei-c.org/ns/1.0" type="table" coords="6,308.86,632.78,4.98,8.64" target="#tab_1">2</ns0:ref> compares using pooling functions with using unsupervised domain adaptation (UDA). compares using pooling functions with using unsupervised domain adaptation (UDA).We note that the UDA method AdaptSegNet [[32]] uses a large amount of unlabeled data from the target dataset to adapt the model which has been demonstrated to help generalization.The other methods don't benefit from the unlabeled data.As shown in Table 2<ns0:ref xmlns:ns0="http://www.tei-c.org/ns/1.0" type="table" coords="6,345.62,704.51,3.74,8.64" target="#tab_1">2</ns0:ref>, our proposed G-pooling is able to achieve the best generalization performance., our proposed G-pooling is able to achieve the best generalization performance.For Potsdam��Vaihingen, G-pooling outperforms P-pooling by more than 2%.For Vaihingen��Potsdam, the improvement is even more significant, at least 3.41%.When we compare the knowledge incorporation method G-pooling with the domain adaptation method AdaptSegNet, the performance difference is just 0.61% for Potsdam.The results verify our assumption that incorporating knowledge helps generalizations as well.And the performance is close to that of domain adaptation which utilizes a great amount of unlabeled data to learn the data distribution.Even though knowledge incorporation doesn't outperform data-based domain adaptation, these two methods can be combined to provide even better generalization.

## Domain adaptation vs knowledge incorporation Table

## 6. G-pooling and state-of-the-art methods
In order to verify that our proposed G-pooling is able to improve state-of-the-art segmentation approaches, we select DeepLab [[6]] and SegNet [[3]] as additional network architectures to test G-pooling.As mentioned above, the models in Section 5 use FCN as the network architecture and VGG-16 as the backbone.For fair comparison with FCN, VGG-16 is also used as the backbone in DeepLab and Seg-Net.
DeepLab [[6]] uses a large receptive fields through dilated convolution.For the baseline DeepLab itself, pool4 and pool5 from the backbone VGG-16 are removed and followed by [[32]] and the dilated conv layers with a dilation rate of 2 are replaced with conv5 layers.For the G-pooling version, pool1,pool2 are replaced with G-pooling and we keep pool3.Thus there are three max pooling layers in the baseline and one G-pooling layer and one max pooling layer in our proposed version.SegNet uses an encoder-decoder architecture and preserves the max pooling index for unpooling in the decoder.Similar to Deeplab, there are 5 max pooling layers in total in the encoder of SegNet so pool1,pool2 are replaced with the proposed G pool1 and pool3,pool4 are replaced with G pool2, and pool5 is kept.This leads us to use a 4 �� 4 unpooling window to recover the spatial resolution where the original ones are just 2 �� 2. Thus there are two G-pooling and one max pooling layers in our SegNet version.
As can be seen in Table 4<ns0:ref xmlns:ns0="http://www.tei-c.org/ns/1.0" type="table" coords="7,181.48,584.96,3.74,8.64" target="#tab_3">4</ns0:ref>, G-pooling is able to improve the model accuracy for Potsdam, 67.97% �� 68.33%., G-pooling is able to improve the model accuracy for Potsdam, 67.97% �� 68.33%.And the improvement on the generalization test Potsdam��Vaihingen is even more obvious, G-pooling improves mIoU from 38.57 to 40.04.Similar observations can be made for SegNet and FCN.For Vaihingen, even though the model accuracy is not as high as the baseline, the difference is small.The mIoU of our versions of DeepLab, SegNet and FCN is less than 1% lower.We note that Vaihingen is an easier dataset than Potsdam, since it only includes urban scenes while Potsdam includes both urban and nonurban.However, the generalizability of our model using G-pooling is much better.As shown, when testing Potsdam using a model trained on Vaihingen, FCN with G-pooling is able to achieve 23.02% mIoU which is an improvement of 7.54% IoU.The same observations can be made for DeepLab and SegNet.

## 7. Discussion
Incorporating knowledge is not a novel approach for neural networks.Before deep learning, there was work on rule-based neural networks which required expert knowledge to design the network for specific applications.Due to the large capacity of deep models, deep learning has become the primary approach to address vision problems.However, deep learning is a data-driven approach which relies significantly on the amount of training data.If the model is trained with a large amount of data then it will have good generalization.But the case is often, particularly in overhead image segmentation, that the dataset is not large enough like it is in ImageNet/Cityscapes.This causes overfitting.Early stopping, cross-validation, etc. can help to avoid overfitting.Still, if domain shift exists between the training and test sets, the deep models do not perform well.In this work, we propose a knowledge-incorporated approach to reduce overfitting.We address the question of how to incorporate the knowledge directly into the deep models by proposing a novel pooling method for overhead image segmentation.But some issues still need discussing as follows.Scenarios using G-pooling As mentioned in section 3, Gpooling is developed using Getis-Ord G * i analysis which quantifies how the spatial convergence occurs.This is a simulated process design for geospatial data downsampling.Thus it's not necessarily appropriate for other image datasets.This is more general restriction of incorporating of knowledge.The Getis-Ord G * i provides a method to identify spatial clusters while training.The effect is similar to conditional random fields/Markov random fields in standard computer vision post-processing methods.However, it is different from them since the spatial clustering is dynamically changing based on the feature maps and the geospatial location while post-processing methods rely on the prediction of the models.

## Local geospatial pattern
We now explain how G-pooling works in deep neural networks.Getis-Ord G * i analysis is usually used to analyze a global region hotspot detection which describes the geospatial convergence.As shown in Figure 3<ns0:ref xmlns:ns0="http://www.tei-c.org/ns/1.0" type="figure" coords="8,79.69,644.74,3.74,8.64" target="#fig_1">3</ns0:ref>, G-pooling will be applied twice to downsample the feature map., G-pooling will be applied twice to downsample the feature map.The spatial size of the G-pooling will be 64 �� 64 and 16 �� 16 respectively.And the max-pooling will lead to the size of feature map being reduced by 1/2 while ours it will be by 1/4.This is because we want to compute G * i over a larger region.
Even though G * i is usually computed over a larger region than in our framework, it still provides captures spatial convergence within a small region.Also, two G-pooling operations are applied at different scales of feature map and so a larger region in the input image is really considered.Specifically, the first 4 �� 4 pooling window is slid over the 256 �� 256 feature map and the output feature map has size 64 �� 64.This is fed through the next conv layers and a second G-pooling is applied.At this stage, the input feature map is 64 �� 64 and so when a 4 �� 4 sliding window is now used, a region of 16 �� 16 is really considered, which is 1/16 of the whole image.
Limitations There are some limitations of our work.For example, we didn't investigate the optimal window size for performing Getis-Ord G * i analysis.We also only consider one kind of spatial pattern, clusters.And, there might be better places than pooling to incorporate knowledge in CNN architectures.

## 8. Conclusion
In this paper, we investigate how geospatial knowledge can be incorporated into deep learning for geospatial image analysis.We demonstrate that incorporating geospatial rules improves performance.We realize, though, that ours is just preliminary work into geospatial guided deep learning.We note the limitations of our approach, for example, that the prior distribution does not provide benefits for classes in which this prior knowledge is not relevant.Our proposed approach does not show much improvement on the single dataset case especially a small dataset.ISPRS Vaihingen is a very small dataset which contains around only 500 images of size of 256 �� 256.In the future, we will explore other ways to encode geographic rules so they can be incorporated into deep learning models.

## Figure

### fig_0

<ns0:graphic xmlns:ns0="http://www.tei-c.org/ns/1.0" coords="3,50.11,72.00,236.24,172.70" type="bitmap" />
			

### fig_1

<ns0:graphic xmlns:ns0="http://www.tei-c.org/ns/1.0" coords="4,56.02,72.00,224.43,100.54" type="bitmap" />
			

### fig_2

<ns0:graphic xmlns:ns0="http://www.tei-c.org/ns/1.0" coords="8,50.11,72.00,495.03,282.09" type="bitmap" />
			


## Table

### tab_0

<ns0:figure xmlns:ns0="http://www.tei-c.org/ns/1.0" type="table" xml:id="tab_0" coords="5,50.11,73.56,495.00,253.90">
				<ns0:head>Table 1 :</ns0:head>
				<ns0:label>1</ns0:label>
				<ns0:figDesc>
					<ns0:div>
						<ns0:p>
							<ns0:s coords="5,86.82,73.88,233.55,8.64">Experimental results of FCN using VGG-16 as backbone.</ns0:s>
							<ns0:s coords="5,324.96,73.56,220.15,8.96;5,50.11,85.84,173.12,8.64">Stride conv, P-pooling and ours G-pooling are used to replaced the standard max/average pooling.</ns0:s>
						</ns0:p>
					</ns0:div>
				</ns0:figDesc>
				<ns0:table coords="5,108.49,111.34,378.24,216.12">
					</ns0:table>
			</ns0:figure>
			

### tab_1

<ns0:figure xmlns:ns0="http://www.tei-c.org/ns/1.0" type="table" xml:id="tab_1" coords="6,50.11,73.56,495.00,199.26">
				<ns0:head>Table 2 :</ns0:head>
				<ns0:label>2</ns0:label>
				<ns0:figDesc>
					<ns0:div>
						<ns0:p>
							<ns0:s coords="6,87.65,73.88,104.88,8.64">Cross-location evaluation.</ns0:s>
							<ns0:s coords="6,197.95,73.56,347.16,8.96;6,50.11,85.84,217.61,8.64">We compare the generalization capability of using G-pooling with domain adaptation method AdaptSegNet which utilize the unlabeled data.</ns0:s>
						</ns0:p>
					</ns0:div>
				</ns0:figDesc>
				<ns0:table coords="6,115.27,111.02,364.68,161.80">
					</ns0:table>
			</ns0:figure>
			

### tab_2

<ns0:figure xmlns:ns0="http://www.tei-c.org/ns/1.0" type="table" xml:id="tab_2" coords="6,50.11,432.15,236.25,75.01">
				<ns0:head>Table 3 :</ns0:head>
				<ns0:label>3</ns0:label>
				<ns0:figDesc>
					<ns0:div>
						<ns0:p>
							<ns0:s coords="6,85.11,432.15,201.25,8.64;6,50.11,444.11,161.32,8.64">The average percentage of detected spatial clusters per feature map with different threshold.</ns0:s>
						</ns0:p>
					</ns0:div>
				</ns0:figDesc>
				<ns0:table coords="6,98.97,469.61,138.53,37.55">
					</ns0:table>
			</ns0:figure>
			

### tab_3

<ns0:figure xmlns:ns0="http://www.tei-c.org/ns/1.0" type="table" xml:id="tab_3" coords="7,308.86,162.59,236.25,240.57">
				<ns0:head>Table 4 :</ns0:head>
				<ns0:label>4</ns0:label>
				<ns0:figDesc>
					<ns0:div>
						<ns0:p>
							<ns0:s coords="7,343.27,162.59,201.84,8.64;7,308.86,174.23,236.25,8.96;7,308.86,186.50,26.74,8.64">Experimental results on comparing w/o and w/ proposed G-pooling for the state-of-the-art segmentation networks.</ns0:s>
							<ns0:s coords="7,340.93,186.50,204.18,8.64;7,308.86,198.46,156.30,8.64">P��V indicates the model trained on Potsdam and test on Vaihingen, and versa the verses.</ns0:s>
						</ns0:p>
					</ns0:div>
				</ns0:figDesc>
				<ns0:table coords="7,314.84,223.96,229.63,179.20">
					</ns0:table>
			</ns0:figure>
		


## References

[1] . ISPRS 2D Semantic Labeling Challenge. [http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html](http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html)
[2] . Beyond RGB: Very High Resolution Urban Remote Sensing with Multimodal Deep Networks. ISPRS Journal of Photogrammetry and Remote Sensing, 2018. 2
[3] . SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2017. 4
[4] . Comprehensive Survey of Deep Learning in Remote Sensing: Theories, Tools, and Challenges for the Community. Journal of Applied Remote Sensing, 2017. 1
[5] . Knowledge-guided Golf Course Detection using a Convolutional Neural Network Fine-tuned on Temporally Augmented Data. J. Appl. Remote Sens, 2017.
[6] . DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2017. 5
[7] . Rethinking Atrous Convolution for Semantic Image Segmentation. 2017.
[8] . Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. 2018.
[9] . Vehicle Detection in Satellite Images by Hybrid Deep Convolutional Neural Networks. IEEE Geoscience and Remote Sensing Letters, 2014. 1
[10] . No More Discrimination: Cross City Adaptation of Road Scene Segmenters. 2017.
[11] . Deep residual learning for image recognition. 2016.
[12] . CyCADA: Cycle-Consistent Adversarial Domain Adaptation. 2018.
[13] . FCNs in the Wild: Pixel-Level Adversarial and Constraint-based Adaptation. 2016.
[14] . Densely Connected Convolutional networks. 2017.
[15] . Physics-Guided Neural Networks (PGNN): An Application in Lake Temperature Modeling.
[16] . DelugeNets: Deep Networks with Efficient and Flexible Cross-Layer Information Inflows. 2017.
[17] . Deep Learning Classification of Land Cover and Crop Types Using Remote Sensing Data. IEEE Geoscience and Remote Sensing Letters, 2017.
[18] . Generalizing pooling functions in convolutional neural networks: Mixed, gated, and tree. 2016.
[19] . Semantic Labeling in Very High Resolution Images via a Self-Cascaded Convolutional Neural Network. ISPRS Journal of Photogrammetry and Remote Sensing, 2018. 2
[20] . Fully Convolutional Networks for Semantic Segmentation. 2015.
[21] . Learning Traffic as Images: A Deep Convolutional Neural Network for Large-Scale Transportation Network Speed Prediction. Sensors, 2017.
[22] . High-Resolution Aerial Image Labeling with Convolutional Neural Networks. IEEE Transactions on Geoscience and Remote Sensing, 2017.
[23] . Learning to Detect Roads in High-Resolution Aerial Images. 2010.
[24] . A Relation-Augmented Fully Convolutional Network for Semantic Segmentation in Aerial Scenes. 2019.
[25] . Local Spatial Autocorrelation Statistics: Distributional Issues and an Application. Geographical Analysis, 1995.
[26] . U-net: Convolutional networks for biomedical image segmentation. 2015.
[27] . Detailpreserving pooling in deep networks. 2018.
[28] . Fully Convolutional Networks for Dense Semantic Labelling of High-Resolution Aerial Imagery. 2016.
[29] . Very deep convolutional networks for large-scale image recognition. 2014.
[30] . Striving for simplicity: The all convolutional net. 2015.
[31] . A Computer Movie Simulating Urban Growth in the Detroit Region. Economic Geography, 1970.
[32] . Learning to Adapt Structured Output Space for Semantic Segmentation. 2018.
[33] . Domain Adaptation for Structured Output via Discriminative Representations. 2019.
[34] . Adversarial Discriminative Domain Adaptation. 2017.
[35] . Building detail-sensitive semantic segmentation networks with polynomial pooling. 2019.
[36] . Learning Building Extraction in Aerial Scenes with Convolutional Networks. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2017.
[37] . Wide Residual Networks. 2016.
[38] . Airport Detection from Remote Sensing Images using Transferable Convolutional Neural Networks. 2016.
[39] . Pyramid Scene Parsing Network. 2017.
[40] . Deep Learning in Remote Sensing: A Comprehensive Review and List of Resources. IEEE Geoscience and Remote Sensing Magazine, 2017.
[41] . Improving semantic segmentation via video propagation and label relaxation. 2019.