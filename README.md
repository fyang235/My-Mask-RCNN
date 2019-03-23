# My-Mask-RCNN
My Implementation of Mask-RCNN with MS-COCO dataset

# Background
Mask_RCNN is the a neural network designed for object detection and smantic segementation tasks. It was the state of art algorithm when published. The orignal paper is [here](https://arxiv.org/abs/1703.06870).

The Mas_RCNN algrithm coroperates many sub-nets such as Region Proposal Net (RPN), Feature Pyramid Net (FPN), Fully Convolutional Net (FCN) as well as ResNet which severs as the backbone network. Diving into the implementation of Mask_RCNN is a good way to grasp the constructing and per forman of these sub-nets and thus using them as bricks in future.

# Requirements
1. Tensorflow
2. Keras
3. MicroSoft COCO dataset
4. Python

# Usage
1. For training:
    train a new model:   
      ```python3 train --dataset=/path/to/your/dataset   ```
    use pre_trained model:   
      ```python3 train --dataset=/path/to/your/dataset --model_path=/path/to/your/pre_trained_model  ```
2. For evaluating:   
      ```python2 evaluate --dataset=/path/to/your/dataset --model_path=/path/to/your/trained_model  ```
      
# Testing
I tested some modules of the Mask_RCNN model, check out the jupyter notebook files for details. These documents are not very well organised and I keep updating them later.   

Some examples are shown below.  

## Test visulize
Visulize the ground truth label to make sure our data is loaded correctly.

![](https://github.com/fyang235/My-Mask-RCNN/blob/master/images/test_visulize.png)    
 
## Test RPN ground truth  
The RPN ground truth is generate using the anchors and the image ground turth. The Intersection
 of Union (IoU) is the metric of selecting positive and negative samples. Anchors whose 
 IoU is larger than 70% is positive candidate and those less than 30% is considered as 
 negative candidates. Finally, balance the portion of positive and negative samples to make 
 sure positive count is not greater than half and a total number of 256 samples are generated
 for RPN training.
 
![](https://github.com/fyang235/My-Mask-RCNN/blob/master/images/test_rpn_bbox_gt.png)    
 
 
## Use mini masks
In order to save memory and accelerate training, a 28x28 mini mask is used. 
The original mask is croped and minimized to target size.
To get better mask resoluton and smoother bounary, bigger mini mask size can be employed.   

![](https://github.com/fyang235/My-Mask-RCNN/blob/master/images/test_original_mask.png.png)  
![](https://github.com/fyang235/My-Mask-RCNN/blob/master/images/original_mask.png)    
 
## Test anchor generation  
The anchors are generated at each other pixel (stride=2) with 3 ratios (0.5, 1.0, 2.0) and 
5 scales (512, 256, 128, 64, 32). 
Using 1024x1024 images as input, a total number of 65472 anchors will be generated.   

![](https://github.com/fyang235/My-Mask-RCNN/blob/master/images/test_anchor_generation.png)   
![](https://github.com/fyang235/My-Mask-RCNN/blob/master/images/test_anchor_generation2.png)   

## Test apply regression results to boxes
The RPN model predicts the score of every anchor and calculate the refinements of positive
anchors. 256 anchors are used to train the RPN. Apply refinements to boxes to get the region 
of interests (RoIs). The number of RoIs is reduced to 2000 for training and 1000 for 
inference by ProposalLayer.  

![](https://github.com/fyang235/My-Mask-RCNN/blob/master/images/test_apply_delta_to_box1.png)   
![](https://github.com/fyang235/My-Mask-RCNN/blob/master/images/test_apply_delta_to_box2.png)   

## Test ProposalLayer
The proposal layer take the output of RPN and removes the lower score anchors and conducts 
non-maximum supression. The anchors also need to be clipped to fit the image window.

![](https://github.com/fyang235/My-Mask-RCNN/blob/master/images/test_proporal_layer.png)   

## Test DetectionTargetLayer
The DetectionTargetLayer generate ground truth labels for mask rcnn classifier and regressor 
heads, 200 RoIs are fed to training.

![](https://github.com/fyang235/My-Mask-RCNN/blob/master/images/test_detection_target_layer1.png)   
![](https://github.com/fyang235/My-Mask-RCNN/blob/master/images/test_detection_target_layer2.png)   
![](https://github.com/fyang235/My-Mask-RCNN/blob/master/images/test_detection_target_layer3.png)   
![](https://github.com/fyang235/My-Mask-RCNN/blob/master/images/test_detection_target_layer4.png)   


## Test data generator
Since the COCO dataset is 20 GB large and there is no way to fit it into the memory, a data
generator is build to fit the training and validation data into the model online. Test the
generator and make sure it functions well.

![](https://github.com/fyang235/My-Mask-RCNN/blob/master/images/test_mrcnn_generator.png)   


## Test model predictions
Finally, we can test the entire model. As you can see the model makes correct predictions for
most instances but miss classifies the barrel to cup as well. I did not train this model for
a long time there should be improvement if trained longer with better hyper parameters.

![](https://github.com/fyang235/My-Mask-RCNN/blob/master/images/test_model_prediction.png)  

Enjoy!




