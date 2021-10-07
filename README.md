# Cloud Segmentation 
## A Solution involving Pixel Level Classification
- Traditional Image segmentation revolve around Clustering and PCA, in Deep Learning, this paradigm can be solved with powerful convolutional and transformer based algorithims,
UNet Family with ResNet backbones are the go to method to start with. 
- The dataset is open sourced, thanks to Kaggle : 38CloudsDataset
- Here, I have experimented with UNet with Resnet Backbone and used different losses because of computational limits, I have been wanting to experiment with Attention based UNets and Transformer based Segmentation algorithms however, the Kaggle compute limitations and absence of computationally efficent writeup of algorithms,my hands were tied but one thing i could experiment with limited compute was didfferent penalties to mistakes.
- Following writeup talks about the data makeup which is imbalanced and ways of to tackling it, and other quirks such as incorporating Weights & Biases into your code for better result visualizations and inferences.

### Exploratory Data Analysis
### Importance of Channels:
![image](https://user-images.githubusercontent.com/47039231/136316384-038558d4-a623-43c7-96af-d330916decdf.png)
- Here, the importance of Near infra-red channel is evident, I
see a substantial amount of depth information about the object is preserved by Near Infrared compared to
RGB Bands where its hard to say anything about presence of clouds. My Highschool Physics 
intuition says since near infra-red has higher wavelength, it’s very hard to 
scatter, thus has more depth information compared to other colour channels like 
Red, Green, Blue which get scattered more easily. Thus, NIR is a very important 
channel in this dataset and cannot be discarded. Thus, I worked with 4 channel 
images in all of the experiments.
- Working with 4 channels means no pretrained weights without moidyfingn earlier layers of network, a sacrifice to be sure but a welcome one.
### Imbalance in Labels:
![image](https://user-images.githubusercontent.com/47039231/136316605-8a156cd0-c8c0-41c6-b7ff-6ca8fdbf9cbe.png)

                                  1. Y Axis: Counts of Labels that is per pixel count. Label - 0, Label - 1

                                  2. X Axis: Labels (0: Background and 1: Cloud)

                                  3. Label 0: 863885830 , Label 1: 374744570 
-  Le Meme has arrived 
 ![image](https://user-images.githubusercontent.com/47039231/136317455-0a5f6f15-1afc-4c10-b93b-86305800b207.png)
 
- This plot makes it clear that there is a substantial imbalance in classes, so i need to indicate to remeber with while penalising algortihms mistake, although inducing a new bias to solve an exisiting bias isnt an optimal soluiton, rather a work around, the community still debates about which is the best method to mitigate imbalances, 
So I construct/choose our loss function 
in such a way this information is used while penalising the errors made by the 
model for respective labels. E.g in case of Binary cross entropy this info could be 
passed and the function penalises accordingly as a pos_weight argument in this 
case which is the ratio of classes.

                          Criterion : nn.BCEWithLogitsLoss(pos_weight=torch.tensor(Ratio))

                          Ratio = Label 0(Majority)/Label 1 (Minority)

                          Where, Ratio in this case is 2.3053
### Loss Functions
- A several loss functions were experimented taking into account the imbalance 
namely

          • Focal Loss

          • Focal Tversky Loss

          • Weighted BCE and Vanilla BCE

          • Dice Loss

          • Jaccard/IOU Loss

          • Mixed Focal Loss (alpha * Focal Loss + beta * Dice coefficient)

- However, I used only Focal, Mixed Focal, Focal Tversky, Weighted BCE. 
- Reason: After doing a brief research for many segmentation problems especially 
when there is a imbalance, many users recommend them, also considering time 
limitations I used the ones which I could understand. In a similar Kaggle 
competition with cloud segmentation the Leader board topper used focal loss thus 
I went with it

### Models
- A variety of models were considered mostly from UNet Family, as it is known 
for its performances and reliability, since time was a hurdle, it made sense to stick 
to tried and tested models.
Following models were implemented from scratch

        • Vanilla UNet
        
        • UNet with Attention Mechanism
        
        • R2UNet (Not implemented from scratch)
        
        • UNet with Resnet encoder
### Training Approaches
- A combination of above models and above loss functions were experimented.
However, due to a technical issue which is still unclear, my kernels were 
incredibly slow and never allowed me to train my models, on average each 
training epoch took 35-45 mins and each validation epoch took 10 minutes 
making it incredibly harder to even tune the hyperparameters. Almost every 
kernel crashed in the process. On attention unets it took 1hr+ since all I could fit 
was 5 images in one batch. Even there was of a memory allocation issue with the 
written from scratch models, since they are not as optimized as the ones offered 
by different distros like segmentations_pytorch. For, e.g. the 2 versions of UNets 
written from scratch could hardly fit more than 10 images in some experiments, 
thus switched to pytorch_segmentation resnet encoder which allowed me to fit 
12-16 images in a batch without crashing frequently. Thus, my results are from 
those models because they were the only ones which I could train for longer 
durations. Owing to the limitations, in some experiments I decided to randomly
split the data into 50% and train 5 epochs on that notice the metrics and tune the 
hyperparameters if need be and sampled the data again continued the process
recurrently, while in others I trained on 6000 images and validated on 2400 
images. 
- A Summary of All Approaches I experimented. I used Weights and Biases to log 
my metrics
