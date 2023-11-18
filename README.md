# Pancreas_segmentation

## Do this before running the code.

## Open the files in Sample_image and Model with 7-Zip and decompress them.

## Change the path in the Code/Path: path=your_file_location(Pancreas_segmentation)
![Screenshot_64](https://github.com/jayasurya321/Pancreas_segmentation/assets/150137811/e3208219-feac-45f7-91ed-d71c834c824a)

## Install python on the local machine.
## Run pip install tensorflow in the terminal.

## Run pip install -r requirements.txt in the terminal.




![P1](https://github.com/jayasurya321/Pancreas_segmentation/assets/150137811/a39e1928-875b-4925-a925-bf51cc1cf573)



![P5](https://github.com/jayasurya321/Pancreas_segmentation/assets/150137811/34ecadb3-32af-4467-91a6-25930f22697b)

![P6](https://github.com/jayasurya321/Pancreas_segmentation/assets/150137811/d80b634d-9023-4b33-bb31-1703472d02c3)

![P7](https://github.com/jayasurya321/Pancreas_segmentation/assets/150137811/1f8477f0-3fbb-4de7-9171-e409190b52c1)

____________________________________________________________________________________________________________________________________________________________________

# Abstract

The segmentation of the pancreas from an abdominal CT scan is a formidable challenge due to the considerable variability in its shape, size, and area. The organ is quite diminutive in size compared to others and occupies a modest amount within the belly. The segmentation of the pancreas is a difficult task due to these considerations. Our proposal involves the implementation of a 3D-Net model to effectively address the challenges at hand. First, the Hounsfield unit (HU) of the CT scan is used to separate the pancreas from the area around it using a segmentation method based on intensity. An overlaid mask is utilized to identify the approximate area of the pancreas. The NIH pancreatic dataset, which is made up of 82 abdomen contrast-enhanced CT volumes, is used to evaluate the proposed study. The proposed study obtained a competitive outcome with a Dice Similarity Coefficient value of 0.6956.


EXPERIMENT CONFIGURATION

Dataset
The NIH-provided dataset serves as the basis for testing the proposed work. It consists of 82 abdominal contrast-enhanced 3D CT images and has been manually labeled the segmentation of the pancreas as ground-truth slice-by-slice. We use 74 of them for training and 8 are used for testing.

The training is done on a system with 30GB free space, 32GB(2x16GB) Ram DDR4 3200, and 16GB GPU Memory. The training of the U-Net takes around 2 hours for 60 epochs.

![image](https://github.com/jayasurya321/Pancreas_segmentation/assets/150137811/46cdfc60-9241-4d83-a4c6-8b12911bafc9)

Results and Discussion

In this work, we proposed a pancreas segmentation method with intensity-based segmentation to reduce the burden of network training and increase accuracy. 

![image](https://github.com/jayasurya321/Pancreas_segmentation/assets/150137811/c2ffac0a-bb93-43da-b709-5b1df61e2f61)

Figure 2 Loss Function Graph

![image](https://github.com/jayasurya321/Pancreas_segmentation/assets/150137811/9f36a9b3-2f6f-48c6-b4bf-de3e4ad11b22)

Figure 3 Precision Function Graph

![image](https://github.com/jayasurya321/Pancreas_segmentation/assets/150137811/2be1a7a2-d135-4fb6-9752-c98c91e3fef2)

Figure 4 Recall Function Graph

Table 1 and 2 shows the performance in the train and test images respectively.


	         Precision	Recall	DSC
           Mean	   0.8905  	0.8025	0.84

Table 2. Model’s Performance in Training Images

The high recall and low precision shows that the model has a high number of false negatives. This is also seen in figure 4 where the prediction covers not only the ground truth but also the surrounding area.

	         Precision	Recall	 DSC
          Mean	  0.8003   	0.6151	0.6956

Table 2. Model’s Performance in Testing Images


Conclusion and Future Work	

In this work, the proposed work uses intensity-based segmentation and 3D -Net to segment the pancreas from an abdominal CT scan. The model is applied to the NIH dataset of pancreas containing 82 contrast-enhanced CT scans and has achieved competitive results and demonstrated efficiency. The proposed work is also flexible and can be used for the segmentation of other organs as well.

The current version of the proposed work still has room for improvement. The size of the pancreas is small and variable and therefore results in low scores. Future improvements can introduce mechanisms to address the small size of the pancreas and class imbalance and to sample patches effectively. More complex deep learning architectures can be tested such as V-Net, Mask R-CNN, etc. Transfer learning can be explored to improve the ability of the model to generalize better. Overall improvements can be made from a better and larger dataset to train the model more accurately. 
