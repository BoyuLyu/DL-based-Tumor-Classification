# Deep Learning Based Tumor Type Classification Using Gene Expression Data
https://dl.acm.org/citation.cfm?id=3233588

# Description
In the folder "model" are the codes used for this paper. The main parts are the training and testing as well as the GradCam. Data are downloaded from http://gdac.broadinstitute.org/

# Data
(1) Raw data from Broad Institute Firehose

https://virginiatech-my.sharepoint.com/:u:/g/personal/boyu93_vt_edu/EX83oKOMrKJKnGzfwuQvUgcBoERS0kUB-eQOQOwk2ZBuxw?e=hL0H0S

(2) Data after preprocessing (training & testing images 102x102)

https://virginiatech-my.sharepoint.com/:u:/g/personal/boyu93_vt_edu/EVsGVrZghoBCiNel6sKQYbABTcLnNbxuaUgP8IsASigmrg?e=ynhXRv


Each of the subfolder contains the images used for training and testing for each fold in the cross validation. Also, the weights trained after each fold are included (.pth file).

# Hardware

I implemented my codes using the Newriver cluster of Virginia Tech with Tesla P100 GPU.
https://www.arc.vt.edu/computing/newriver/

# Citation 
Please cite:

https://dl.acm.org/citation.cfm?id=3233588

Boyu Lyu and Anamul Haque. 2018. Deep Learning Based Tumor Type Classification Using Gene Expression Data. In Proceedings of the 2018 ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics (BCB '18). ACM, New York, NY, USA, 89-96. DOI: https://doi.org/10.1145/3233547.3233588
