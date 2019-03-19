# Deep Learning Based Tumor Type Classification Using Gene Expression Data
https://dl.acm.org/citation.cfm?id=3233588

# Description
In the folder "model" are the codes used for this paper. The main parts are the training and testing as well as the GradCam. Data are downloaded from http://gdac.broadinstitute.org/

# Data
(1) Raw data from Broad Institute Firehose

https://drive.google.com/drive/folders/1LfOiyMgnoQy3jaJ37jLeARfw7riLwkyW?usp=sharing

(2) Data after preprocessing (training & testing images 102x102)

https://drive.google.com/file/d/1zUepILj0is71LxPAWAZKmJ7-Kk7L9_XO/view?usp=sharing

Each of the subfolder contains the images used for training and testing for each fold in the cross validation. Also, the weights trained after each fold are included (.pth file).

# Hardware

I implemented my codes using the Newriver cluster of Virginia Tech with Tesla P100 GPU.
https://www.arc.vt.edu/computing/newriver/
