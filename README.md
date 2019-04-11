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

# Citation 
Please cite:
@inproceedings{Lyu:2018:DLB:3233547.3233588,
 author = {Lyu, Boyu and Haque, Anamul},
 title = {Deep Learning Based Tumor Type Classification Using Gene Expression Data},
 booktitle = {Proceedings of the 2018 ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics},
 series = {BCB '18},
 year = {2018},
 isbn = {978-1-4503-5794-4},
 location = {Washington, DC, USA},
 pages = {89--96},
 numpages = {8},
 url = {http://doi.acm.org/10.1145/3233547.3233588},
 doi = {10.1145/3233547.3233588},
 acmid = {3233588},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {convolutional neural network, deep learning, pan-cancer atlas, tumor type classification},
} 
