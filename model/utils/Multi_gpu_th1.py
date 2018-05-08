import numpy as np
from skimage import io
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim
import os
from skimage import io
import pandas as pd
import numpy as np
from skimage import img_as_float
from tqdm import tqdm

num_epochs = 200
batch_size = 500
learning_rate = 0.0001
for i in range(0,10):
    train_csv_path ='/home/boyu93/VT_coursework/Computation_Life_Sci/project/Project/preprocessed_results_th1_reduced/img_fold' + str(i) + '/train/labels_train.csv'
    train_root_path ='/home/boyu93/VT_coursework/Computation_Life_Sci/project/Project/preprocessed_results_th1_reduced/img_fold' + str(i) + '/train'
    test_csv_path ='/home/boyu93/VT_coursework/Computation_Life_Sci/project/Project/preprocessed_results_th1_reduced/img_fold' + str(i) + '/test/labels_test.csv'
    test_root_path ='/home/boyu93/VT_coursework/Computation_Life_Sci/project/Project/preprocessed_results_th1_reduced/img_fold' + str(i) + '/test'
    test_label_path = '/home/boyu93/VT_coursework/Computation_Life_Sci/project/Project/' \
        'preprocessed_results_th1_reduced/img_fold' + str(i) + '/test/test_label_run_th1.csv'
    predicted_label_path = '/home/boyu93/VT_coursework/Computation_Life_Sci/project/Project/' \
        'preprocessed_results_th1_reduced/img_fold' + str(i) + '/test/predicted_label_run_th1.csv'
    model_path = '/home/boyu93/VT_coursework/Computation_Life_Sci/project/Project/preprocessed_results_th1_reduced/img_fold' + str(i) + '/network_0505_th1.pth'


    class TumorDatasetTrain(Dataset):
        
        def __init__(self, csv_file , root_dir, transform=None):
            self.labels_frame = np.array(pd.read_csv(csv_file, skiprows=1, sep=',', header=None)).astype(np.int)
            self.root_dir = root_dir
            self.transform = transform
        
        def __len__(self):
            return len(self.labels_frame)
        
        def __getitem__(self, idx):
            img_name = str(idx)+'.png'
            img_path = os.path.join(self.root_dir, img_name)
            """
                !!!Pay attention!!!
                The image size is set here
                """
            img = np.empty(shape=(1, 102, 102))
            img[0, :, :] = (img_as_float(io.imread(img_path)) - 0.5)/0.5
            label = np.array([self.labels_frame[idx,1]-1])
            train_sample = {'image': img, 'label': label}
            
            if self.transform:
                train_sample = self.transform(train_sample)
            return train_sample


    class TumorDatasetTest(Dataset):
        
        def __init__(self, csv_file , root_dir, transform=None):
            self.labels_frame = np.array(pd.read_csv(csv_file, skiprows= 1, sep=',', header= None)).astype(np.int)
            self.root_dir = root_dir
            self.transform = transform
        
        def __len__(self):
            return len(self.labels_frame)
        
        def __getitem__(self, idx):
            img_name = str(idx) + '.png'
            img_path = os.path.join(self.root_dir, img_name)
            """
                !!!Pay attention!!!
                The image size is set here
                """
            img = np.empty(shape=(1, 102, 102))
            img[0, :, :] = (img_as_float(io.imread(img_path)) - 0.5)/0.5
            label = np.array([self.labels_frame[idx, 1]-1])
            test_sample = {'image': img, 'label': label}
            
            if self.transform:
                test_sample = self.transform(test_sample)
            return test_sample


    class ToTensor(object):
        
        def __call__(self, sample):
            image, labels = sample['image'], sample['label']
            return {'image': torch.from_numpy(image), 'label': torch.LongTensor(labels)}


    train_dataset = TumorDatasetTrain(csv_file=train_csv_path, root_dir=train_root_path,
                                      transform=transforms.Compose([ToTensor()]))
    test_dataset = TumorDatasetTest(csv_file=test_csv_path, root_dir=test_root_path, transform=transforms.Compose([ToTensor()]))
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)


    class Net(nn.Module):
        
        def __init__(self, num_of_classes):
            super(Net, self).__init__()
            # input image channel, output channels, kernel size square convolution
            # kernel
            # input size = 102, output size = 100
            self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            # input size = 50, output size = 48
            self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            # input size = 24, output size = 24
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(256)
            self.drop2D = nn.Dropout2d(p=0.25, inplace=False)
            self.vp = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(256*12*12, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, num_of_classes)
        
        def forward(self, x):
            in_size = x.size(0)
            x = F.relu(self.bn1(self.vp(self.conv1(x))))
            x = F.relu(self.bn2(self.vp(self.conv2(x))))
            x = F.relu(self.bn3(self.vp(self.conv3(x))))
            x = self.drop2D(x)
            x = x.view(in_size, -1)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x


    net = Net(num_of_classes=33)
    net = net.double()
    net = nn.DataParallel(net)
    net.cuda()
    print(net)
    print('using gpu #', torch.cuda.current_device())

    running_loss = []
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate)
    for epoch in tqdm(range(num_epochs)):
        running_loss_tmp = 0.0
        for i, sample in enumerate(dataloader_train):
            # images = Variable(image).cuda()
            images = Variable(sample['image']).cuda()
            # labels = Variable(labels).cuda()
            labels = Variable(sample['label']).cuda()
            labels = labels.view(-1)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss_tmp += loss.data
        print('epoch', epoch, ':loss is', running_loss_tmp)
        running_loss.append(running_loss_tmp)
        if (epoch > 3) and (abs(running_loss[epoch] - running_loss[epoch-1]) <= 0.0001) and (abs(running_loss[epoch - 1] - running_loss[epoch - 2]) <= 0.0001):
            break
    print('training finished')

    torch.save(net.state_dict(), model_path)

    correct = 0
    total = 0
    predicted_save = np.array([])
    test_label_save = np.array([])

    for ii, test_sample in enumerate(dataloader_test):
        test_imgs = Variable(test_sample['image']).cuda()
        test_label = Variable(test_sample['label']).cuda()
        outputs = net(test_imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += test_label.size(0)
        predicted_tmp = predicted.cpu().numpy()
        test_label_tmp = test_label.squeeze().data.cpu().numpy()
        correct += (predicted_tmp == test_label_tmp).sum()
        predicted_save = np.append(predicted_save, predicted_tmp)
        test_label_save = np.append(test_label_save, test_label_tmp)

    pd.DataFrame(predicted_save).to_csv(predicted_label_path, header=None, index= None)
    pd.DataFrame(test_label_save).to_csv(test_label_path, header=None, index= None)
    print('Accuracy of this fold is %d %%' % (100*correct/total))

