import os
import torch
import itertools
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Function
from torchvision.io import read_image
from shutil import copyfile

img_height = 220
img_width = 220
img_per_label_min_plus = 20
img_per_label_max_plus = 30
class DataReader():
    def __init__(self, data_dir, transform):
        self.images_dict = {}
        self.id2image = {}
        self.labels = None
        self.dir_path = data_dir
        self.transform = transform
        self.load_images()
  
    def load_images(self):
        # returns labels/names list
        self.labels = os.listdir(self.dir_path)
        for label in self.labels:
            path = os.path.join(self.dir_path, label)
            images = os.listdir(path)
            self.images_dict[label] = images
            for image_id in images:
                img_path = os.path.join(path, image_id)
                self.id2image[image_id] = self.transform(Image.open(img_path))
    
    def generate_data(self):
        labels = []
        image_ids = []
        for label, images in self.images_dict.items():
            num_images = len(images)
            labels.extend([label] * num_images)
            image_ids.extend(images)
        return image_ids, labels
        
    def get_image(self, image_id):
        return self.id2image[image_id]
    
    def __len__(self):
        return len(self.img_labels)
    

# Divide data set into training, validation, and test
class DataDivider():
    def __init__(self,data_dir,train_dir,valid_dir,test_dir,img_per_label_min,img_per_label_max):
        self.labels = None
        self.data_dir = data_dir
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.img_per_label_min = img_per_label_min
        self.img_per_label_max = img_per_label_max
        self.data_dict = {}
        self.train_dict = {}
        self.valid_dict = {}
        self.test_dict = {}
        self.num_train = 0
        self.num_valid = 0
        self.num_test = 0
        self.num_total = 0
        self.num_total_label = 0
        
        self.img_select()
        self.divide_data()
        self.create_dataset()
        
    # Only select people who have 20 or more pictures
    def img_select(self):
        self.labels = os.listdir(self.data_dir)
        for label in self.labels:
            dir_this = os.path.join(self.data_dir,label)
            imgs_this = os.listdir(dir_this)
            cnt_this = len(imgs_this)
            if cnt_this >= self.img_per_label_min and cnt_this <= self.img_per_label_max:
                self.data_dict[label] = imgs_this
                self.num_total += cnt_this
                self.num_total_label += 1
    
    # For each person, 2 for validation, 2 for test
    def divide_data(self):
        for label,image_list in self.data_dict.items():
            num_img_this = len(image_list)
            if num_img_this >= img_per_label_min_plus:
                num_img_valid = 2
                num_img_test = 2
                num_img_train = num_img_this - num_img_valid - num_img_test
                self.num_train += num_img_train
                self.num_valid += num_img_valid
                self.num_test += num_img_test
                self.train_dict[label] = image_list[:num_img_train]
                self.valid_dict[label] = image_list[num_img_train:num_img_train+num_img_valid]
                self.test_dict[label] = image_list[num_img_train+num_img_valid:]
            else:
                num_img_train = num_img_this
                self.num_train += num_img_train
                self.train_dict[label] = image_list[:num_img_train]

    def create_dataset(self):
        self.output_data(self.train_dict, train_dir)
        self.output_data(self.valid_dict, valid_dir)
        self.output_data(self.test_dict, test_dir)
        print("OK!")
    
    def output_data(self,data_dict,output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for label,image_list in data_dict.items():
            label_dir_src = os.path.join(self.data_dir,label)
            label_dir_out = os.path.join(output_dir,label)
            if not os.path.exists(label_dir_out):
                os.makedirs(label_dir_out)
            
            for image in image_list:
                img_path_src = os.path.join(label_dir_src,image)
                img_path_out = os.path.join(label_dir_out,image)
                copyfile(img_path_src,img_path_out)
    
    
data_dir = '../lfw-deepfunneled'
train_dir = '../train_set'
valid_dir = '../valid_set'
test_dir = '../test_set'

transform = transforms.Compose([transforms.Resize((img_height, img_width)), 
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))
                                ])

# Divide original dataset to training, validation and test
dataset = DataDivider(data_dir,train_dir,valid_dir,test_dir,img_per_label_min_plus,img_per_label_max_plus)
print(dataset.num_total,dataset.num_total_label)
print(dataset.num_train,dataset.num_valid,dataset.num_test)

# Create training, validation, and test dataset
#datareader_train = DataReader(train_dir,transform)
##dataset_train = datareader_train.generate_data()
#datareader_valid = DataReader(valid_dir,transform)
#dataset_valid = datareader_valid.generate_data()
#datareader_test = DataReader(test_dir,transform)
#dataset_test = datareader_test.generate_data()

## show an example image
#x = datareader_valid.get_image("Gray_Davis_0023.jpg")
#plt.imshow(x.numpy()[0])
#plt.show()