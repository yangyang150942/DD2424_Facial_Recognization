
# coding: utf-8

# In[ ]:


import os
import time
import torch
import itertools
import torchvision
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import torchvision.models as models
import math
from torch.autograd import Function
from PIL import Image

class TripletNet(nn.Module):
    def __init__(self, cnn):
        super(TripletNet, self).__init__()
        self.embedding = cnn
    def forward(self, images_tensor):
        embeds = self.embedding(images_tensor)
        return embeds

class ZFnet(nn.Module):
    def __init__(self):
        super(ZFnet, self).__init__()
        self.features = nn.Sequential(
            #网络
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            #nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=1,stride=1),
            nn.Conv2d(64,192,kernel_size=3,stride=1,padding=1),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            nn.Conv2d(192,192,kernel_size=1,stride=1),
            nn.Conv2d(192,384,kernel_size=3,stride=1,padding=1),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            nn.Conv2d(384,384,kernel_size=1,stride=1),
            nn.Conv2d(384,256,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(256,256,kernel_size=1,stride=1),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(256,256,kernel_size=1,stride=1),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )
        self.classifier = nn.Sequential(
            #分类器
            nn.Linear(256*7*7,128*1*32),
            nn.Linear(128*1*32, 128),
        )
    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class Resnet18(nn.Module):
    def __init__(self, fwd_out_dim = 128):
        super(Resnet18, self).__init__()
        self.fwd_out_dim = fwd_out_dim
        self.resnet18 = models.resnet18(pretrained=False)
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, fwd_out_dim)
        self.init_weights()
    
    def init_weights(self):
        self.resnet18.fc.weight.data.normal_(0.0, 0.02)
        self.resnet18.fc.bias.data.fill_(0)
    
    def forward(self, images):
        embed = self.resnet18(images)
        # embed = self.batch_norm(embed)
        return embed


class LoadImgData:
    def __init__(self, dir_path, transform):
        self.images_dict = {}
        self.id_to_image = {}
        self.labels = None
        self.dir_path = dir_path
        self.transform = transform
        self.load_images()

    def load_images(self):
        self.labels = os.listdir(self.dir_path)
        for label in self.labels:
            path = os.path.join(self.dir_path, label)
            images = os.listdir(path)
            self.images_dict[label] = images
            for image_id in images:
                img_path = os.path.join(path, image_id)
                self.id_to_image[image_id] = self.transform(Image.open(img_path))

    def gen_lbs_img_pairs(self):
        lbs_img_pairs = {'labels': [], 'image_ids': []}
        for label, images in self.images_dict.items():
            num_images = len(images)
            lbs_img_pairs['labels'].extend([label] * num_images)
            lbs_img_pairs['image_ids'].extend(images)
        return lbs_img_pairs

    def get_image(self, image_id):
        return self.id_to_image[image_id]
    
class Tripleloss(nn.Module):
    def __init__(self,margin):
        super(Tripleloss,self).__init__()
        self.mg = margin
    def forward(self, anchor, positive, negative):
        margin = self.mg
        dis_pos = torch.pow(anchor - positive, 2).sum(dim = 1)
        dis_neg = torch.pow(anchor - negative, 2).sum(dim = 1)
        loss = dis_pos - dis_neg + margin
        relu = nn.ReLU()
        loss = relu(loss)
        Tloss = torch.mean(loss)
        return Tloss

def generate_minibatches(pairs, batch_size, seed):
    lbs = pairs['labels']
    img = pairs['image_ids']
    num = len(lbs)
    mini_batches = []
    sfd_prs = chaos_pairs(pairs,seed)
    sfd_lbs = sfd_prs['sfd_lbs']
    sfd_img = sfd_prs['sfd_img']
    num_totalbatches = math.floor(num / batch_size)
    for i in range(num_totalbatches):
        minibatch_lbs = sfd_lbs[i * batch_size: i * batch_size + batch_size]
        minibatch_img = sfd_img[i * batch_size: i * batch_size + batch_size]
        mini_batches.append((minibatch_lbs, minibatch_img))
    return mini_batches,sfd_lbs

def chaos_pairs(pairs, seed):
    image_ids = pairs['image_ids']
    labels = pairs['labels']
    sfd_prs = {'sfd_img': [], 'sfd_lbs': []}
    num_images = len(image_ids)
    torch.manual_seed(seed)
    perm = list(torch.randperm(num_images))
    for i in range(num_images):
        sfd_prs['sfd_img'].append(image_ids[perm[i]])
        sfd_prs['sfd_lbs'].append(labels[perm[i]])
    return sfd_prs

def to_tensor_batch(minibatch_img, ld_train):
    batch_size = len(minibatch_img)
    to_tensors = torch.zeros(batch_size, 3, 220, 220)
    for i in range(batch_size):
        id = minibatch_img[i]
        img = ld_train.get_image(id)
        to_tensors[i, :, :, :] = img
    to_tensors = Variable(to_tensors)
    with torch.cuda.device(0):
        to_tensors = to_tensors.cuda()
    return to_tensors

def gener_triplets(mini_batch, img_fdot_prs, fwd_out_dim):
    minibatch_lbs, minibatch_img = mini_batch
    img_prd = itertools.product(minibatch_img, repeat=3)
    lbs_prd = itertools.product(minibatch_lbs, repeat=3)
    triplet = []
    for img, lbs in zip(img_prd, lbs_prd):
        img_anc, img_pos, img_neg = img
        lbs_anc, lbs_pos, lbs_neg = lbs
        if (lbs_anc == lbs_pos) and (lbs_anc != lbs_neg) and (img_anc != img_pos):
            triplet.append((img_anc, img_pos, img_neg))

    num_triplets = len(triplet)
    anc = torch.zeros(num_triplets, fwd_out_dim)
    pos = torch.zeros(num_triplets, fwd_out_dim)
    neg = torch.zeros(num_triplets, fwd_out_dim)

    with torch.cuda.device(0):
        anc = anc.cuda()
        pos = pos.cuda()
        neg = neg.cuda()

    for i in range(num_triplets):
        img_anc, img_pos, img_neg = triplet[i]
        anc[i, :] = img_fdot_prs[img_anc]
        pos[i, :] = img_fdot_prs[img_pos]
        neg[i, :] = img_fdot_prs[img_neg]
    return anc, pos, neg

def memory_id(id_list, embeds):
    ids = {}
    num_id = len(id_list)
    for i in range(num_id):
        id = id_list[i]
        ids[id] = embeds[i, :]
    return ids

def final_label2embeds(model, ld_train):
    labels_list = []
    image_ids = []
    for label, images_list in ld_train.images_dict.items():
        image_ids.append(images_list[0])
        labels_list.append(label)
    images_tensor = to_tensor_batch(image_ids, ld_train)
    with torch.no_grad():
        embeds = model(images_tensor)
    label2embeds = memory_id(labels_list, embeds)
    return label2embeds

def accuracy(data, dataloader, label2embeds, model, fwd_out_dim):
    image_ids = data['image_ids']
    Y = data['labels']
    minibatch_size = 32
    minibatches,sfd_lbs = generate_minibatches(data, minibatch_size,0)
    num_data = minibatch_size*len(minibatches)
    embeds = torch.zeros(num_data, fwd_out_dim)
    start = 0
    end = 0
    for minibatch_now in minibatches:
        _, minibatch_X = minibatch_now
        minibatch_now_size = len(minibatch_X)
        end += minibatch_now_size
        images_tensor = to_tensor_batch(minibatch_X, dataloader)
        with torch.no_grad():
            embeds[start:end, :] = model(images_tensor)
        start = end
    acc = 0
    for i in range(num_data):
        embed = embeds[i]
        target_label = sfd_lbs[i]
        predicted_label = who_is_it(label2embeds, embed)
        if predicted_label == target_label:
            acc += 1
    return acc, num_data

def who_is_it(label2embeds, embed):
    labels = []
    num_labels = len(label2embeds)
    fwd_out_dim = embed.shape[0]
    embeds = torch.zeros(num_labels, fwd_out_dim)
    i = 0
    for label, cur_embed in label2embeds.items():
        labels.append(label)
        embeds[i, :] = cur_embed
        i += 1
    dist = torch.pow(embeds - embed, 2).sum(dim = 1)
    index = torch.argmin(dist).tolist()
    return labels[index]

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((220, 220)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))
                                    ])

ld_train = LoadImgData("./train_dataset", transform)
lbs_img_train = ld_train.gen_lbs_img_pairs()
ld_test = LoadImgData("./test_dataset", transform)
lbs_img_test = ld_test.gen_lbs_img_pairs()
fwd_out_dim = 128
model = TripletNet(ZFnet())
#model = Resnet18(fwd_out_dim)
num_epochs = 200
minibatch_size = 32
eta = 1e-4
margin = 0.2
with torch.cuda.device(0):
    model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr = eta)
Tloss = Tripleloss(margin)
for epoch in range(num_epochs):
        model.train()
        minibatches,sfd_lbs = generate_minibatches(lbs_img_train, minibatch_size, epoch)
        loss = []
        for minibatch_now in minibatches:
            model.zero_grad()
            images_tensor = to_tensor_batch(minibatch_now[1], ld_train)
            embeds = model(images_tensor)
            id_embeds = memory_id(minibatch_now[1], embeds)
            anchor, positive, negative = gener_triplets(minibatch_now, id_embeds, fwd_out_dim)
            if anchor.shape[0] != 0:
                l = Tloss.forward(anchor, positive, negative)
                loss.append(l)
                l.backward()
                optimizer.step()
        label2embeds = final_label2embeds(model, ld_train)
        train_acc, num_train = accuracy(lbs_img_train, ld_train, label2embeds, model,fwd_out_dim)
        test_acc, num_test = accuracy(lbs_img_test, ld_test, label2embeds, model,fwd_out_dim)
        print('%d %f %f %f' 
        %(epoch, torch.mean(torch.Tensor(loss)), train_acc/num_train, test_acc/num_test))

