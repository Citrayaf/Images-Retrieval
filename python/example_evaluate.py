# EXAMPLE_EVALUATE  Code to evaluate example results on ROxford and RParis datasets.
# Revisited protocol has 3 difficulty setups: Easy (E), Medium (M), and Hard (H), 
# and evaluates the performance using mean average precision (mAP), as well as mean precision @ k (mP@k)
#
# More details about the revisited annotation and evaluation can be found in:
# Radenovic F., Iscen A., Tolias G., Avrithis Y., Chum O., Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking, CVPR 2018
#
# Authors: Radenovic F., Iscen A., Tolias G., Avrithis Y., Chum O., 2018

import os
import numpy as np

from scipy.io import loadmat

from dataset import configdataset
from download import download_datasets, download_features
from evaluate import compute_map

#---------------------------------------------------------------------
# Set data folder and testing parameters
#---------------------------------------------------------------------
# Set data folder, change if you have downloaded the data somewhere else
data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
# Check, and, if necessary, download test data (Oxford and Pairs), 
# revisited annotation, and example feature vectors for evaluation
download_datasets(data_root)
download_features(data_root)

# Set test dataset: roxford5k | rparis6k
test_dataset = 'rparis6k'

#---------------------------------------------------------------------
# Evaluate
#---------------------------------------------------------------------

print('>> {}: Evaluating test dataset...'.format(test_dataset)) 
# config file for the dataset
# separates query image list from database image list, when revisited protocol used
cfg = configdataset(test_dataset, os.path.join(data_root, 'datasets'))

# load query and database features
print('>> {}: Loading features...'.format(test_dataset))    
features = loadmat(os.path.join(data_root, 'features', '{}_resnet_rsfm120k_gem.mat'.format(test_dataset)))
Q = features['Q']
X = features['X']

# perform search
print('>> {}: Retrieval...'.format(test_dataset))
sim = np.dot(X.T, Q)
ranks = np.argsort(-sim, axis=0)
# revisited evaluation
gnd = cfg['gnd']

# evaluate ranks
ks = [1, 5, 10]

# search for easy
gnd_t = []
for i in range(len(gnd)):
    g = {}
    g['ok'] = np.concatenate([gnd[i]['easy']])
    g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
    gnd_t.append(g)
mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, ks)
# print(gnd)
# search for easy & hard
gnd_t = []
for i in range(len(gnd)):
    g = {}
    g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
    g['junk'] = np.concatenate([gnd[i]['junk']])
    gnd_t.append(g)
mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, ks)

# search for hard
gnd_t = []
for i in range(len(gnd)):
    g = {}
    g['ok'] = np.concatenate([gnd[i]['hard']])
    g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
    gnd_t.append(g)
mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, ks)

print('>> {}: mAP E: {}, M: {}, H: {}'.format(test_dataset, np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
print('>> {}: mP@k{} E: {}, M: {}, H: {}'.format(test_dataset, np.array(ks), np.around(mprE*100, decimals=2), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))


# perform demo
import re
from PIL import Image
import cv2

list_gmbrs = cfg['imlist']
list_gnds = cfg['qimlist']

iii = 16
sim = np.dot(X.T, Q[:,iii])
ranks = np.argsort(-sim, axis=0)
hasil = ranks[0:9]
# ------------------------------------------------------------------------------------------------------------
#Easy

gnd_t = []
for i in range(len(gnd)):
    g = {}
    g['ok'] = np.concatenate([gnd[i]['easy']])
    g['junk'] = np.concatenate([gnd[i]['junk']])
    gnd_t.append(g)
qgnd = np.array(gnd_t[iii]['ok'])

mark  = [np.in1d(qgnd, ranks)]
pos = qgnd[mark]
pos =  pos[0:5]

i=0
lebar = 0 #
panjang = 0 #
gambar_fitur = []
gambar = []
for ii in range(len(pos)):
    line = re.sub('[\[\'!@#$,]', '', list_gmbrs[pos[ii]])
    a = Image.open(os.path.join(cfg['dir_data'], 'jpg',line+'.jpg'))
    print(line)
    gumbr = np.array(a)
    masuk = cv2.cvtColor(gumbr,cv2.COLOR_BGR2RGB)
    masuk2 = cv2.resize(masuk, (416,416), interpolation = cv2.INTER_AREA)
    gambar.append(masuk2)

n = len(pos)
lebar = 0 + (416*n)
panjang = 0 + (416)
ez = np.zeros((panjang,lebar,3),dtype=np.uint8)
currentX = 0
currentY = 0    


for x in range(n):
    im = np.asarray(gambar[x])
    if currentY == 0:
        # print(currentX,im.shape[0])
        ez[currentX:im.shape[0]+currentX,currentY:im.shape[1]+currentY]=im
    else :
        # final[:im.shape[0],currentY:im.shape[1]+currentY]=im
        ez[currentX:im.shape[0]+currentX,currentY:im.shape[1]+currentY]=im
    currentY=im.shape[1]+currentY
    # print(currentY)
    if currentY >= lebar:
        currentX = im.shape[0] + currentX
        currentY = 0
#--------------------------------------------------------------------------------------------
#Hard

gnd_t = []
for i in range(len(gnd)):
    g = {}
    g['ok'] = np.concatenate([gnd[i]['hard']])
    g['junk'] = np.concatenate([gnd[i]['junk']])
    gnd_t.append(g)
qgnd = np.array(gnd_t[iii]['ok'])

mark  = [np.in1d(qgnd, ranks)]
pos = qgnd[mark]
pos =  pos[0:5]

i=0
lebar = 0 #
panjang = 0 #
gambar_fitur = []
gambar = []
for ii in range(len(pos)):
    line = re.sub('[\[\'!@#$,]', '', list_gmbrs[pos[ii]])
    a = Image.open(os.path.join(cfg['dir_data'], 'jpg',line+'.jpg'))
    print(line)
    gumbr = np.array(a)
    masuk = cv2.cvtColor(gumbr,cv2.COLOR_BGR2RGB)
    masuk2 = cv2.resize(masuk, (416,416), interpolation = cv2.INTER_AREA)
    gambar.append(masuk2)

n = len(pos)
lebar = 0 + (416*n)
panjang = 0 + (416)
harr = np.zeros((panjang,lebar,3),dtype=np.uint8)
currentX = 0
currentY = 0    


for x in range(n):
    im = np.asarray(gambar[x])
    if currentY == 0:
        # print(currentX,im.shape[0])
        harr[currentX:im.shape[0]+currentX,currentY:im.shape[1]+currentY]=im
    else :
        # final[:im.shape[0],currentY:im.shape[1]+currentY]=im
        harr[currentX:im.shape[0]+currentX,currentY:im.shape[1]+currentY]=im
    currentY=im.shape[1]+currentY
    # print(currentY)
    if currentY >= lebar:
        currentX = im.shape[0] + currentX
        currentY = 0
#---------------------------------------------------------------------------------------------------
#Medium

gnd_t = []
for i in range(len(gnd)):
    g = {}
    g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
    g['junk'] = np.concatenate([gnd[i]['junk']])
    gnd_t.append(g)
qgnd = np.array(gnd_t[iii]['ok'])

mark  = [np.in1d(qgnd, ranks)]
pos = qgnd[mark]
pos =  pos[0:5]

i=0
lebar = 0 #
panjang = 0 #
gambar_fitur = []
gambar = []
for ii in range(len(pos)):
    line = re.sub('[\[\'!@#$,]', '', list_gmbrs[pos[ii]])
    a = Image.open(os.path.join(cfg['dir_data'], 'jpg',line+'.jpg'))
    print(line)
    gumbr = np.array(a)
    masuk = cv2.cvtColor(gumbr,cv2.COLOR_BGR2RGB)
    masuk2 = cv2.resize(masuk, (416,416), interpolation = cv2.INTER_AREA)
    gambar.append(masuk2)

n = len(pos)
lebar = 0 + (416*n)
panjang = 0 + (416)
med = np.zeros((panjang,lebar,3),dtype=np.uint8)
currentX = 0
currentY = 0    


for x in range(n):
    im = np.asarray(gambar[x])
    if currentY == 0:
        # print(currentX,im.shape[0])
        med[currentX:im.shape[0]+currentX,currentY:im.shape[1]+currentY]=im
    else :
        # final[:im.shape[0],currentY:im.shape[1]+currentY]=im
        med[currentX:im.shape[0]+currentX,currentY:im.shape[1]+currentY]=im
    currentY=im.shape[1]+currentY
    # print(currentY)
    if currentY >= lebar:
        currentX = im.shape[0] + currentX
        currentY = 0
# ------------------------------------------------------------------------------------------------------------
line = re.sub('[\[\'!@#$,]', '', list_gnds[iii])
a = Image.open(os.path.join(cfg['dir_data'], 'groundtruth',line+'.jpg'))
print(line)
gumbr = np.array(a)
masuk = cv2.cvtColor(gumbr,cv2.COLOR_BGR2RGB)
masuk2 = cv2.resize(masuk, (416,416), interpolation = cv2.INTER_AREA)
cv2.imshow('',masuk2)
cv2.waitKey()

cv2.imshow('',ez)
cv2.waitKey()

cv2.imshow('',med)
cv2.waitKey()

cv2.imshow('',harr)


cv2.waitKey()
