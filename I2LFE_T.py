import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats
import pickle
import random
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import utils
import models
from scipy import io
import torch.fft as fft
import spectral
# np.random.seed(1337)
pcachannel = 20
parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 160)
parser.add_argument("-c","--src_input_dim",type = int, default = 128)
parser.add_argument("-d","--tar_input_dim",type = int, default = 21)# PaviaU=103；salinas=204;MUUFL=64;Trento=63;Houston=144
parser.add_argument("-d2","--tar_input_dim2",type = int, default = 1)#MUUFL=2;Trento=1;Houston=1
parser.add_argument("-n","--n_dim",type = int, default = 100)
parser.add_argument("-w","--class_num",type = int, default = 6) #MUUFL=11;Trento=6;Houston=15
parser.add_argument("-s","--shot_num_per_class",type = int, default = 1)
parser.add_argument("-b","--query_num_per_class",type = int, default = 19)
parser.add_argument("-e","--episode",type = int, default= 20000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
# target
parser.add_argument("-m","--test_class_num",type=int, default=6) #MUUFL=11;Trento=6;Houston=15
parser.add_argument("-z","--test_lsample_num_per_class",type=int,default=5, help='5 4 3 2 1')

args = parser.parse_args(args=[])
patchsize = 4
# Hyper Parameters
FEATURE_DIM = args.feature_dim  ##160
SRC_INPUT_DIMENSION = args.src_input_dim ##128
TAR_INPUT_DIMENSION = args.tar_input_dim ##64
TAR_INPUT_DIMENSION2 = args.tar_input_dim2
N_DIMENSION = args.n_dim ##100
CLASS_NUM = args.class_num  ##16
SHOT_NUM_PER_CLASS = args.shot_num_per_class ##1
QUERY_NUM_PER_CLASS = args.query_num_per_class ##19
EPISODE = args.episode
TEST_EPISODE = args.test_episode ##600
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit ##10

# Hyper Parameters in target domain data set
TEST_CLASS_NUM = args.test_class_num # the number of class ##11
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class # the number of labeled samples per class 5 4 3 2 1 ##5
train_indices2 = []
test_indices2 = []
da_train_indices2 = []

utils.same_seeds(0)
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')
_init_()

# load source domain data set
with open(os.path.join('datasets',  'Chikusei_imdb_128.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)
print(source_imdb.keys())
print(source_imdb['Labels']) ##source_imdb（77592,9,9,128） data,set,label set[1 1 1 1...]

# process source domain data set
data_train = source_imdb['data'] # (77592, 9, 9, 128)
labels_train = source_imdb['Labels'] # 77592
print(data_train.shape)
print(labels_train.shape)
keys_all_train = sorted(list(set(labels_train)))  # class [0,...,18]
print(keys_all_train) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print(label_encoder_train) ##[0:0,1:1,2:2,3:3....18:18]

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
print(train_set.keys())
data = train_set  ##19
del train_set
del keys_all_train
del label_encoder_train

print("Num classes for source domain datasets: " + str(len(data)))  ##19
print(data.keys()) ##[5 0 8 13 7 6 4 17 10 1 3 12 16 14 2 9 11 15 18]
data = utils.sanity_check(data) # 200 labels samples per class
print("Num classes of the number of class larger than 200: " + str(len(data))) ##data：# 18个类，每类200个样本

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))  # （9,9,100）-> (100,9,9)
        data[class_][i] = image_transpose

# source few-shot classification data
metatrain_data = data
print(len(metatrain_data.keys()), metatrain_data.keys()) ##18 [5，0,8,13,7,6,4,170,1,1,3,12,1,14,2,9,11,15]
del data

# source domain adaptation data
print(source_imdb['data'].shape) # (77592, 9, 9, 100) ##(77592,9,9,128)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0)) #(9, 9, 100, 77592)
print(source_imdb['data'].shape) # (77592, 9, 9, 100)##(9,9,128,77592)
print(source_imdb['Labels'])
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=128, shuffle=True, num_workers=0)
del source_dataset, source_imdb

#######################################################################################################################
#######################################################################################################################
## target domain data set
# load target domain data set
# test_data1 = 'datasets/Trento/HSI.mat'
# test_label1 = 'datasets/Trento/gt.mat'
test_data2 = 'datasets/Trento/LiDAR.mat'
test_label2 = 'datasets/Trento/gt.mat'

test_label1 = 'datasets/Trento/gt.mat'
input_hsi = loadmat('datasets/Trento/HSI.mat')['HSI']
input_lidar = loadmat('datasets/Trento/LiDAR.mat')['LiDAR']
# 高光谱降维处理
array_x1 = input_hsi.reshape(np.prod(input_hsi.shape[:2]), np.prod(input_hsi.shape[2:]))
pca = PCA(n_components=pcachannel)
array_x2 = pca.fit_transform(array_x1)
print(array_x2.shape)
pca_hsi = array_x2.reshape(input_hsi.shape[0], input_hsi.shape[1], array_x2.shape[1])
lidar = input_lidar.reshape(input_lidar.shape[0], input_lidar.shape[1], 1)
print('hsi shape:', input_hsi.shape)
print('pca hsi shape:', pca_hsi.shape)
print('lidar shape:', lidar.shape)
concatenated_array = np.concatenate((pca_hsi, lidar), axis=2)
io.savemat('test_data_T5.mat',{'test_data_T5':concatenated_array})
test_data1 = 'test_data_T5.mat'

Data_Band_Scaler, GroundTruth = utils.load_data(test_data1, test_label1)  ##(145,145,200) (145,145)
Data_Band_Scaler2, GroundTruth2 = utils.load_lidar(test_data2, test_label2)


# get train_loader and test_loader  ##target
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class): ##(145,145,200) (145,145) 16 5
    print(Data_Band_Scaler.shape) ##(145,145,200)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape ## nRow:145 nColumn:145 nBand:200

    '''label start'''
    num_class = int(np.max(GroundTruth)) ##16
    data_band_scaler = utils.flip(Data_Band_Scaler) ##(435,,435,200)
    groundtruth = utils.flip(GroundTruth)  ##(435,435)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = patchsize
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth] ##左右各扩4 (153,153)
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,:] ##(153,153,200)

    [Row, Column] = np.nonzero(G)  # (10249,) (10249,) ##Row:10249 Column:10249
    # print(Row)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row) ##153
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_train = {} # Data Augmentation ##数据扩增
    m = int(np.max(G))  # 16
    nlabeled =TEST_LSAMPLE_NUM_PER_CLASS  ##5
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1) ##32.5
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)  ##向上取整 33.0

    for i in range(m): ##i从0到15 ##索引indices:len46
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1] ##enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        np.random.shuffle(indices)
        nb_val = shot_num_per_class  ##5
        train[i] = indices[:nb_val]  ##0-4 九类每类5个样本
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1): ##0-40
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    global train_indices2
    train_indices2 = train_indices
    global test_indices2
    test_indices2 = test_indices
    global da_train_indices2
    da_train_indices2 = da_train_indices

    print('the number of train_indices:', len(train_indices))  ##80
    print('the number of test_indices:', len(test_indices)) ##10169
    print('the number of train_indices after data argumentation:', len(da_train_indices))  ##3200
    print('labeled sample indices:',train_indices)

    nTrain = len(train_indices) ##80
    nTest = len(test_indices) ##10169
    da_nTrain = len(da_train_indices) ##3200

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32) ##(9,9,200,10249)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64) ##10249
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64) ##10249

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest): ##iSample:3199
        imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data HSI is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class,shuffle=False, num_workers=0)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training 数据扩增
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],  dtype=np.float32)  ## (9,9,200,3200)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64) ##3200 [0 -15]
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64) ##3200

    da_RandPerm = np.array(da_train_indices) ##3200
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = utils.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-16 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return train_loader, test_loader, imdb_da_train ,G,RandPerm,Row, Column,nTrain ##RandPerm Row Column:10249 nTrain:80


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train,G,RandPerm,Row, Column,nTrain = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler,  GroundTruth=GroundTruth, \
                                                                     class_num=class_num,shot_num_per_class=shot_num_per_class)  # 9 classes and 5 labeled samples per class
    train_datas, train_labels = train_loader.__iter__().__next__() ##train_datas(80,200,9,9)
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape) # size of train datas: torch.Size([80,200,9,9])

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)  ## (9, 9, 200, 3200)
    print(imdb_da_train['Labels']) ##[0 0 0...15 15 15]
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  ##(3200,200,9,9) # (9,9,100, 1800)->(1800, 100, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']  ## (3200,)
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    # target domain : batch samples for domian adaptation
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=128, shuffle=True, num_workers=0)
    del target_dataset

    print(RandPerm)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    return train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain

def get_train_test_loader2(Data_Band_Scaler2, GroundTruth2, class_num, shot_num_per_class): ##(145,145,200) (145,145) 16 5
    print(Data_Band_Scaler2.shape) ##(145,145,200)
    [nRow, nColumn, nBand] = Data_Band_Scaler2.shape ## nRow:145 nColumn:145 nBand:200
    RandPerm2 = RandPerm

    '''label start'''
    num_class = int(np.max(GroundTruth2)) ##16
    data_band_scaler2 = utils.flip(Data_Band_Scaler2) ##(435,,435,200)
    groundtruth2 = utils.flip(GroundTruth2)  ##(435,435)
    del Data_Band_Scaler2
    del GroundTruth2

    HalfWidth = patchsize
    G2 = groundtruth2[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth] ##左右各扩4 (153,153)
    data = data_band_scaler2[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,:] ##(153,153,200)

    [Row2, Column2] = np.nonzero(G)  # (10249,) (10249,) ##Row:10249 Column:10249
    # print(Row)
    del data_band_scaler2
    del groundtruth2

    nSample = np.size(Row2) ##153
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_train = {} # Data Augmentation ##数据扩增
    m = int(np.max(G))  # 16
    nlabeled =TEST_LSAMPLE_NUM_PER_CLASS  ##5
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1) ##32.5
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)  ##向上取整 33.0

    for i in range(m): ##i从0到15 ##索引indices:len46
        indices = [j for j, x in enumerate(Row2.ravel().tolist()) if G[Row2[j], Column2[j]] == i + 1] ##enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        np.random.shuffle(indices)
        nb_val = shot_num_per_class  ##5
        train[i] = indices[:nb_val]  ##0-4 九类每类5个样本
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1): ##0-40
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    np.random.shuffle(test_indices2)

    print('the number of train_indices:', len(train_indices2))  ##80
    print('the number of test_indices:', len(test_indices2)) ##10169
    print('the number of train_indices after data argumentation:', len(da_train_indices2))  ##3200
    print('labeled sample indices:',train_indices2)

    nTrain2 = len(train_indices2) ##80
    nTest = len(test_indices2) ##10169
    da_nTrain2 = len(da_train_indices2) ##3200

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain2 + nTest], dtype=np.float32) ##(9,9,200,10249)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([nTrain2 + nTest], dtype=np.int64) ##10249
    imdb['set'] = np.zeros([nTrain2 + nTest], dtype=np.int64) ##10249

    print(RandPerm2)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    for iSample in range(nTrain2 + nTest): ##iSample:3199
        imdb['data'][:, :, :, iSample] = data[Row2[RandPerm2[iSample]] - HalfWidth:  Row2[RandPerm2[iSample]] + HalfWidth + 1,
                                         Column2[RandPerm2[iSample]] - HalfWidth: Column2[RandPerm2[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row2[RandPerm2[iSample]], Column2[RandPerm2[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain2]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data LiDAR is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader2 = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class,shuffle=False, num_workers=0)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader2 = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training 数据扩增
    imdb_da_train2 = {}
    imdb_da_train2['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain2],  dtype=np.float32)  ## (9,9,200,3200)
    imdb_da_train2['Labels'] = np.zeros([da_nTrain2], dtype=np.int64) ##3200 [0 -15]
    imdb_da_train2['set'] = np.zeros([da_nTrain2], dtype=np.int64) ##3200

    da_RandPerm = np.array(da_train_indices2) ##3200
    for iSample in range(da_nTrain2):  # radiation_noise，flip_augmentation
        imdb_da_train2['data'][:, :, :, iSample] = utils.radiation_noise(
            data[Row2[da_RandPerm[iSample]] - HalfWidth:  Row2[da_RandPerm[iSample]] + HalfWidth + 1,
            Column2[da_RandPerm[iSample]] - HalfWidth: Column2[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train2['Labels'][iSample] = G[Row2[da_RandPerm[iSample]], Column2[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train2['Labels'] = imdb_da_train2['Labels'] - 1  # 1-16 0-15
    imdb_da_train2['set'] = np.ones([da_nTrain2]).astype(np.int64)
    print('ok')

    return train_loader2, test_loader2, imdb_da_train2 ,G2,RandPerm2,Row2, Column2,nTrain2 ##RandPerm Row Column:10249 nTrain:80


def get_target_dataset2(Data_Band_Scaler2, GroundTruth2, class_num, shot_num_per_class):
    train_loader2, test_loader2, imdb_da_train,G2,RandPerm2,Row2, Column2,nTrain2 = get_train_test_loader2(Data_Band_Scaler2=Data_Band_Scaler2,  GroundTruth2=GroundTruth2, \
                                                                     class_num=class_num,shot_num_per_class=shot_num_per_class)  # 9 classes and 5 labeled samples per class
    train_datas, train_labels = train_loader.__iter__().__next__() ##train_datas(80,200,9,9)
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape) # size of train datas: torch.Size([80,200,9,9])

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)  ## (9, 9, 200, 3200)
    print(imdb_da_train['Labels']) ##[0 0 0...15 15 15]
    del Data_Band_Scaler2, GroundTruth2

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  ##(3200,200,9,9) # (9,9,100, 1800)->(1800, 100, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']  ## (3200,)
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data2 = target_da_train_set
    print(target_da_metatrain_data.keys())

    # target domain : batch samples for domian adaptation
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    target_dataset2 = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader2 = torch.utils.data.DataLoader(target_dataset2, batch_size=128, shuffle=True, num_workers=0)
    del target_dataset2

    return train_loader2, test_loader2, target_da_metatrain_data2, target_loader2,G2,RandPerm2,Row2, Column2,nTrain2

class channel_attention(nn.Module):
    # ratio代表第一个全连接的通道下降倍数
    def __init__(self, in_channel, ratio=4):
        super().__init__()

        # 全局最大池化 [b,c,h,w]==>[b,c,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 第一个全连接层, 通道数下降4倍（可以换成1x1的卷积，效果相同）
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # 第二个全连接层, 恢复通道数（可以换成1x1的卷积，效果相同）
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)

        # relu激活函数
        self.relu = nn.ReLU()

        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        b, c, h, w = inputs.shape

        # 输入图像做全局最大池化 [b,c,h,w]==>[b,c,1,1]
        max_pool = self.max_pool(inputs)

        # 输入图像的全局平均池化 [b,c,h,w]==>[b,c,1,1]
        avg_pool = self.avg_pool(inputs)

        # 调整池化结果的维度 [b,c,1,1]==>[b,c]
        max_pool = max_pool.view([b, c])
        avg_pool = avg_pool.view([b, c])

        # 第一个全连接层下降通道数 [b,c]==>[b,c//4]

        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)

        # 激活函数
        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)

        # 第二个全连接层恢复通道数 [b,c//4]==>[b,c]
        # （可以换成1x1的卷积，效果相同）
        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)

        # 将这两种池化结果相加 [b,c]==>[b,c]
        x = x_maxpool + x_avgpool

        # sigmoid函数权值归一化
        x = self.sigmoid(x)

        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])

        # # 输入特征图和通道权重相乘 [b,c,h,w]
        # outputs = x2 * x
        # outputs2 = inputs * x

        return x #, outputs, outputs2,


# （2）空间注意力机制
class spatial_attention(nn.Module):
    # 卷积核大小为7*7
    def __init__(self, kernel_size=7):
        super().__init__()

        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2

        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)

        # 在通道维度上平均池化 [b,1,h,w]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)

        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)

        # 空间权重归一化
        x = self.sigmoid(x)

        # # 输入特征图和空间权重相乘
        # outputs = x2 * x
        # outputs2 = inputs * x

        return x #, outputs, outputs2,

class Separable_CNN(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0, kernel_size=1, stride=1):
        super(Separable_CNN, self).__init__()
        self.depth_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )
        self.point_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class lidar_spat_feature(nn.Module):
    def __init__(self, input_tensor):
        super(lidar_spat_feature, self).__init__()
        # Spatial Branch
        self.conv21 = Separable_CNN(in_channels=1, out_channels=24,
                                            padding=0 , kernel_size=(1, 1, 1), stride=(1, 1, 1))
                                            # kernel_size=(1, 1, band), stride=(1, 1, 1))
        # Dense block
        self.batch_norm21 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv22 = Separable_CNN(in_channels=24, out_channels=12, padding=(1, 1, 0),
                                            kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv23 = Separable_CNN(in_channels=36, out_channels=12, padding=(1, 1, 0),
                                            kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm23 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv24 = Separable_CNN(in_channels=48, out_channels=12, padding=(1, 1, 0),
                                            kernel_size=(3, 3, 1), stride=(1, 1, 1))

    def forward(self, X):
        X = X.unsqueeze(1)
        x21 = self.conv21(X)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1)
        return x25


def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
    )
    return layer

class residual_block(nn.Module):

    def __init__(self, in_channel,out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel,out_channel)
        self.conv2 = conv3x3x3(out_channel,out_channel)
        self.conv3 = conv3x3x3(out_channel,out_channel)

    def forward(self, x): #(1,1,100,9,9)
        x1 = F.relu(self.conv1(x), inplace=True) #(1,8,100,9,9)  (1,16,25,5,5)
        x2 = F.relu(self.conv2(x1), inplace=True) #(1,8,100,9,9) (1,16,25,5,5)
        x3 = self.conv3(x2) #(1,8,100,9,9) (1,16,25,5,5)

        out = F.relu(x1+x3, inplace=True) #(1,8,100,9,9)  (1,16,25,5,5)
        return out

class D_Res_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2):
        super(D_Res_3d_CNN, self).__init__()

        self.block1 = residual_block(in_channel,out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4,2,2),padding=(0,1,1),stride=(4,2,2))
        self.block2 = residual_block(out_channel1,out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4,2,2),stride=(4,2,2), padding=(2,1,1))
        self.conv = nn.Conv3d(in_channels=out_channel2,out_channels=32,kernel_size=3, bias=False)

        self.final_feat_dim = 160

    def forward(self, x): #x:(400,100,9,9)
        x = x.unsqueeze(1) # (400,1,100,9,9)
        x = self.block1(x) #(1,8,100,9,9)
        x = self.maxpool1(x) #(1,8,25,5,5)
        x = self.block2(x) #(1,16,25,5,5)
        x = self.maxpool2(x) #(1,16,7,3,3)
        x = self.conv(x) #(1,32,5,1,1)
        x = x.view(x.shape[0],-1) #(1,160)
        # y = self.classifier(x)
        return x


class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x

class SelfAttention_HL(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention_HL, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        # 计算 query, key, value
        query = self.query(x1)  # [128, 160]
        key = self.key(x2)  # [128, 160]
        value = self.value(x2)

        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(key.size(-1), dtype=torch.float32))  # [128, 128]
        attention = self.softmax(scores)  # [128, 128]
        out = torch.matmul(attention, value)

        return attention, out

class LowFrequencyEnhancementModuleConv(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2):
        super(LowFrequencyEnhancementModuleConv, self).__init__()

        # 使用卷积层进行低频增强
        self.c1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.c2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        # 假设输入x形状为 [batch_size, feature_dim]，需要调整为 [batch_size, 1, feature_dim]
        x = x.unsqueeze(1)  # 增加一个维度，转换为 [batch_size, 1, feature_dim]

        # 第一次卷积操作
        x_conv1 = self.c1(x)
        # 第二次卷积操作
        x_conv2 = self.c2(x_conv1)

        # 使用残差连接
        return x.squeeze(1) + x_conv2.squeeze(1)  # 返回增强后的结果

# class LowFrequencyEnhancementModuleConv(nn.Module):
#     def __init__(self):
#         super(LowFrequencyEnhancementModuleConv, self).__init__()
#
#         # 全连接层替代卷积层进行低频增强和高频抑制
#         self.low_freq_fc = nn.Linear(160, 160)
#         self.high_freq_fc = nn.Linear(160, 160)
#
#         # 残差连接：对输入做全连接层，调整通道数
#         self.residual_fc = nn.Linear(160, 160)
#
#         # 中间全连接层，用于低频和高频的组合处理
#         self.middle_fc = nn.Linear(160, 160)
#
#         # 激活函数
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         # 低频增强：使用全连接层来提取低频信息
#         low_freq = self.low_freq_fc(x)
#
#         # 高频抑制：使用全连接层来减少高频信息
#         high_freq = self.high_freq_fc(x)
#
#         # 残差连接：对输入做全连接层，调整通道数
#         residual = self.residual_fc(x)
#
#         # 低频和高频的加减操作中间添加全连接层
#         combined = low_freq - high_freq
#         combined = self.middle_fc(combined)  # 插入全连接层
#
#         # 将低频增强信息与原始输入高频信息结合，使用残差连接
#         output = combined + residual
#
#         # 激活函数
#         output = self.relu(output)
#
#         # 返回最终输出
#         return output

class LowFrequencyEnhancementModule(nn.Module):
    def __init__(self):
        super(LowFrequencyEnhancementModule, self).__init__()

        # 低频增强卷积层
        self.low_freq_conv1 = nn.Conv2d(100, 64, kernel_size=5, stride=1, padding=2)
        self.low_freq_bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # 高频抑制卷积层
        self.high_freq_conv = nn.Conv2d(100, 64, kernel_size=3, stride=1, padding=1)

        # 残差连接
        self.residual_conv = nn.Conv2d(100, 64, kernel_size=1, stride=1, padding=0)  # 对输入做1x1卷积

        # 输出通道扩展层
        self.conv3 = nn.Conv2d(64, 160, kernel_size=1, stride=1, padding=0)

        # 全局平均池化层
        self.pool2 = nn.AdaptiveAvgPool2d(1)  # 将空间维度压缩为1x1

    def forward(self, x):
        # 低频增强部分：使用较大的卷积核来提取低频信息
        low_freq = self.low_freq_conv1(x)
        low_freq = self.low_freq_bn1(low_freq)
        low_freq = self.relu(low_freq)

        # 高频抑制部分：使用较小的卷积核来减少高频信息
        high_freq = self.high_freq_conv(x)

        # 残差连接：对原始输入做1x1卷积
        residual = self.residual_conv(x)

        # 将低频信息与输入的高频信息结合
        output = low_freq - high_freq + residual

        # 通道扩展至160
        output = self.conv3(output)

        # 全局平均池化，去除空间维度
        output = self.pool2(output)

        # 去除多余的维度，使输出形状为[128, 160]
        output = output.view(x.size(0), -1)  # 压缩为[128, 160]

        # 返回最终输出
        return output


class LowFrequencyEnhancementModule2(nn.Module):
    def __init__(self):
        super(LowFrequencyEnhancementModule2, self).__init__()

        # 低频增强卷积层：使用较大的卷积核来提取低频信息
        self.low_freq_conv = nn.Conv2d(60, 160, kernel_size=5, stride=1, padding=2)

        # 高频抑制卷积层：使用较小的卷积核来减少高频信息
        self.high_freq_conv = nn.Conv2d(60, 160, kernel_size=3, stride=1, padding=1)

        # 残差连接：对原始输入做1x1卷积，调整通道数
        self.residual_conv = nn.Conv2d(60, 160, kernel_size=1, stride=1, padding=0)

        # 全局平均池化，将空间维度[9,9]压缩为1x1
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 低频增强：使用较大的卷积核来提取低频信息
        low_freq = self.low_freq_conv(x)

        # 高频抑制：使用较小的卷积核来减少高频信息
        high_freq = self.high_freq_conv(x)

        # 残差连接：对输入做1x1卷积，调整通道数
        residual = self.residual_conv(x)

        # 将低频增强信息与原始输入高频信息结合，使用残差连接
        output = low_freq - high_freq + residual

        # 激活函数
        output = self.relu(output)

        # 全局平均池化，将空间维度压缩为1x1
        output = self.global_pool(output)

        # 将通道信息拉平为[100, 160]，去掉空间维度
        output = output.view(output.size(0), -1)

        # 返回最终输出
        return output

class Network(nn.Module):
    def __init__(self, n_bands, n_classes):
        super(Network, self).__init__()
        self.n_bands = n_bands
        self.n_classes = n_classes
        self.lidar_spat = lidar_spat_feature(self)
        self.hsi_feature = D_Res_3d_CNN(1,8,16)
        self.conv2 = nn.Conv2d(60, 160, kernel_size=9)#60,160
        self.SAM = spatial_attention()
        self.CAM = channel_attention(160)
        self.conv2d = nn.Conv2d(2, 1, kernel_size=1)

        self.gate = nn.Parameter(torch.ones(1))
        self.gate2 = nn.Parameter(torch.ones(1))

        # self.gated_unit = GatedFusionUnit(160)
        self.low_freq_module = LowFrequencyEnhancementModule()
        self.low_freq_module2 = LowFrequencyEnhancementModule2()
        self.low_freq_module_F = LowFrequencyEnhancementModuleConv()

        self.att = SelfAttention_HL(160)

        self.final_feat_dim = FEATURE_DIM
        self.classifier = nn.Linear(160, out_features=CLASS_NUM)#160
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.target2_mapping = Mapping(TAR_INPUT_DIMENSION2, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)

    def forward(self, x1, x2, domain='target'):
        if domain == 'target':
            x1 = self.target_mapping(x1)
            # print(x1.shape) #torch.Size([128, 100, 9, 9])

            output_tensor = self.low_freq_module(x1)
            # print(output_tensor.shape)

            feature1 = self.hsi_feature(x1)
            # x2 = self.target2_mapping(x2)
            feature2 = self.lidar_spat(x2)
            # feature2 = self.attention_spatial(feature2)
            feature2 = feature2.squeeze(2)

            output_tensor2 =  self.low_freq_module2(feature2)
            # print((feature2.shape)) #torch.Size([100, 60, 9, 9])
            # feature2 = feature2.reshape(feature2.shape[0], -1, 9, 9)
            # _, f2 = self.Attention(feature2)
            feature2 = self.conv2(feature2)
            # feature2 = feature2.view(feature2.shape[0], -1)  # (128,160)
            feature1 = torch.unsqueeze(feature1, -1)
            feature1 = torch.unsqueeze(feature1, -1)
            # LMSE1 = criterion(feature1, feature2)
            c_att1 = self.CAM(feature1)
            c_att2 = self.CAM(feature2)
            c_att = c_att1 * c_att2
            c_att1 = c_att + c_att1
            c_att2 = c_att + c_att2
            feature1 = feature1 * c_att1
            feature2 = feature2 * c_att2

            s_att2 = self.SAM(feature2)
            s_att1 = self.SAM(feature1)
            s_att = s_att1 * s_att2
            s_att1 = torch.cat([s_att, s_att1], dim=1)
            s_att2 = torch.cat([s_att, s_att2], dim=1)
            s_att1 = self.conv2d(s_att1)
            s_att2 = self.conv2d(s_att2)
            feature1 = feature1 * s_att1
            feature2 = feature2 * s_att2

            # feature = torch.cat([feature1, feature2], dim=1)
            feature = feature1 + feature2
            feature = self.gate * feature

            feature = feature.view(feature.shape[0], -1)  # [30,220*9*9]
            feature1 = feature1.view(feature1.shape[0], -1)
            feature2 = feature2.view(feature2.shape[0], -1)
            output_tensor_F = self.low_freq_module_F(feature)

            output_tensor = output_tensor# + feature1
            output_tensor = self.low_freq_module_F(output_tensor)
            output_tensor2 = output_tensor2# + feature2
            output_tensor2 = self.low_freq_module_F(output_tensor2)
            output_tensor = output_tensor + output_tensor_F #torch.cat([output_tensor, feature], dim=1)

        output = self.classifier(output_tensor)
        return output_tensor, output, feature1, feature2#, Lmmd

class Feature(nn.Module):
    def __init__(self, n_bands, n_classes):
        super(Feature, self).__init__()
        self.n_bands = n_bands
        self.n_classes = n_classes
        self.feature_encoder = D_Res_3d_CNN(1,8,16)
        self.final_feat_dim = FEATURE_DIM  # 64+32
        self.classifier = nn.Linear(in_features=self.final_feat_dim, out_features=CLASS_NUM)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)

    def forward(self, x, domain='soure'):  # x
        # print(x.shape)
        x = self.source_mapping(x)
        feature = self.feature_encoder(x)
        output = self.classifier(feature)
        return feature, output

class MMD_loss(nn.Module):
    def __init__(self, kernel_type='linear', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            loss = self.linear_mmd2(source, target)
            return loss
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            # print("rrrrrrrrrrrrbbbbbbbbbbbbbbbffffffffffff")
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            # with torch.no_grad():
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            # del XX, YY, XY, YX
            # torch.cuda.empty_cache()
            return loss

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.xavier_uniform_(m.weight, gain=1)
#         if m.bias is not None:
#             m.bias.data.zero_()
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight, 1.0, 0.02)
#         m.bias.data.zero_()
#     elif classname.find('Linear') != -1:
#
#         nn.init.xavier_normal_(m.weight)
#         if m.bias is not None:
#             m.bias.data = torch.ones(m.bias.data.size())

def weights_init(m):
    if isinstance(m, nn.Conv1d):  # 如果是卷积层
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):  # 如果是批归一化层
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):  # 如果是全连接层
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())


crossEntropy = nn.CrossEntropyLoss().cuda()
domain_criterion = nn.BCEWithLogitsLoss().cuda()
# criterion = nn.MSELoss().cuda()

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits



# run 10 times
nDataSet = 1
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None
i=0
seeds = [1220]#[1330, 1220, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]
for iDataSet in range(nDataSet):
    # load target domain data for training and testing
    np.random.seed(seeds[iDataSet])
    train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS) ##(145,145,200) (145,145) 16 5
    train_loader2, test_loader2, target_da_metatrain_data2, target_loader2, G2, RandPerm2, Row2, Column2, nTrain2 = get_target_dataset2(
        Data_Band_Scaler2=Data_Band_Scaler2, GroundTruth2=GroundTruth2, class_num=TEST_CLASS_NUM,shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)
    # model
    encoder = Network(TAR_INPUT_DIMENSION, CLASS_NUM)
    F_encoder = Feature(TAR_INPUT_DIMENSION, CLASS_NUM)
    mmd = MMD_loss(kernel_type='rbf')

    F_encoder.apply(weights_init)
    encoder.apply(weights_init)

    encoder.cuda( )
    F_encoder.cuda()

    F_encoder.train()
    encoder.train()

    # optimizer
    feature_encoder_optim = torch.optim.Adam(F_encoder.parameters(), lr=args.learning_rate)
    feature_encoder_optim2 = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)

    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    target_iter2 = iter(target_loader2)
    len_dataloader = min(len(source_loader), len(target_loader))
    len_dataloader2 = min(len(source_loader), len(target_loader2))
    train_start = time.time()

    for episode in range(2000):  # EPISODE = 90000
        # get domain adaptation data from  source domain and target domain
        try:
            source_data, source_label = source_iter.__next__()
        except Exception as err:
            source_iter = iter(source_loader)
            source_data, source_label = source_iter.__next__()

        try:
            target_data, target_label = target_iter.__next__()
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, target_label = target_iter.__next__()

        try:
            target_data2, target_label2 = target_iter2.__next__()
        except Exception as err:
            target_iter2 = iter(target_loader2)
            target_data2, target_label2 = target_iter2.__next__()

        # source domain few-shot + domain adaptation
        if episode % 2 == 0:
            '''Few-shot claification for source domain data set'''
            # get few-shot classification samples 拿小样本数据做支持集和查询集
            task = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

            # sample datas
            supports, support_labels = support_dataloader.__iter__().__next__()
            querys, query_labels = query_dataloader.__iter__().__next__()

            # calculate features 特征提取
            support_features, support_outputs = F_encoder(supports.cuda())
            query_features, query_outputs = F_encoder(querys.cuda())
            target_features, target_outputs, target_HSI, target_lidar = encoder(target_data.cuda(), target_data2.cuda(), domain='target')
            source_features = torch.cat([support_features, query_features])
            # # print("source_features: ",source_features.size()) #torch.Size([120, 160])
            # # print("target_features: ", target_features.size()) #torch.Size([128, 241])
            # source_features = source_features.reshape(150,128)
            # target_features = target_features.reshape(241,128)

            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)
            else:
                support_proto = support_features

            # fsl_loss
            logits = euclidean_metric(query_features, support_proto)
            f_loss = crossEntropy(logits, query_labels.cuda().long())

            # MMD
            # domain_loss = mmd(source_features, target_features)
            domain_loss1 = mmd(source_features, target_features)
            # domain_loss2 = mmd(source_features, target_HSI)
            domain_loss3 = mmd(source_features, target_lidar)
            domain_loss = domain_loss3 + domain_loss1# + domain_loss1

            # total_loss = fsl_loss + domain_loss
            loss = f_loss + domain_loss

            # Update parameters
            F_encoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            feature_encoder_optim2.step()
            # print("source loss")

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels.long()).item()
            total_num += querys.shape[0]
        # target domain few-shot + domain adaptation
        else:
            '''Few-shot classification for target domain data set'''
        # get few-shot classification samples
            task = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

            task2 = utils.Task(target_da_metatrain_data2, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS,QUERY_NUM_PER_CLASS)
            support_dataloader2 = utils.get_HBKC_data_loader(task2, num_per_class=SHOT_NUM_PER_CLASS, split="train",shuffle=False)
            query_dataloader2 = utils.get_HBKC_data_loader(task2, num_per_class=QUERY_NUM_PER_CLASS, split="test",shuffle=True)

            # sample datas
            supports, support_labels = support_dataloader.__iter__().__next__()
            querys, query_labels = query_dataloader.__iter__().__next__()

            supports2, support_labels2 = support_dataloader2.__iter__().__next__()
            querys2, query_labels2 = query_dataloader2.__iter__().__next__()

            # calculate features
            support_features, support_outputs, support_HSI, support_lidar= encoder(supports.cuda(),  supports2.cuda(), domain='target')
            query_features, query_outputs, query_HSI, query_lidar= encoder(querys.cuda(), querys2.cuda(), domain='target')
            source_features, source_outputs = F_encoder(source_data.cuda())
            target_features = torch.cat([support_features, query_features])
            target_HSI = torch.cat([support_HSI, query_HSI])
            target_lidar = torch.cat([support_lidar, query_lidar])
            # # print("source_features: ",source_features.size()) #torch.Size([128, 160])
            # # print("target_features: ",target_features.size()) #torch.Size([120, 241])
            # source_features = source_features.reshape(-1, 40)
            # target_features = target_features.reshape(-1, 40)

            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)
            else:
                support_proto = support_features

            # fsl_loss
            logits = euclidean_metric(query_features, support_proto)
            f_loss = crossEntropy(logits, query_labels.cuda().long())

            # MMD
            # domain_loss = mmd(source_features, target_features)
            domain_loss1 = mmd(source_features, target_features)
            # domain_loss2 = mmd(source_features, target_HSI)
            domain_loss3 = mmd(source_features, target_lidar)
            domain_loss = domain_loss3 + domain_loss1#  + domain_loss3

            # total_loss = fsl_loss + domain_loss
            loss = f_loss + domain_loss

            # Update parameters
            F_encoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            feature_encoder_optim2.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels.long()).item()
            total_num += querys.shape[0]

        if (episode + 1) % 50 == 0:  # display
            train_loss.append(loss.item())
            print('episode {:>3d}:  fsl loss: {:6.4f}, acc {:6.4f}, loss: {:6.4f}'.format(
                episode + 1, \
                # domain_loss.item(),
                f_loss.item(),
                total_hit / total_num,
                loss.item()))
            # print('episode {:>3d}:  fsl loss: {:6.4f}, acc {:6.4f}, loss: {:6.4f}'.format(
            #     episode + 1, \
            #     f_loss.item(),
            #     total_hit / total_num,
            #     loss.item()))

        if (episode + 1) % 200 == 0 or episode == 0:
            # test
            print("Testing ...")
            train_end = time.time()
            encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)


            train_datas, train_labels = train_loader.__iter__().__next__()
            train_datas2, train_labels2 = train_loader2.__iter__().__next__()
            train_features, _, _, _ = encoder(Variable(train_datas).cuda(),Variable(train_datas2).cuda(), domain='target')

            max_value = train_features.max()
            min_value = train_features.min()
            print(max_value.item())
            print(min_value.item())
            train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

            KNN_classifier = KNeighborsClassifier(n_neighbors=1)
            KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)  # .cpu().detach().numpy()

            for (test_datas1, test_labels1), (test_datas2, test_labels2) in zip(test_loader, test_loader2):
                test_features, _, _, _ = encoder(Variable(test_datas1).cuda(), Variable(test_datas2).cuda(), domain='target')
                batch_size = test_labels1.shape[0]

                test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())
                test_labels = test_labels1.numpy()
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                total_rewards += np.sum(rewards)
                counter += batch_size

                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)

                accuracy = total_rewards / 1.0 / counter  #
                accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)

            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format( total_rewards, len(test_loader.dataset),
                100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()

            # Training mode
            encoder.train()
            if test_accuracy > last_accuracy:
                # save networks
                torch.save(encoder.state_dict(),str("checkpoints/DFSL_feature_encoder_" + "Trento" +str(iDataSet) +"iter_" + str(TEST_LSAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                print("save networks for episode:",episode+1)
                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                OA = acc
                C = metrics.confusion_matrix(labels, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)

                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

            print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))

    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_G,best_RandPerm,best_Row, best_Column,best_nTrain = G, RandPerm, Row, Column, nTrain
    print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
    print('***********************************************************************************')

AA = np.mean(A, 1)

AAMean = np.mean(AA,0)
AAStd = np.std(AA)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

OAMean = np.mean(acc)
OAStd = np.std(acc)

kMean = np.mean(k)
kStd = np.std(k)
print ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
print ("average OA: " + "{:.2f}".format( OAMean) + " +- " + "{:.2f}".format( OAStd))
print ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
print ("accuracy for each class: ")
for i in range(CLASS_NUM):
    print ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))


best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))

#################classification map################################

for i in range(len(best_predict_all)):  # predict ndarray <class 'tuple'>: (9729,)
    best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = best_predict_all[i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0, 0, 1]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0, 1, 0]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [0, 1, 1]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [1, 0, 0]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [1, 0, 1]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [1, 1, 0]
        # if best_G[i][j] == 7:
        #     hsi_pic[i, j, :] = [0.5, 0.5, 1]
        # if best_G[i][j] == 8:
        #     hsi_pic[i, j, :] = [0.65, 0.35, 1]
        # if best_G[i][j] == 9:
        #     hsi_pic[i, j, :] = [0.75, 0.5, 0.75]
        # if best_G[i][j] == 10:
        #     hsi_pic[i, j, :] = [0.75, 1, 0.5]
        # if best_G[i][j] == 11:
        #     hsi_pic[i, j, :] = [0.5, 1, 0.65]
        # if best_G[i][j] == 12:
        #     hsi_pic[i, j, :] = [0.65, 0.65, 0]
        # if best_G[i][j] == 13:
        #     hsi_pic[i, j, :] = [0.75, 1, 0.65]
        # if best_G[i][j] == 14:
        #     hsi_pic[i, j, :] = [0, 0, 0.5]
        # if best_G[i][j] == 15:
        #     hsi_pic[i, j, :] = [0, 1, 0.75]
        # if best_G[i][j] == 16:
        #     hsi_pic[i, j, :] = [0.5, 0.75, 1]

utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24,  "classificationMap/Trento_{}shot.png".format(TEST_LSAMPLE_NUM_PER_CLASS))
