from torch.utils.data import Dataset
import os
import cv2
import mediapipe as mp
import numpy as np
import shutil
import random


def moveFile(train_img_Dir):
    test_list = []
    for subdir in os.listdir(train_img_Dir):
        path = os.path.join(train_img_Dir, subdir)
        if not os.path.isdir(os.path.join(test_path, subdir)):
            os.mkdir(os.path.join(test_path, subdir))
        img_pathDir = os.listdir(path)  # 提取图片的原始路径
        filenumber = len(img_pathDir)
        # 自定义test的数据比例
        test_rate = 0.2  # 如0.2，就是20%的意思
        test_picknumber = int(filenumber * test_rate)  # 按照test_rate比例从文件夹中取一定数量图片
        # 选取移动到test中的样本
        sample1 = random.sample(img_pathDir, test_picknumber)  # 随机选取picknumber数量的样本图片
        for i in range(0, len(sample1)):
            sample1[i] = sample1[i][:-4]  # 去掉图片的拓展名，移动标注时需要这个列表
        for name in sample1:
            test_list.append(int(name))
            src_img_name1 = os.path.join(os.path.join(train_path, subdir), name)
            dst_img_name1 = os.path.join(os.path.join(test_path, subdir), name)
            shutil.move(src_img_name1 + '.JPG', dst_img_name1 + '.JPG')  # 加上图片的拓展名，移动图片
    return test_list


data_path = './data/'
dataset_path = os.path.join(data_path, "SLDD")
train_path = './data/train'
test_path = './data/test'
test_list = []
if not os.path.isdir(train_path):
    os.mkdir(train_path)
    # mp.solutions.hands，是人的手
    mp_hands = mp.solutions.hands

    mpDraw = mp.solutions.drawing_utils

    # 参数：1、是否检测静态图片，2、手的数量，3、检测阈值，4、跟踪阈值
    hands_mode = mp_hands.Hands(max_num_hands=1)
    idx = 0
    data = []
    for i in os.listdir(dataset_path):
        path = os.path.join(train_path, i)
        if not os.path.isdir(path):
            os.mkdir(path)
        path = os.path.join(dataset_path, i)
        for j in os.listdir(path):
            img = cv2.imread(os.path.join(path, j))
            img = cv2.resize(img, (500, 500))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands_mode.process(img)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mpDraw.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS)
                data0 = (results.multi_hand_landmarks[0].landmark[0].x + results.multi_hand_landmarks[0].landmark[
                    5].x) / 2, (results.multi_hand_landmarks[0].landmark[0].y +
                                results.multi_hand_landmarks[0].landmark[5].y) / 2
                data.append((np.array(
                    [[i.x - data0[0], i.y - data0[1], i.z] for i in results.multi_hand_landmarks[0].landmark], np.float32), int(i)))
                shutil.copy(os.path.join(path, j),
                            os.path.join(os.path.join(train_path, i), str(idx) + '.' + j.split('.')[1]))
                idx += 1
    np.save(os.path.join(data_path, 'data.npy'), data)
if not os.path.isdir(test_path):
    os.mkdir(test_path)
    test_list = moveFile(train_path)

if not test_list:
    for i in os.listdir(test_path):
        for j in os.listdir(os.path.join(test_path, i)):
            test_list.append(int(j.split('.')[0]))

data = np.load("./data/data.npy", allow_pickle=True)


class SLDDDataset(Dataset):
    def __init__(self, root):
        self.features = []
        self.labels = []
        if root == "train":
            for i in range(len(data)):
                if i not in test_list:
                    self.features.append(data[i][0])
                    self.labels.append(data[i][1])
        elif root == "test":
            for i in range(len(data)):
                if i in test_list:
                    self.features.append(data[i][0])
                    self.labels.append(data[i][1])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]