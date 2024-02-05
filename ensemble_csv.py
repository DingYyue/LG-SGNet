import argparse
import pickle
import os
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA'},
                        help='the work folder for storing results') # 选择数据集
    parser.add_argument('--alpha',
                        default=1,
                        help='weighted summation',
                        type=float) # 设置默认的加权权重

    parser.add_argument('--joint-dir',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results') # 关节点测试结果
    parser.add_argument('--bone-dir',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results') # 骨架测试结果
    parser.add_argument('--joint-motion-dir', default=None) # 关节运动测试结果
    parser.add_argument('--bone-motion-dir', default=None) # 骨骼运动测试结果

    arg = parser.parse_args() # 解析参数

    dataset = arg.dataset # 获取数据集类型
    if 'UCLA' in arg.dataset:
        label = []
        with open('/21085401076/data/home/sdc1/dy/CTR-GCN-main/CTR-GCN-main/data/' + 'NW-UCLA/' + '/val_label.pkl', 'rb') as f:
            data_info = pickle.load(f)
            for index in range(len(data_info)):
                info = data_info[index]
                label.append(int(info['label']) - 1)
    elif 'ntu120' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('/21085401076/data/home/sdc1/dy/CTR-GCN-main/CTR-GCN-main/data/' + 'ntu120/' + 'NTU120_CSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in arg.dataset:
            npz_data = np.load('/21085401076/data/home/sdc1/dy/CTR-GCN-main/CTR-GCN-main/data/' + 'ntu120/' + 'NTU120_CSet.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('/21085401076/data/home/sdc1/dy/CTR-GCN-main/CTR-GCN-main/data/' + 'ntu/' + 'NTU60_CS.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xview' in arg.dataset:
            npz_data = np.load('/21085401076/data/home/sdc1/dy/CTR-GCN-main/CTR-GCN-main/data/' + 'ntu/' + 'NTU60_CV.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    else:
        raise NotImplementedError

    with open(os.path.join(arg.joint_dir, 'epoch1_test_score.pkl'), 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(os.path.join(arg.bone_dir, 'epoch1_test_score.pkl'), 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    if arg.joint_motion_dir is not None:
        with open(os.path.join(arg.joint_motion_dir, 'epoch1_test_score.pkl'), 'rb') as r3:
            r3 = list(pickle.load(r3).items())
    if arg.bone_motion_dir is not None:
        with open(os.path.join(arg.bone_motion_dir, 'epoch1_test_score.pkl'), 'rb') as r4:
            r4 = list(pickle.load(r4).items())

    right_num = total_num = right_num_5 = 0

    if arg.joint_motion_dir is not None and arg.bone_motion_dir is not None:
        arg.alpha = [0.61, 0.78, 0.28, 0.33]
        pre_list = []
        label_list = []
        csv_f = np.zeros((60,60))
        # len(label)代表的是动作的数量
        for i in tqdm(range(len(label))):
            l = label[i] # label[i]表示第i个动作属于的动作类别
            _, r11 = r1[i] # _代表的是r1中测试的第几个动作； r11表示的是该动作属于某个动作的分数大小
            _, r22 = r2[i]
            _, r33 = r3[i]
            _, r44 = r4[i]
            # 将每个流中某个动作属于某个类别的测试分数分别乘以该流所对应的系数，再相加
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3]
            # argsort()返回的是列表中元素从小到大排列时，每个元素在原列表中的位置索引
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r) # 返回最大值在原列表中对应位置的索引
            pre_list.append(r)
            label_list.append(l)
            right_num += int(r == int(l))
            total_num += 1
        # label_list = np.concatenate(label_list)
        # pre_list = np.concatenate(pre_list)
        confusion = confusion_matrix(label_list, pre_list)
        list_diag = np.diag(confusion)
        list_raw_sum = np.sum(confusion, axis=1)
        each_acc = list_diag / list_raw_sum
        with open('class_acc.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(each_acc)
            writer.writerows(confusion)
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
    elif arg.joint_motion_dir is not None and arg.bone_motion_dir is None:
        arg.alpha = [0.6, 0.6, 0.4]
        for i in tqdm(range(len(label))):
            l = label[:, i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2]
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
    else:
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            r = r11 + r22 * arg.alpha
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
