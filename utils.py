# 使用一些图像处理的函数,以及读取文件的各种函数
import torch
import cv2
import os
import xml.etree.ElementTree as ET
import parameters
import numpy as np
import torch.nn.functional as f
import random
from torchviz import make_dot, make_dot_from_trace

# -----------------------------------文件处理相关--------------------------------



# 将文件夹中所有的文件根据文件名排序,并将文件名放入name_list中
def name_list_in_a_dir(dir_path):
    name_list = os.listdir(dir_path)
    name_list.sort()
    name_list = [os.path.join(dir_path, name) for name in name_list]
    return name_list

def read_anno_in_xml_file(anno_path):
    xml_tree = ET.parse(anno_path)  # 获得一棵树
    xml_root = xml_tree.getroot()  # 得到树根
    img_size = xml_root.find('size')
    img_width   = int(img_size.find('width').text)
    img_height  = int(img_size.find('height').text)
    img_channel = int(img_size.find('depth').text)
    img_size_para = [img_height, img_width, img_channel]  # (高度,宽度,深度)
    cls_name = []
    bb = []
    for object in xml_root.iter("object"):
        temp_cls_name = object.find("name").text
        bnd_root = object.find('bndbox')
        xmin = int(bnd_root.find('xmin').text)
        ymin = int(bnd_root.find('ymin').text)
        xmax = int(bnd_root.find('xmax').text)
        ymax = int(bnd_root.find('ymax').text)
        temp_bb = [xmin, ymin, xmax, ymax]
        cls_id = parameters.Parameters.clsname_to_label[temp_cls_name]
        cls_name.append(cls_id)
        bb.append(temp_bb)
    # 其中img_size_para的格式为(高度,宽度,深度)
    return img_size_para, cls_name, bb


# ----------------------------------图像处理相关-------------------------------
# 所有的图像格式为[高度,宽度,通道数]
# 图像左上角的起点坐标为0 0,右下角结束坐标为width-1, height-1
def read_img(img_path):
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# 将img缩放到dis_h,dis_w的尺寸,并返回缩放比
def img_resize(img, dis_h, dis_w):
    img_h, img_w, _ = img.shape
    img_after_resize = cv2.resize(img, (dis_w, dis_h))
    scale_h = dis_h / img_h
    scale_w = dis_w / img_w
    return img_after_resize, scale_h, scale_w


# bb是一个list,格式为xmin, ymin, xmax, ymax
# 将其缩放到scale_h,scale_w的倍数
# img_h img_w是缩放后的图像尺寸
def anno_resize(bb, scale_h, scale_w, img_h, img_w):
    bb_resize_list = []
    for b in bb:
        temp_xmin = max(int(b[0] * scale_w), 0)
        temp_xmax = min(int(b[2] * scale_w), img_w-1)
        temp_ymin = max(int(b[1] * scale_h), 0)
        temp_ymax = min(int(b[3] * scale_h), img_h-1)
        bb_resize_list.append([temp_xmin, temp_ymin, temp_xmax, temp_ymax])
    return bb_resize_list


# img is read by read_img
# bb is a list of bounding box
def showimg_with_bb(img, bb_list, type='corner',color = (0,0,255)):
    if type is 'xywh':
        for bb in bb_list:
            x, y, w, h = bb
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color)
    if type is 'corner':
        for bb in bb_list:
            xmin, ymin, xmax, ymax = bb
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color)
    showimg(img)


def showimg(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)

# --------------------------------anchor处理相关---------------------------
# 根据参数在原图中生成anchors
def gen_anchor_box():
    img_w = parameters.Parameters.resize_img_w  # 图像宽度
    img_h = parameters.Parameters.resize_img_h  # 图像高度
    start_point_x = parameters.Parameters.anchor_start_point[0]  # anchor的起始中心在原图中的坐标,因为要在conv53层映射回原图
    start_point_y = parameters.Parameters.anchor_start_point[1]  # anchor的起始中心在原图中的坐标
    step = parameters.Parameters.anchor_step
    x_list = [x for x in np.arange(start_point_x, img_w, step)]
    y_list = [y for y in np.arange(start_point_y, img_h, step)]
    anchors = None
    # anchor的生成是先遍历行,然后再遍历列
    for y in y_list:
        for x in x_list:
            temp_step  = np.array([x, y, x, y])
            new_anchor = parameters.Parameters.anchor_per_point_at_00 + temp_step
            if anchors is None:  # 初始化
                anchors = new_anchor
            else:
                anchors = np.vstack((anchors, new_anchor))
    return anchors

# anchors is a [17100 4] np array
# gt is a [n 4] np array
# the type is [xmin ymin xmax ymax]
#
def gen_label_IoU_from_anchor(anchors, gt):
    anchors_num = anchors.shape[0]
    anchors_label = np.zeros((anchors_num,))
    anchors_to_gt = (-1.0)*np.ones((anchors_num, 4))  # 当对应的label是正样本的时候,将其对应的gt放入
    gt_num = gt.shape[0]
    for i in range(anchors_num):
        bbx_anchor = anchors[i, :]
        if is_valid_bbx(bbx_anchor) is False:  # 不合法
            anchors_label[i] = -1  # 不合法的都是-1
            continue
        max_iou = 0
        for j in range(gt_num):
            iou = compute_iou(anchors[i, :], gt[j, :])
            if iou > max_iou:
                max_iou = iou
                max_iou_gt = gt[j, :]


        # 只要>0则说明有重叠度,此时就要有对应的gt
        if max_iou > 0:
            anchors_to_gt[i, :] = max_iou_gt

        if max_iou > 0.7:  # 大于0.7的说明其中有目标,label为1
            anchors_label[i] = 1
        elif max_iou < 0.3:
            anchors_label[i] = 0  # 小于0.3为0 说明是背景
        else:
            anchors_label[i] = -1  # 小于0.7 大于 0.3的是无用的,也置为-1
    return anchors_label, anchors_to_gt


# 从anchor_labels中获得rpn 训练所用的样本的索引
# 这里的anchor_labels中对应的1的是正样本,对应为0的是负样本,对应的-1的是不可用样本
def get_rpn_train_cls_sample_index(anchor_labels):
    train_num = parameters.Parameters.rpn_total_cls_sample_num
    index_pos = np.where(anchor_labels == 1)[0]  # 正样本的全部标签
    if index_pos.shape[0] > train_num:
        index_pos = index_pos[:train_num]
    index_pos_num = index_pos.shape[0]
    # --------------这里暂时使用正负样本数相同
    # index_neg_num = train_num - index_pos_num
    index_neg_num = index_pos_num
    pre_index_neg = np.where(anchor_labels == 0)[0]  # 负样本的全部标签
    index_neg = np.random.choice(pre_index_neg, index_neg_num, replace=False)
    train_index = np.append(index_pos, index_neg)
    return train_index # correct

# 获取用于训练rpn reg的正样本的index,全部的正样本都要训练
def get_rpn_train_reg_sample_index(anchor_labels):
    index_pos = np.where(anchor_labels == 1)[0]  # 正样本的全部标签
    return index_pos

# batch_feature_map是一个[1 18 38 50]或一个[1 36 38 50]的tensor
# index是一个向量,每个元素对应要提取的数据的index
# 返回一个[n 2]或者[n 4]的数据,其中n是index中的元素个数,也是要选择的样本数
def get_data_from_rpn_featuremap_according_to_index(batch_feature_map, index):
    n = index.shape[0]
    flag = int(batch_feature_map.shape[1] / parameters.Parameters.anchors_num)
    res_tensor = torch.FloatTensor(n, flag)
    for i in range(n):
        temp_index = index[i]
        row_id, col_id, anchor_index = [int(i) for i in get_row_col_from_featuremap(temp_index)]
        res_tensor[i, :] = batch_feature_map[0, flag*anchor_index:flag*(anchor_index+1), row_id, col_id]
    return res_tensor

# 这里的index是从0开始的
def get_row_col_from_featuremap(index, rows=38, cols=50, anchor_num=9):
    row_index = np.floor(index / (cols * anchor_num))
    col_index = np.floor((index - row_index*(cols * anchor_num))/anchor_num)
    anchor_index = index - row_index*(cols * anchor_num) - col_index*anchor_num
    return row_index, col_index, anchor_index

# batch_feature_map是一个[1 18 38 50]或一个[1 36 38 50]的tensor
# 返回一个[n 2]或者[n 4]的数据,其中n是17100个anchor
def extend_rpn_cls_or_reg_score_into_17100(batch_feature_map, total_anchor_num = 17100):
    flag = int(batch_feature_map.shape[1] / parameters.Parameters.anchors_num)
    # res_tensor = torch.FloatTensor(total_anchor_num, flag)
    # for i in range(total_anchor_num):
    #     row_id, col_id, anchor_index = [int(i) for i in get_row_col_from_featuremap(i)]
    #     temp = batch_feature_map[0, flag * anchor_index:flag * (anchor_index + 1), row_id, col_id]
    #     res_tensor[i] = temp

    res_tensor = batch_feature_map.permute(0, 2, 3, 1).contiguous().view(1, -1, flag).squeeze(0)  # 已验证,相同
    return res_tensor


# 获得anchors的数据信息,参数都是numpy类型的
# anchors是一个17100 4的包围框
# gt是指当前图片中目标的位置,n 4,其中n是物体的个数
# class_name是gt对应的是n个物体的类别
def get_anchors_info(anchors, gt, cls_name):
    n = anchors.shape[0]  # anchor的个数
    assert n == 17100
    # --------物体的个数
    object_num = gt.shape[0]  # 每一行都是一个目标的gt
    # ---------------
    anchors_corresponding_gt = (-1.0)*np.ones_like(anchors)  # 每个锚点对应的gt
    anchors_max_iou = np.zeros((n,))  # 每个锚点对应的最大的iou
    anchors_corresponding_class = 20 * np.ones((n,))  # 每个锚点对应的种类
    for i in range(n):
        temp_anch = anchors[i,:]  # 拿出一个锚点
        #if is_valid_bbx(temp_anch) is not True:
            # anchors_max_iou[i] = -1.0
            # continue
        for j in range(object_num):
            temp_object = gt[j, :]  # 拿出一个gt
            iou = compute_iou(temp_anch, temp_object)  # 计算iou
            if anchors_max_iou[i] < iou:  # 对应锚点的现有值小于iou
                anchors_max_iou[i] = iou  # 更改对应的iou
                anchors_corresponding_class[i] = cls_name[j]  # 更改对应的种类
                anchors_corresponding_gt[i, :] = temp_object  # 更改对应的gt
    anchor_info = {'anchors_location': anchors,
                   'anchors_corresponding_gt': anchors_corresponding_gt,
                   'anchors_max_iou': anchors_max_iou,
                   'anchors_corresponding_class': anchors_corresponding_class}
    return anchor_info

# 根据anchors_info对样本打上rpn 正负样本的标签
# 其中正样本为1, 负样本为0, 不可用样本为-1
# 可以当成正样本的打上标签为1,能选为负样本的打上标签为0
# 既不能当负样本又不能当负样本的打上标签为-1
# 可以为正样本的具有如下规则:
# 与某个gt相交最大的作为正样本
# 与iou>0.7的可以作为正样本
def get_rpn_train_sample_labels(anchors_info, gt, gt_labels):
    anchor_labels = (-1.0) * np.ones_like(anchors_info['anchors_max_iou'])  # 初始化为什么都不能用的样本
    iou_7 = np.where(anchors_info['anchors_max_iou'] >= 0.7)  # 大于0.7的都是正样本
    anchor_labels[iou_7] = 1  # 重合度大于0.7可以作为正样本
    iou_3 = np.where((anchors_info['anchors_max_iou'] <= 0.3) & (anchors_info['anchors_max_iou'] >= 0))  # 小于0.3的都是负样本
    anchor_labels[iou_3] = 0  # 小于0.3可以作为负样本
    # 该张图片中总共有gt_num个目标
    gt_num = gt.shape[0]
    for i in range(gt_num):
        gt_index = np.where(anchors_info['anchors_corresponding_gt'] == gt[i])  # 与当前gt相同种类的bb集合索引
        gt_index = np.unique(gt_index[0])
        if gt_index.shape[0] < 1:  # 没有与这个框相交最大的
            continue
        max_iou_index = np.where(
            anchors_info['anchors_max_iou'][gt_index] == np.max(anchors_info['anchors_max_iou'][gt_index])
        )
        anchor_labels[gt_index[max_iou_index]] = 1
    assert np.where(anchor_labels == 1)[0].shape[0] > 0
    # print('rpn pos samples number is {}'.format(np.where(anchor_labels == 1)[0].shape[0]))
    return anchor_labels

# 对anchor_info中的所有anchor做bbreg
def do_bbreg_for_all_anchors(anchors_info, bbreg_score):
    anchors_location = anchors_info['anchors_location']  # 获得生成的anchor的location
    anchors_location_after_rpn_reg = np.zeros_like(anchors_location)
    n = anchors_location.shape[0]  # 多少个锚点
    assert n == 17100
    assert bbreg_score.shape[0] == n
    for i in range(n):
        anchor = anchors_location[i,:]  # 获取一个锚点的位置
        reg = bbreg_score[i, :]
        reg = np.array([reg[1], reg[0], reg[3], reg[2]])  # 这里是要改回来的呀,这里看一下原版是怎么做的 dx dy dw dh
        # bbreg_from_minmax_to_minmax这个函数是正确的
        anchors_location_after_rpn_reg[i,:] = bbreg_from_minmax_to_minmax(anchor, reg)
    return anchors_location_after_rpn_reg



# proposals 是一个17100 4 的矩阵,每一行都是一个proposal
# scores 表示每一个proposal在rpn分类阶段的分类分数
# num表示proposals中选择多少个送入fast rcnn里面训练
# 返回选出的proposals的index
def nms(proposals, scores, num = 128):
    sort_dec_index = np.argsort(-scores)  # 从大到小排序scores
    index = np.zeros((num,))
    selected_proposal = (-1) * np.ones((num,4))
    step = 0
    for i in sort_dec_index:
        if is_valid_bbx(proposals[i, :]) is not True:  # 不合法的直接跳过
            continue
        if step == 0:  # 合法的并且分数最高的
            index[step] = i
            selected_proposal[step, :] = proposals[i, :]
            step += 1
            continue
        # 下面的函数用来判断当前proposal_need_judge,是否与已经选择的proposals的iou大于0.5
        def judge_proposal(selected_proposal, proposal_need_judge):
            for p in selected_proposal:
                if compute_iou(p, proposal_need_judge) > parameters.Parameters.proposal_nms_thres:
                    return False  # 重合度大于阈值,无用
            return True  # 重合度小于阈值,可用
        if is_valid_bbx(proposals[i, :]) and judge_proposal(selected_proposal, proposals[i, :]):
            selected_proposal[step, :] = proposals[i, :]
            index[step] = i
            step += 1
        if step >= num:
            break
    return index

# rois 是一个n*4的xmin的包围框集合
# gts是当前图中的目标位置,共有
# cls_names是gt对应的类别标签
def get_labels_gt_iou_from_img(rois, gts, cls_names):
    selected_rois_pos = []
    roi_max_iou_pos = []
    roi_to_gt_pos = []  # 默认对应的gt是-1
    roi_to_label_pos = []  # 默认为是背景

    selected_rois_neg = []
    roi_max_iou_neg = []
    roi_to_gt_neg = []  # 默认对应的gt是-1
    roi_to_label_neg = []  # 默认为是背景

    for roi in rois:
        max_iou = 0
        for gt, name in zip(gts, cls_names):
            iou = compute_iou(roi, gt)
            if max_iou < iou:
                max_iou = iou
                max_iou_gt = gt
                max_iou_name = name
        if max_iou > 0.7:  # 正样本
            selected_rois_pos.append(roi)
            roi_max_iou_pos.append(max_iou)
            roi_to_gt_pos.append(max_iou_gt)
            roi_to_label_pos.append(max_iou_name)
        if max_iou < 0.1 and max_iou > 0: # 可以做负样本的
            selected_rois_neg.append(roi)
            roi_max_iou_neg.append(0)
            roi_to_gt_neg.append([-1, -1, -1, -1])
            roi_to_label_neg.append(20)  # 背景的标签

    pos_num = len(selected_rois_pos)  # 正样本的数量

    selected_rois_pos = np.array(selected_rois_pos)
    roi_max_iou_pos = np.array(roi_max_iou_pos)
    roi_to_gt_pos = np.array(roi_to_gt_pos)
    roi_to_label_pos = np.array(roi_to_label_pos)

    selected_rois_neg = np.array(random.sample(selected_rois_neg, pos_num))
    roi_max_iou_neg = np.array(random.sample(roi_max_iou_neg, pos_num))
    roi_to_gt_neg = np.array(random.sample(roi_to_gt_neg, pos_num))
    roi_to_label_neg = np.array(random.sample(roi_to_label_neg, pos_num))

    selected_rois = np.vstack((selected_rois_pos, selected_rois_neg))
    roi_max_iou = np.concatenate((roi_max_iou_pos, roi_max_iou_neg))
    roi_to_gt = np.vstack((roi_to_gt_pos, roi_to_gt_neg))
    roi_to_label = np.concatenate((roi_to_label_pos, roi_to_label_neg))

    # 返回值roi_to_gt对应的roi对应的gt, roi_to_label对应的是种类的标签
    return selected_rois, roi_max_iou, roi_to_gt, roi_to_label

# 在训练阶段选择roi
# 原则是:与gt相交超过0.7的作为正样本,小于0.3的作为负样本
def chose_rpn_proposal_train(anchors_info, gts, cls_names):
    locations = anchors_info['anchors_location_after_rpn_reg']
    # 将合法的pro的index返回的函数
    def valid_index_of_pro(locations):
        valid_index = []
        for index, loc in enumerate(locations):
            #if is_valid_bbx(loc):  # 全部的都合适
            valid_index.append(index)
        return np.array(valid_index)
    valid_index = valid_index_of_pro(locations)
    assert len(valid_index) > 0
    selected_rois, roi_max_iou, roi_gt, labels = get_labels_gt_iou_from_img(locations[valid_index], gts, cls_names)
    return selected_rois, roi_max_iou, roi_gt, labels


# 根据条件选择用于fast rcnn分类用的proposal
# 条件1: 选择分类分数最高的6000个proposal
# 条件2: 这些proposal的感受野必须要在原图内,不能超过边框
# 输入的是17100个anchor的信息
# 这个函数的主要目的有以下几个:
# 与gt相交超过0.7的作为正样本,有几个选几个
# 与gt相交小于0.3的作为负样本,有几个正样本就选几个负样本
# 训练阶段
def chose_rpn_proposal(anchors_info, gts, cls_names):
    rpn_cls_foreground_score = anchors_info['rpn_cls_foreground_score']
    proposals_index = nms(anchors_info['anchors_location_after_rpn_reg'], rpn_cls_foreground_score).astype(int)
    locations = anchors_info['anchors_location_after_rpn_reg'][proposals_index]  # 有128 4矩阵,每行一个roi
    roi_gt, labels = get_labels_gt_from_img(locations, gts, cls_names)
    return locations, gts, labels
    # sort_dec_index = np.argsort(-rpn_cls_foreground_score)  # 从大到小排序的索引
    # fastrcnn_train_sample_num = parameters.Parameters.train_rpn_proposal_num_to_fastrcnn_after_nms
    # anchor_proposal = (-1)*np.ones((fastrcnn_train_sample_num,4))  # 选出的样本的anchor的位置在这里
    # train_proposal_gt = (-1)*np.ones((fastrcnn_train_sample_num,4))  # 选出的样本的anchor对应的gt在这里
    # train_proposal_labels = (-1)*np.ones((fastrcnn_train_sample_num,))  # 选出的样本的对应的label在这里
    # train_proposal_reg = (-1)*np.ones((fastrcnn_train_sample_num,4))  # 选出的
    # anchor_proposal_num = 0
    # for (anchor, gt_index) in zip(anchors_info['anchors_location_after_rpn_reg'][sort_dec_index], sort_dec_index):
    #     # 如果超过了6000个就不给proposal了
    #     if anchor_proposal_num >= parameters.Parameters.train_rpn_proposal_num_to_fastrcnn_after_nms:
    #         break
    #     if is_valid_bbx(anchor):  # 这有问题?为什么6000个都没有?
    #         anchor_proposal[anchor_proposal_num, :]   = anchor
    #         train_proposal_gt[anchor_proposal_num, :] = anchors_info['anchors_corresponding_gt'][gt_index]
    #         train_proposal_labels[anchor_proposal_num] = anchors_info['anchors_corresponding_class'][gt_index]
    #         anchor_proposal_num += 1
    #
    # anchors_info['train_proposal_into_fastrcnn_location'] = anchor_proposal
    # anchors_info['train_proposal_into_fastrcnn_gt'] = train_proposal_gt
    # anchors_info['train_proposal_into_fastrcnn_labels'] = train_proposal_labels
    # anchors_info['train_proposal_into_fastrcnn_reg_score'] = get_tx_ty_tw_th_list_from_two_bblist(train_proposal_gt, anchor_proposal)
    # return anchors_info

# 根据anchors info中的信息计算roipooling的值
# anchors_info['train_proposal_into_fastrcnn'] 保存了128个 proposal,xmin格式的
# featuremap_conv53是从backbone中提取的feature map,是一个tensor
def compute_roi_pooling_from_rpn_proposal(roi_loc, featuremap_conv53):
    # 将proposal提取出来
    train_proposal_into_fastrcnn = roi_loc
    roipooling_feature77 = None
    for proposal in train_proposal_into_fastrcnn:
        xmin_fm, ymin_fm, xmax_fm, ymax_fm = [np.floor(i).astype(int) for i in
                                              (proposal / parameters.Parameters.anchor_step)]  # proposal在特征图中的坐标
        xmin_fm = max(0, xmin_fm)
        ymin_fm = max(0, ymin_fm)
        xmax_fm = min(49, xmax_fm)
        ymax_fm = min(37, ymax_fm)
        temp_feature_map = featuremap_conv53[:, :, ymin_fm:ymax_fm + 1, xmin_fm:xmax_fm + 1]
        # assert xmin_fm >= 0
        # assert ymin_fm >= 0
        # assert xmax_fm < 50
        # assert ymax_fm < 38
        if roipooling_feature77 is None:
            roipooling_feature77 = f.adaptive_max_pool2d(temp_feature_map, parameters.Parameters.roi_pooling_size)
        else:
            assert temp_feature_map.shape[2] > 0
            roipooling_feature77 = torch.cat(
                (roipooling_feature77, f.adaptive_max_pool2d(temp_feature_map, parameters.Parameters.roi_pooling_size)),
                dim=0)
    return roipooling_feature77

# 每一个样本对应一个标签
def gen_fastrcnn_cls_label(anchors_info):

    pass


# 根据类别将fastrcnn最后的reg分类分数提取出来计算loss
def gen_fastrcnn_reg_score_to_compute_reg_loss(fastrcnn_reg_net_score, labels):
    '''fastrcnn_reg_net_score 是一个[n 84]的tensor,其中n是proposal的个数,84对应分类
    labels是n维的向量,其中每一个维度都是该proposal对应的标签,其中第20号标签对应的是背景
    '''
    selected_proposal_reg_score = None
    for index, score in enumerate(fastrcnn_reg_net_score):
        l = labels[index].astype(int)  # 种类的标签
        if selected_proposal_reg_score is None:
            selected_proposal_reg_score = score[l*4:(l+1)*4]
            selected_proposal_reg_score = selected_proposal_reg_score.unsqueeze(0)
        else:
            selected_proposal_reg_score = torch.cat((selected_proposal_reg_score,score[l*4:(l+1)*4].unsqueeze(0)) , dim=0)
    return selected_proposal_reg_score



# 对xmin格式的包围框做回归 都是numpy格式
# anchor 是一个xmin,ymin,xmax,ymax的一维向量
# reg_score是一个tx,ty,tw,th的变换
# 返回一个变换后的xmin格式的包围框
def bbreg_from_minmax_to_minmax(anchor, reg_score):
    anchor_xywh = bb_from_minmax_xywh(anchor)
    anchor_xywh_after_reg = bbreg_xywhbb_trans(anchor_xywh, reg_score)
    anchor_minmax = bb_from_xywh_minmax(anchor_xywh_after_reg)
    return anchor_minmax


# bb是一个xywh格式的包围框
# reg_score是一个tx,ty,tw,th的变换
# 返回一个xywh格式的包围框
def bbreg_xywhbb_trans(bb, reg_score):
    x, y, w, h = bb
    dx, dy, dw, dh = reg_score
    res_x = w * dx + x
    res_y = h * dy + y
    res_w = w * np.exp(dw)
    res_h = h * np.exp(dh)
    return np.array([res_x, res_y, res_w, res_h])




# 判断一个bbx是否是合法的
# 不能超出边框才是合法的
def is_valid_bbx(bbx):
    xmin, ymin, xmax, ymax = bbx
    if xmin < 0 or ymin < 0 \
        or xmax > parameters.Parameters.resize_img_w-1 \
        or ymax > parameters.Parameters.resize_img_h-1:
        return False
    else:
        return True




def compute_iou(box1, box2, wh=False):
    """
    compute the iou of two boxes.
    Args:
        box1, box2: [xmin, ymin, xmax, ymax] (wh=False) or [xcenter, ycenter, w, h] (wh=True)
        wh: the format of coordinate.
    Return:
        iou: iou of box1 and box2.
    """
    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
        xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
        xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
        xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)

    ## 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    ## 计算两个矩形框面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1])) # 计算交集面积
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)  # 计算交并比

    return iou


# -----------------------------------包围框回归相关代码----------------
# anchor_array, gt_array是一个二维数组,每一行是一个xmin,ymin,xmax,ymax
#
def get_tx_ty_tw_th_list_from_two_bblist(gt_array, anchor_array):
    rows = gt_array.shape[0]
    txywh_array = np.ones_like(gt_array)
    for i in range(rows):
        temp_gt     = gt_array[i,:]
        temp_anchor = anchor_array[i,:]
        txywh_array[i,:] = get_tx_ty_tw_th_from_two_bb(temp_gt, temp_anchor)
    return txywh_array


# bb1 bb2 是xmin,ymin,xmax,ymax类型的包围框
# 都是numpy类型的数据
def get_tx_ty_tw_th_from_two_bb(gt, anchor):
    gt_xywh = bb_from_minmax_xywh(gt)
    anchor_xywh = bb_from_minmax_xywh(anchor)
    tx = (gt_xywh[0] - anchor_xywh[0]) / anchor_xywh[2]
    ty = (gt_xywh[1] - anchor_xywh[1]) / anchor_xywh[3]
    tw = np.log(gt_xywh[2] / anchor_xywh[2])
    th = np.log(gt_xywh[3] / anchor_xywh[3])
    return np.array([tx, ty, tw, th])

# xywh的xy是中心而不是左上角
def bb_from_minmax_xywh(bb):
    xmin = bb[0]
    ymin = bb[1]
    xmax = bb[2]
    ymax = bb[3]
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2
    w = xmax - xmin + 1
    h = ymax - ymin + 1
    return np.array([x, y, w, h])


# xywh中的xy是中心点,而不是左上角的点
def bb_from_xywh_minmax(bb):
    x = bb[0]
    y = bb[1]
    w = bb[2]
    h = bb[3]
    xmin = x - (w-1)/2
    ymin = y - (h-1)/2
    xmax = xmin + w - 1
    ymax = ymin + h - 1
    return np.array([xmin, ymin, xmax, ymax])



# 讲最新的模型读入net,并删除老模型
def read_net(filepath, net):
    # 读取训练数据
    if len(os.listdir(filepath)) > 0:  # 文件夹不为空
        model_list = os.listdir(filepath)
        model_list.sort()
        model_path = filepath + model_list[-1]
        net.load_state_dict(torch.load(model_path))
        for name in model_list:
            file_name = os.path.join(filepath, name)
            os.remove(file_name)
        torch.save(net.state_dict(), '{}0.pkl'.format(filepath))
        return net

if __name__ == '__main__':
    #anno_path = '/home/fanfu/data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/000005.xml'
    #img_path = '/home/fanfu/data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000005.jpg'
    #_, _, bb = read_anno_in_xml_file(anno_path)
    #img = read_img(img_path)
    #img2, sh, sw = img_resize(img, 800, 2000)
    #print(sh, sw)
    #showimg(img2)
    anchor = np.array([-82.50967 ,-37.254833,   98.50967,     53.254833])  # ymin xmin ymax xmax
    reg = np.array([-0.11372066,0.04183003 , -0.44202712,  -0.1670937])  # y x h w
    res = bbreg_from_minmax_to_minmax(anchor, reg)  #  706.9256, 401.58728, 886.53973, 781.46106
    print(res)
















