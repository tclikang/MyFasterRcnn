# 使用一些图像处理的函数,以及读取文件的各种函数
import torch
import cv2
import os
import xml.etree.ElementTree as ET
import parameters
import numpy as np

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
    img_w = parameters.Parameters.resize_img_w
    img_h = parameters.Parameters.resize_img_h
    start_point_x = parameters.Parameters.anchor_start_point[0]
    start_point_y = parameters.Parameters.anchor_start_point[1]
    step = parameters.Parameters.anchor_step
    x_list = [x for x in range(start_point_x, img_w, step)]
    y_list = [y for y in range(start_point_y, img_h, step)]
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
def gen_label_IoU_from_anchor(anchors, gt):
    anchors_num = anchors.shape[0]
    anchors_label = np.zeros((anchors_num,))
    anchors_to_gt = np.zeros((anchors_num, 4))  # 当对应的label是正样本的时候,将其对应的gt放入
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
        if max_iou > 0.7:  # 大于0.7的说明其中有目标,label为1
            anchors_label[i] = 1
            anchors_to_gt[i, :] = max_iou_gt
        elif max_iou < 0.3:
            anchors_label[i] = 0  # 小于0.3为0 说明是背景
        else:
            anchors_label[i] = -1  # 小于0.7 大于 0.3的是无用的,也置为-1
    return anchors_label, anchors_to_gt


def get_rpn_train_cls_sample_index(anchor_labels):
    train_num = parameters.Parameters.rpn_total_cls_sample_num
    index_pos = np.where(anchor_labels == 1)[0]  # 正样本的全部标签
    if index_pos.shape[0] > train_num:
        index_pos = index_pos[:train_num]
    index_pos_num = index_pos.shape[0]
    index_neg_num = train_num - index_pos_num
    pre_index_neg = np.where(anchor_labels == 0)[0]  # 负样本的全部标签
    index_neg = np.random.choice(pre_index_neg, index_neg_num, replace=False)
    train_index = np.append(index_pos, index_neg)
    return train_index

# 获取用于训练rpn reg的正样本的index,全部的正样本都要训练
def get_rpn_train_reg_sample_index(anchor_labels):
    index_pos = np.where(anchor_labels == 1)[0]  # 正样本的全部标签
    return index_pos

# batch_feature_map是一个[1 18 38 50]或一个[1 36 38 50]的tensor
# index是一个向量,每个元素对应要提取的数据的index
# 返回一个[n 2]或者[n 4]的数据,其中n是index中的元素个数,也是要选择的样本数
def get_data_from_rpn_featuremap_according_to_index(batch_feature_map, index):
    n = index.shape[0]
    flag = int(batch_feature_map.shape[1] / 9)
    res_tensor = torch.FloatTensor(n, flag)
    for i in range(n):
        temp_index = index[i]
        row_id, col_id, anchor_index = [int(i) for i in get_row_col_from_featuremap(temp_index)]
        res_tensor[i, :] = batch_feature_map[0,flag*anchor_index:flag*(anchor_index+1), row_id, col_id]
    return res_tensor


# 这里的index是从0开始的
def get_row_col_from_featuremap(index, rows=38, cols=50, anchor_num=9):
    row_index = np.floor(index / (cols * anchor_num))
    col_index = np.floor((index - row_index*(cols * anchor_num))/anchor_num)
    anchor_index = index - row_index*(cols * anchor_num) - col_index*anchor_num
    return row_index, col_index, anchor_index



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

def bb_from_minmax_xywh(bb):
    xmin = bb[0]
    ymin = bb[1]
    xmax = bb[2]
    ymax = bb[3]
    x = xmin
    y = ymin
    w = xmax - x + 1
    h = ymax - y + 1
    return np.array([x, y, w, h])





if __name__ == '__main__':
    #anno_path = '/home/fanfu/data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/000005.xml'
    #img_path = '/home/fanfu/data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000005.jpg'
    #_, _, bb = read_anno_in_xml_file(anno_path)
    #img = read_img(img_path)
    #img2, sh, sw = img_resize(img, 800, 2000)
    #print(sh, sw)
    #showimg(img2)

    r, c, a = get_row_col_from_featuremap(9999)
    print(r,c,a)













