import numpy as np

class Parameters:
    image_path =      '/home/fanfu/data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages'
    annotation_path = '/home/fanfu/data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations'
    resize_img_w = 800  # 每一张图片缩放到什么宽度
    resize_img_h = 600  # 每一张图片缩放到什么高度
    anchors_num = 9  # 锚点的个数
    anchor_per_point_at_00 = np.array([[-84., -40., 99., 55.],
                            [-176., -88., 191., 103.],
                            [-360., -184., 375., 199.],
                            [-56., -56., 71., 71.],
                            [-120., -120., 135., 135.],
                            [-248., -248., 263., 263.],
                            [-36., -80., 51., 95.],
                            [-80., -168., 95., 183.],
                            [-168., -344., 183., 359.]]) - 7.5
    train_rpn_proposal_num_to_fastrcnn = 6000  # 先用于预选择的有6000个
    train_rpn_proposal_num_to_fastrcnn_nms_thres = 0.5 # 最后fast rcnn分类的时候anchor的重合度大于0.5的分为正样本
    proposal_nms_thres = 0.5  # nms的阈值
    train_rpn_proposal_num_to_fastrcnn_after_nms = 128  # 从6000个中根据规则提取128个样本用于训练,其中32个正样本和96个负样本
    roi_pooling_size = 7
    anchor_start_point = np.array([7.5, 7.5])  # 因为下标从0开始,所以起始坐标是7.5 7.5
    anchor_step = 16
    rpn_total_cls_sample_num = 256  # 总共256个sample用于训练
    clsname_to_label = {'person': 0,
                        'bird': 1,
                        'cat': 2,
                        'cow': 3,
                        'dog': 4,
                        'horse': 5,
                        'sheep': 6,
                        'aeroplane': 7,
                        'bicycle': 8,
                        'boat': 9,
                        'bus': 10,
                        'car': 11,
                        'motorbike': 12,
                        'train': 13,
                        'bottle': 14,
                        'chair': 15,
                        'diningtable': 16,
                        'pottedplant': 17,
                        'sofa': 18,
                        'tvmonitor': 19,
                        'background': 20}
    def __init__(self):
        pass










