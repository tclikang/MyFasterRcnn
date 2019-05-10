import torch
import torchvision.models
import torch.nn
import dataset as ds
import torch.nn.functional as f
from parameters import Parameters
import utils
import numpy as np
import visdom


class faster_rcc_net(torch.nn.Module):
    def __init__(self):
        super(faster_rcc_net, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=False)
        # vgg的backbone的pooling是要修改maxpooling2d模型
        self.vgg_backbone = vgg16.features[:30]  # 不要最后的pooling层,修改Maxpool2d层
        self.vgg_backbone[4]  = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.vgg_backbone[9]  = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.vgg_backbone[16] = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.vgg_backbone[23] = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        # ----------上面修改了vgg模型
        self.vgg_cls = vgg16.classifier
        self.rpn_3x3 = torch.nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3,stride=1,padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.rpn_1x1_cls = torch.nn.Conv2d(512, 18, kernel_size=1, stride=1)
        self.rpn_1x1_reg = torch.nn.Conv2d(512, 36, kernel_size=1, stride=1)

    def forward(self, img):
        out = self.vgg_backbone(img)
        out = self.relu(self.rpn_3x3(out))
        self.rpn_cls = self.rpn_1x1_cls(out)
        self.rpn_reg = self.rpn_1x1_reg(out)
        # rpn_cls shape is 1 18 37 50  是rpn出来的分数,未进行softmax
        # rpn_reg shape is 1 36 37 50  是rpn出来的分数
        return self.rpn_cls, self.rpn_reg

    # anchor_labels 是一个17100的向量,其中如果该anchor与gt的iou>0.7值就是1
    # 若0.3<iou<0.7则是-1,若iou<0.3则是-1
    # rpn_cls每次选出256个样本训练,其中正样本全部用来训练,负样本负责填充
    # 暂时弃用
    def compute_rpn_cls_loss_test(self, anchor_labels):
        train_index = utils.get_rpn_train_cls_sample_index(anchor_labels)  # 用于训练的样本索引
        batch_size, _, feature_map_h, feature_map_w = self.rpn_cls.shape
        rpn_cls_permute = self.rpn_cls.permute(0, 2, 3, 1).contiguous()  # 1 38 50 18
        rpn_softmax_scores = f.softmax(rpn_cls_permute.view(batch_size, feature_map_h, feature_map_w,
                                                    Parameters.anchors_num, 2), dim=4) # 1 38 50 9 2
        # rpn_softmax_scores_fg = rpn_softmax_scores[:,:,:,:,1].contiguous()  # 1 38 50 9
        rpn_softmax_scores_reshape = rpn_softmax_scores.view(-1,2)
        rpn_softmax_scores_train = rpn_softmax_scores_reshape[train_index]
        rpn_cls_loss = f.cross_entropy(rpn_softmax_scores_train.cuda(),
                                       torch.from_numpy(anchor_labels[train_index]).long().cuda())
        return rpn_cls_loss

    # anchor_labels 是一个17100的向量,其中如果该anchor与gt的iou>0.7值就是1
    # 若0.3<iou<0.7则是-1,若iou<0.3则是-1
    # rpn_cls每次选出256个样本训练,其中正样本全部用来训练,负样本负责填充
    def compute_rpn_cls_loss(self, anchor_labels):
        train_index = utils.get_rpn_train_cls_sample_index(anchor_labels)  # 用于训练的样本索引
        cls_sample = utils.get_data_from_rpn_featuremap_according_to_index(self.rpn_cls, train_index)
        rpn_cls_loss = f.cross_entropy(cls_sample.cuda(),
                                       torch.from_numpy(anchor_labels[train_index]).long().cuda())

        return rpn_cls_loss

    # anchor_labels是样本的标签17100个,其中正样本是1,负样本是0,无用的是-1
    # anchor_to_gt是正样本对应的人工标记的样本框,也就是说当anchor_labels中的正样本标签是1的时候,
    # anchor_to_gt里面的数值是gt
    def compute_rpn_reg_loss(self,anchor_labels, anchor_to_gt, anchors):
        train_index = utils.get_rpn_train_reg_sample_index(anchor_labels)
        if train_index.shape[0] == 0:
            return 0  # 返回loss为0
        # 从[1 36 38 50]中取出用于训练的数据
        # reg_sample是网络输出的分数
        reg_sample = utils.get_data_from_rpn_featuremap_according_to_index(self.rpn_reg, train_index)
        reg_gt_sample = anchor_to_gt[train_index]  # 对应的gt
        reg_anchors = anchors[train_index]  # 对应的anchor的位置
        reg_target = utils.get_tx_ty_tw_th_list_from_two_bblist(reg_gt_sample, reg_anchors)
        reg_target = torch.from_numpy(reg_target).cuda()
        reg_loss = f.smooth_l1_loss(reg_sample.cuda(), reg_target.float())
        return reg_loss


if __name__ == '__main__':
    viz = visdom.Visdom()
    model = faster_rcc_net().cuda()
    anchors = utils.gen_anchor_box()
    # print(model.vgg_backbone)
    # print(model.vgg_cls)
    dataset = ds.my_dataset()
    optimizer = torch.optim.SGD(model.parameters(),
                                0.01,
                                0.9,
                                5e-4)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=1,
                                              shuffle=True)
    for img, class_name, anno in data_loader:
        img = img.cuda()
        anno_numpy_2 = anno.squeeze(0).numpy()
        assert len(anno_numpy_2.shape) is 2
        anchor_labels, anchor_to_gt = utils.gen_label_IoU_from_anchor(anchors, anno_numpy_2)
        # print(len(np.where(anchor_labels == 1)[0]))
        out_cls, out_reg = model(img)  # 1 18 37 50
        # rpn_cls_loss_test = model.compute_rpn_cls_loss_test(anchor_labels)
        rpn_cls_loss = model.compute_rpn_cls_loss(anchor_labels)
        rpn_reg_loss = model.compute_rpn_reg_loss(anchor_labels, anchor_to_gt, anchors)
        rpn_total_loss = rpn_cls_loss + rpn_reg_loss * 10.0
        optimizer.zero_grad()
        rpn_total_loss.backward()
        optimizer.step()
        print(rpn_cls_loss)


















