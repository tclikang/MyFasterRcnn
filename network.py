import torch
import torchvision.models
import torch.nn
import dataset as ds
import torch.nn.functional as f
from parameters import Parameters
import utils
import numpy as np
import visdom
import tools_visual
import os
from torchviz import make_dot, make_dot_from_trace

class faster_rcc_net(torch.nn.Module):
    def __init__(self, init_best_model=False, model_path='/home/fanfu/PycharmProjects/MyFasterRcnn/best_model/bestmodel.pth'):
        super(faster_rcc_net, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        # vgg的backbone的pooling是要修改maxpooling2d模型
        self.vgg_backbone = vgg16.features[:30]  # 不要最后的pooling层,修改Maxpool2d层
        self.vgg_backbone[4]  = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.vgg_backbone[9]  = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.vgg_backbone[16] = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.vgg_backbone[23] = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        # ----------上面修改了vgg模型
        self.vgg_cls = vgg16.classifier[0:5]
        self.rpn_3x3 = torch.nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3,stride=1,padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.rpn_1x1_cls = torch.nn.Conv2d(512, 18, kernel_size=1, stride=1)
        self.rpn_1x1_reg = torch.nn.Conv2d(512, 36, kernel_size=1, stride=1)
        # fast rcnn network
        self.fastrcnn_cls = torch.nn.Linear(in_features=4096, out_features=21)
        self.fastrcnn_reg = torch.nn.Linear(in_features=4096, out_features=84)
        # load别人的最好的model
        if init_best_model is True:
            net = torch.load(model_path)
            self.vgg_backbone[0].weight.data = net['extractor.0.weight']
            self.vgg_backbone[0].bias.data = net['extractor.0.bias']
            self.vgg_backbone[2].weight.data = net['extractor.2.weight']
            self.vgg_backbone[2].bias.data = net['extractor.2.bias']
            self.vgg_backbone[5].weight.data = net['extractor.5.weight']
            self.vgg_backbone[5].bias.data = net['extractor.5.bias']
            self.vgg_backbone[7].weight.data = net['extractor.7.weight']
            self.vgg_backbone[7].bias.data = net['extractor.7.bias']
            self.vgg_backbone[10].weight.data = net['extractor.10.weight']
            self.vgg_backbone[10].bias.data = net['extractor.10.bias']
            self.vgg_backbone[12].weight.data = net['extractor.12.weight']
            self.vgg_backbone[12].bias.data = net['extractor.12.bias']
            self.vgg_backbone[14].weight.data = net['extractor.14.weight']
            self.vgg_backbone[14].bias.data = net['extractor.14.bias']
            self.vgg_backbone[17].weight.data = net['extractor.17.weight']
            self.vgg_backbone[17].bias.data = net['extractor.17.bias']
            self.vgg_backbone[19].weight.data = net['extractor.19.weight']
            self.vgg_backbone[19].bias.data = net['extractor.19.bias']
            self.vgg_backbone[21].weight.data = net['extractor.21.weight']
            self.vgg_backbone[21].bias.data = net['extractor.21.bias']
            self.vgg_backbone[24].weight.data = net['extractor.24.weight']
            self.vgg_backbone[24].bias.data = net['extractor.24.bias']
            self.vgg_backbone[26].weight.data = net['extractor.26.weight']
            self.vgg_backbone[26].bias.data = net['extractor.26.bias']
            self.vgg_backbone[28].weight.data = net['extractor.28.weight']
            self.vgg_backbone[28].bias.data = net['extractor.28.bias']
            self.fastrcnn_reg.weight.data = net['head.cls_loc.weight']
            self.fastrcnn_reg.bias.data = net['head.cls_loc.bias']
            self.vgg_cls[0].weight.data = net['head.classifier.0.weight']
            self.vgg_cls[0].bias.data = net['head.classifier.0.bias']
            self.vgg_cls[3].weight.data = net['head.classifier.2.weight']
            self.vgg_cls[3].bias.data = net['head.classifier.2.bias']
            self.fastrcnn_cls.weight.data = net['head.score.weight']
            self.fastrcnn_cls.bias.data = net['head.score.bias']
            self.rpn_3x3.weight.data = net['rpn.conv1.weight']
            self.rpn_3x3.bias.data = net['rpn.conv1.bias']
            self.rpn_1x1_reg.weight.data = net['rpn.loc.weight']
            self.rpn_1x1_reg.bias.data = net['rpn.loc.bias']
            self.rpn_1x1_cls.weight.data = net['rpn.score.weight']
            self.rpn_1x1_cls.bias.data = net['rpn.score.bias']

    # cls_name图片中gt的种类, gts是图中的多个gt
    def forward(self, img, anchors_info, cls_name=None, gts=None):
        self.backbone_featuremap = self.vgg_backbone(img)
        out = self.relu(self.rpn_3x3(self.backbone_featuremap))
        rpn_cls = self.rpn_1x1_cls(out)
        rpn_reg = self.rpn_1x1_reg(out)
        # rpn_cls shape is 1 18 37 50  是rpn出来的分数,未进行softmax
        # rpn_reg shape is 1 36 37 50  是rpn出来的分数
        self.rpn_cls = utils.extend_rpn_cls_or_reg_score_into_17100(rpn_cls).cuda()
        self.rpn_reg = utils.extend_rpn_cls_or_reg_score_into_17100(rpn_reg).cuda()

        # ---- 计算一些anchor_info的东西
        # rpn_cls_loss_test = model.compute_rpn_cls_loss_test(anchor_labels)
        rpn_cls_loss, train_rpn_cls_loss_index = model.compute_rpn_cls_loss(anchors_info)
        rpn_reg_loss = model.compute_rpn_reg_loss(anchors_info)
        print('rpn_cls_loss : {}, rpn_reg_loss: {}'.format(rpn_cls_loss, rpn_reg_loss))

        # -------------rpn部分到上面已经结束了
        # rpn网络结束后将所有anchors做bbreg结果存入:anchors_info['anchors_location_after_rpn_reg']中
        anchors_info['anchors_location_after_rpn_reg'] = utils.do_bbreg_for_all_anchors(anchors_info, model.rpn_reg.cpu().detach().numpy())
        # 讲anchor中属于前景的anchor的概率计算出来,存入anchors_info['rpn_cls_foreground_score']中
        anchors_info['rpn_cls_foreground_score'] = (f.softmax(self.rpn_cls, dim=1)[:, 1]).cpu().detach().numpy()

        # -----------------下面显示rpn前景得分最高的框-------------------
        vis.close()
        # 显示选择出来的rpn样本
        img_show = img.cpu().numpy()
        # ---------------显示最大得分的框
        img_show_index = np.argmax(anchors_info['rpn_cls_foreground_score'])
        img_show_rpn_max_box = [anchors_info['anchors_location'][img_show_index].astype(int)]
        img_show_max_rpn_cls = tools_visual.show_torch_img_with_bblist(img_show, img_show_rpn_max_box)
        #vis.images(img_show_max_rpn_cls, opts=dict(title='origin image', caption='How random.'))
        #----------------显示gts
        img_show_gts = tools_visual.show_torch_img_with_bblist(img_show, gts)
        #vis.images(img_show_gts, opts=dict(title='origin image', caption='How random.'))
        #----------------显示rpns
        img_show_rpns_index = np.where(anchors_info['anchor_labels'] == 1)
        img_show_rpns_boxs = anchors_info['anchors_location'][img_show_rpns_index]
        img_show_rpns = tools_visual.show_torch_img_with_bblist(img_show,img_show_rpns_boxs)

        vis_show_img = np.vstack((img_show_max_rpn_cls,img_show_gts,img_show_rpns))
        vis.images(vis_show_img, opts=dict(title='origin image', caption='How random.'))  # rpn选择正负样本是没有错的
        g = make_dot(rpn_cls_loss)
        # -----------------用于画出得分最高的包围框---------------------






        # 根据anchors_info获取roi的相关信息
        roi = dict()
        roi['location'], roi['max_iou'], roi['gts'], roi['labels'] = utils.chose_rpn_proposal_train(anchors_info,gts,cls_name)


        # 这些roi的特征,是1个4维向量, n 512 7 7,其中n是roi的个数
        #if roi['location'].shape[1] == 0: # 如果没有好的样本,就只计算rpn的样本
        return rpn_cls_loss # + (10.0*rpn_reg_loss)
        roi['feature_77'] = utils.compute_roi_pooling_from_rpn_proposal(roi['location'], model.backbone_featuremap)
        # 计算roi所应该移动的距离
        roi['reg_target'] = utils.get_tx_ty_tw_th_list_from_two_bblist(roi['gts'], roi['location'])
        node_feature_map = self.vgg_cls(
            roi['feature_77'].view(roi['feature_77'].shape[0], -1))
        fastrcnn_cls = self.fastrcnn_cls(node_feature_map)
        fastrcnn_reg = self.fastrcnn_reg(node_feature_map)
        fastrcnn_cls_loss = self.compute_fastrcnn_cls_loss(fastrcnn_cls, roi['labels'])
        fastrcnn_reg_loss = self.compute_fastrcnn_reg_loss(fastrcnn_reg, roi['reg_target'], roi['labels'])

        print('fastrcnn_cls_loss : {}, fastrcnn_reg_loss: {}'.format(fastrcnn_cls_loss, fastrcnn_reg_loss))
        total_loss = 0.1*rpn_cls_loss + rpn_reg_loss + \
                     0.1*fastrcnn_cls_loss + fastrcnn_reg_loss
        return total_loss



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
    def compute_rpn_cls_loss(self, anchors_info):
        anchor_labels = anchors_info['anchor_labels']
        train_index = utils.get_rpn_train_cls_sample_index(anchor_labels)  # 用于训练的样本索引
        cls_sample = self.rpn_cls[train_index]
        rpn_cls_loss = f.cross_entropy(cls_sample.cuda(),
                                       torch.from_numpy(anchor_labels[train_index]).long().cuda())
        return rpn_cls_loss, train_index

    # 计算17100个anchor属于rpn前景的概率
    def compute_softmax_score_of_rpn_cls_object(self):
        # 这里可能有问题??------------ 因为目标的标签是1,背景的标签是0,所以后面是1
        self.rpn_cls_softmax_object_prob = f.softmax(self.rpn_cls, dim=1)[:, 1]


    # anchor_labels是样本的标签17100个,其中正样本是1,负样本是0,无用的是-1
    # anchor_to_gt是正样本对应的人工标记的样本框,也就是说当anchor_labels中的正样本标签是1的时候,
    # anchor_to_gt里面的数值是gt
    def compute_rpn_reg_loss(self, anchors_info):
        anchor_labels = anchors_info['anchor_labels']
        train_index = utils.get_rpn_train_reg_sample_index(anchor_labels)
        # 从[1 36 38 50]中取出用于训练的数据
        # reg_sample是网络输出的分数
        reg_sample = self.rpn_reg[train_index]
        reg_gt_sample = anchors_info['anchors_corresponding_gt'][train_index]  # 对应的gt
        reg_anchors = anchors_info['anchors_location'][train_index]  # 对应的anchor的位置
        reg_target = utils.get_tx_ty_tw_th_list_from_two_bblist(reg_gt_sample, reg_anchors)  # 数学公式还没检查
        reg_target = torch.from_numpy(reg_target).cuda()
        reg_loss = f.smooth_l1_loss(reg_sample.cuda(), reg_target.float())
        return reg_loss

    # 计算rcnn的分类的分支loss
    def compute_fastrcnn_cls_loss(self,fast_rcnn_cls, roi_labels):
        fr_cls_labels = torch.from_numpy(roi_labels).long().cuda()
        fastrcnn_cls_loss = f.cross_entropy(fast_rcnn_cls,fr_cls_labels)
        return fastrcnn_cls_loss

    # 计算rcnn回归的分支的loss
    # reg_target 是回归的目标函数
    # labels是标签
    def compute_fastrcnn_reg_loss(self,fast_rcnn_reg, reg_target, labels):
        object_reg_score = torch.from_numpy(reg_target).float().cuda()
        selected_fast_rcnn_reg = utils.gen_fastrcnn_reg_score_to_compute_reg_loss(fast_rcnn_reg, labels)
        reg_loss = f.smooth_l1_loss(selected_fast_rcnn_reg.cuda(), object_reg_score.float())
        return reg_loss





if __name__ == '__main__':
    vis = visdom.Visdom()
    model = faster_rcc_net(init_best_model=True).cuda()
    # setup tracker
    # net_path = '/home/fanfu/PycharmProjects/SimpleSiamFC/pretrained/siamfc_new'
    net_pretrain_path = '/home/fanfu/PycharmProjects/MyFasterRcnn/mymodel/'
    model = utils.read_net(net_pretrain_path, model)

    anchors = utils.gen_anchor_box()  # 生成原图中的anchor,函数没有问题,xmin ymin格式,包含各种不合法anchor
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
    total_epoch = 50
    for epoch in range(total_epoch):
        step = 0
        for img, class_name, anno in data_loader:
            img = img.cuda()  # 提取图像
            class_name_1 = class_name.squeeze(0).numpy()  # 1维的,图像中目标对应的种类
            anno_numpy_2 = anno.squeeze(0).numpy()  # 图像中目标的gt
            anchors_info = utils.get_anchors_info(anchors, anno_numpy_2, class_name_1)
            # 为rpn打上标签,正样本为1,负样本为0,不可用样本为-1,结果存放在anchors_info中
            anchors_info['anchor_labels'] = utils.get_rpn_train_sample_labels(anchors_info, anno_numpy_2, class_name_1)
            # print(np.where(anchor_labels == 1))
            # print(len(np.where(anchor_labels == 1)[0]))
            total_loss = model(img, anchors_info, class_name_1, anno_numpy_2)  # 1 18 37 50
            # 根据这些proposal提取特征做roipooling
            # anchors_info =
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if step % 50 == 0:
                torch.save(model.state_dict(), '{}E{:0>2d}S{:0>10}.pkl'.format(net_pretrain_path, epoch, step))

            step += 1
            print('step is : ', step)




















