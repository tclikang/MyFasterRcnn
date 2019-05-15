import numpy as np
import torch
import torch.utils.data
from parameters import Parameters
import utils
import torchvision.transforms.functional as ttf


class my_dataset(torch.utils.data.dataset.Dataset):
    def __init__(self):
        self.img_path = Parameters.image_path
        self.anno_path = Parameters.annotation_path
        self.img_name_list = utils.name_list_in_a_dir(self.img_path)

        pass

    def __getitem__(self, index):
        # find the img of index, and apply preprocess on the img
        img_path = self.img_name_list[index]  # 图像的文件位置
        anno_path = img_path.replace('jpg', 'xml')
        anno_path = anno_path.replace('JPEGImages', 'Annotations')  # 标记的文件位置
        img = utils.read_img(img_path)
        img_size, cls_name, bb = utils.read_anno_in_xml_file(anno_path)
        # ---------缩放图像----------
        img_resized, sh, sw = utils.img_resize(img, Parameters.resize_img_h, Parameters.resize_img_w)
        bb_resized = utils.anno_resize(bb,sh,sw,Parameters.resize_img_h, Parameters.resize_img_w)
        # utils.showimg_with_bb(img_resized,bb_resized)
        # img_resized = ttf.to_tensor(img_resized)*255 # np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)  # 将图像放入tensor,归一化到0-1,n c h w
        img_resized = (ttf.to_tensor(img_resized) * 255).float() - torch.from_numpy(
            np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)).float()
        cls_name = np.array(cls_name)
        bb_resized = np.array(bb_resized)
        # 返回一个600 800的图片,返回类别的名字,返回对应bb的位置
        return img_resized, cls_name, bb_resized

    def __len__(self):
        return len(self.img_name_list)

if __name__ == '__main__':
    dataset = my_dataset()
    anchors = utils.gen_anchor_box()  # 生成17100个anchor
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=1,
                                              shuffle=True)
    for img, class_name, anno in data_loader:
        anno_numpy_2 = anno.squeeze(0).numpy()
        assert len(anno_numpy_2.shape) is 2
        anchor_labels = utils.gen_label_IoU_from_anchor(anchors, anno_numpy_2)
        print(anchor_labels[:200])
        print(class_name)
        print(anno)











