# %%
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from tqdm.notebook import tqdm
import os
from skimage.segmentation import slic, felzenszwalb
from Performance import performance
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
from collections import Counter
from NOD import nod
from Getf import getf
from fun import  c, cosine_similarities,connections,data_no
from Visualize import visualize_epoch


os.environ['CUDA_VISIBLE_DEVICES'] = '0'



path = './Houston/HSI.mat'
data = scio.loadmat(path)
hsi_data = data['HSI']
hsi_data = torch.FloatTensor(hsi_data.astype(float))
hsi_data = data_no(hsi_data)

path1 = './Houston/LiDAR.mat'
data1 = scio.loadmat(path1)
lidar_data = data1['LiDAR']
lidar_data = torch.FloatTensor(lidar_data.astype(float))
lidar_data = data_no(lidar_data)


path2 = './Houston/TSLabel.mat'
test_label = scio.loadmat(path2)
test_label = test_label['TSLabel']

path3 = './Houston/TRLabel.mat'
train_label = scio.loadmat(path3)
train_label = train_label['TRLabel']
combined_labels = train_label + test_label
combined_labels = torch.FloatTensor(combined_labels.astype(int))

lidar_data=lidar_data.unsqueeze(2)
hsi_data=torch.cat((hsi_data, lidar_data), dim=2)

data_width = hsi_data.shape[0]
data_height = hsi_data.shape[1]
channel_num = hsi_data.shape[2]
width2 = (data_width // 2) + (data_width % 2)
width3 = (width2 // 2) + (width2 % 2)
width4 = (width3 // 2) + (width3 % 2)
height2 = (data_height // 2) + (data_height % 2)
height3 = (height2 // 2) + (height2 % 2)
height4 = (height3 // 2) + (height3 % 2)


class_num = len(set(np.array(combined_labels.reshape(-1)))) - 1
print('The number of classes is:', class_num)
# %%


Number_class = Counter(list(np.array(combined_labels.reshape(-1))))
count = np.zeros(class_num + 1)
count[np.array(list(Number_class.keys())).astype(int)] = list(Number_class.values())
h = count
count = count[1:]

train_count = np.ceil(count * 0.1).astype(int)
train_count


classes_index = []
for i in range(class_num + 1):  # with the background
    class_index = np.argwhere(np.array(combined_labels) == i)
    np.random.shuffle(class_index)
    classes_index.append(class_index)


test_count = []
print('The match between train_count and train_class: ', len(train_count) == class_num)

train_index = []
test_index = []
for i in range(class_num):
    train_index.append(classes_index[i + 1][:train_count[i]])
    test_index.append(classes_index[i + 1][-(len(classes_index[i + 1]) - train_count[i]):])
    test_count.append(len(classes_index[i + 1]) - train_count[i])
# Get train and test mask
train_mask = torch.zeros(hsi_data.shape[:2])
test_mask = torch.zeros(hsi_data.shape[:2])
for i in range(class_num):
    train_mask[train_index[i][:, 0], train_index[i][:, 1]] = 1
    test_mask[test_index[i][:, 0], test_index[i][:, 1]] = 1
plt.imshow(train_mask * combined_labels)
plt.show()
plt.imshow(test_mask * combined_labels)
plt.show()

# %%
seg_index = (slic(np.array(lidar_data), n_segments=34, compactness=0.145, max_num_iter=10, channel_axis=None))


seg_index = torch.Tensor(seg_index.copy())
# seg_index = ground_turth
Block_num = len(set(np.array(seg_index.reshape(-1))))
print('Block_num:', Block_num)
plt.imshow(seg_index)
plt.show()

adj_mask = torch.ones(Block_num, Block_num).int().cuda()



class Graph2dConvolution(Module):


    def __init__(
            self,
            in_channels,
            out_channels,
            width,
            height,
            block_num,
            adj_mask=None,
            if_feature_update=True,
            for_classification=False
    ):
        super(Graph2dConvolution, self).__init__()

        self.weight = Parameter(torch.randn(in_channels, out_channels))
        self.We = Parameter(torch.randn(out_channels - 1, out_channels - 1))
        self.w5 = Parameter(torch.randn(out_channels, out_channels))
        self.w6 = Parameter(torch.randn(out_channels, out_channels))
        self.w7 = Parameter(torch.randn(16, 16))
        self.w8 = Parameter(torch.randn(16, 16))
        self.w9 = Parameter(torch.randn(16, 16))
        self.w10 = Parameter(torch.randn(16, 16))
        self.w11 = Parameter(torch.randn(145, 145))
        self.w = Parameter(torch.randn(4, 1))
        self.ww = Parameter(torch.randn(17, 16))
        self.wl = Parameter(torch.randn(block_num - 1, block_num - 1))
        self.bn2 = nn.BatchNorm2d(1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.reset_parameters()
        self.in_features = in_channels
        self.out_features = out_channels
        self.block_num = block_num
        self.if_feature_update = if_feature_update

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.w, a=math.sqrt(5))
        init.kaiming_uniform_(self.w5, a=math.sqrt(5))
        init.kaiming_uniform_(self.w6, a=math.sqrt(5))
        init.kaiming_uniform_(self.w7, a=math.sqrt(5))
        init.kaiming_uniform_(self.w8, a=math.sqrt(5))
        init.kaiming_uniform_(self.w9, a=math.sqrt(5))
        init.kaiming_uniform_(self.w10, a=math.sqrt(5))
        init.kaiming_uniform_(self.w11, a=math.sqrt(5))
        init.kaiming_uniform_(self.ww, a=math.sqrt(5))
        init.kaiming_uniform_(self.We, a=math.sqrt(5))
        init.kaiming_uniform_(self.wl, a=math.sqrt(5))

    def forward(self, input, lidar, index):

        w5 = self.w5
        w6 = self.w6
        input = (input.permute(0, 2, 3, 1)).matmul(self.weight).permute(0, 3, 1, 2)

        if lidar.shape[1] == 4:
            lidar = lidar.permute(0, 2, 3, 1).matmul(self.w).permute(0, 3, 1, 2)
            w5 = self.w9
            w6 = self.w10
        if self.if_feature_update:
            index = nn.UpsamplingNearest2d(size=(lidar.shape[2], lidar.shape[3]))(index.float()).long()
            batch_size = input.shape[0]
            channels = input.shape[1]
            channel_means = torch.mean(input, dim=(2, 3))
            channel_means_ = channel_means.repeat(channels, 1, 1).permute(1, 2, 0)
            channel_sub = (channel_means_ - channel_means.unsqueeze(1)).permute(0, 2, 1)
            channel_sub = getf(channel_sub, )

            fh = input.view(1,input.shape[1],-1)

            coh = cosine_similarities(fh)
            edgh = c(coh)
            channel_sub_ = channel_sub.repeat(channels, 1, 1, 1).permute(1, 2, 0, 3)
            channel_sub_ = (channel_sub_ - channel_sub.unsqueeze(1)).permute(0, 2, 1, 3)

            index_ex = torch.zeros(batch_size, self.block_num, lidar.shape[2], lidar.shape[3]).cuda()
            index_ex = index_ex.scatter_(1, index - 1, 1)
            block_value_sum = torch.sum(index_ex, dim=(2, 3))

            li_ = lidar.repeat(self.block_num, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)

            index_ex = index_ex.unsqueeze(2)
            li_means = torch.sum(index_ex * li_, dim=(2, 3, 4)) / (
                    block_value_sum + (block_value_sum == 0).float())

            li_means_ = li_means.repeat(self.block_num, 1, 1).permute(1, 2, 0)
            li_sub = (li_means_ - li_means.unsqueeze(1)).permute(0, 2, 1)
            li_sub = getf(li_sub)
            edgl = connections(li_means)
            li_sub_ = li_sub.repeat(self.block_num, 1, 1, 1).permute(1, 2, 0, 3)
            li_sub_ = (li_sub_ - li_sub.unsqueeze(1)).permute(0, 2, 1, 3)

            M = (self.We).mm(self.We.T)

            ad = channel_sub_.reshape(batch_size, -1, channels - 1).matmul(M)
            ad = torch.mean(ad * channel_sub_.reshape(batch_size, -1, channels - 1), dim=2) \
                .view(batch_size, channels, channels)
            sig = 0.05
            ad = torch.exp(-1 / (2 * sig ** 2) * ad)
            ad = ad * edgh
            ad = nod(ad)
            lm = (self.wl).mm(self.wl.T)
            la = li_sub_.reshape(batch_size, -1, Block_num - 1).matmul(lm)
            la = (torch.mean(la * li_sub_.reshape(batch_size, -1, Block_num - 1), dim=2) \
                  .view(batch_size, Block_num, Block_num))
            la = la * 1
            si = 0.03
            la = torch.exp(-1 / (2 * si ** 2) * la)
            la = la * edgl
            la = nod(la)

            ad_means = torch.matmul(ad, fh)
            ad_means = torch.matmul(ad_means.permute(0, 2, 1), w5)
            ad_means = torch.matmul(ad, ad_means.permute(0, 2, 1))
            ad_means = torch.matmul(ad_means.permute(0, 2, 1), w6)


            ad_means = ad_means.permute(0, 2, 1).view(1, input.shape[1], input.shape[2], input.shape[3])

            la_means = torch.matmul(la, li_sub)
            la_means = torch.matmul(la_means.permute(0, 2, 1), self.w7)
            la_means = torch.matmul(la, la_means.permute(0, 2, 1))
            la_means = torch.matmul(la_means.permute(0, 2, 1), self.w8)
            la_means = torch.mean(la_means, dim=1)
            index_ex = index_ex.squeeze(2)

            l_features = torch.sum(index_ex * (lidar + la_means.unsqueeze(2).unsqueeze(3)), dim=1)
            l_features = l_features.unsqueeze(0)

            l_features = F.gelu(l_features)+lidar
            l_features = self.bn2(l_features)


            feature1 = ad_means+input
            h_feature = F.gelu(feature1)+input
            h_feature = self.bn(h_feature)

        else:
            h_feature = input
            h_feature = self.bn(h_feature)
            h_feature = F.gelu(h_feature)
            l_features = lidar
            l_features = self.bn2(l_features)
            l_features = F.gelu(l_features)
        return h_feature, l_features

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class SegNet(nn.Module):
    def __init__(self, in_channel, block_num, width, height, class_num, adj_mask=None, if_feature_update=True,
                 scale_layer=1):
        super(SegNet, self).__init__()
        print('2D Graph convolution network are contructing~~~')

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.gcn1 = Graph2dConvolution(in_channel, in_channel, width, height, block_num=block_num, adj_mask=adj_mask,
                                       if_feature_update=if_feature_update)

        self.gcn2 = Graph2dConvolution(in_channel, in_channel, width2, height2, block_num=block_num, adj_mask=adj_mask,
                                       if_feature_update=if_feature_update)

        self.gcn3 = Graph2dConvolution(in_channel, in_channel, width3, height, block_num=block_num, adj_mask=adj_mask,
                                       if_feature_update=if_feature_update)

        self.gcn4 = Graph2dConvolution(in_channel, in_channel, width4, height4, block_num=block_num, adj_mask=adj_mask,
                                       if_feature_update=if_feature_update)

        self.gcn = Graph2dConvolution(int(in_channel * scale_layer), class_num, width, height, block_num=block_num,
                                      adj_mask=adj_mask,
                                      if_feature_update=if_feature_update, for_classification=True)

        self.block_num = block_num
        self.Softmax = nn.Softmax(dim=1)
        self.scale_layer = scale_layer

    def forward(self, hsimg, lidar, seg_index):
        Up = nn.UpsamplingBilinear2d(size=(hsimg.shape[2], hsimg.shape[3]))
        lup = nn.UpsamplingBilinear2d(size=(lidar.shape[2], lidar.shape[3]))

        index = seg_index.long().cuda()

        if self.scale_layer >= 1:
            f1, lf1 = self.gcn1(hsimg, lidar, index)
            f1_ = self.maxpool(f1)
            f1 = Up(f1_)
            lf1_ = self.maxpool(lf1)
            lf1 = lup(lf1_)

        if self.scale_layer >= 2:
            f2, lf2 = self.gcn2(f1_, lf1_, index)
            f2_ = self.maxpool(f2)
            f2 = Up(f2_)
            lf2_ = self.maxpool(lf2)
            lf2 = lup(lf2_)

        if self.scale_layer >= 3:
            f3, lf3 = self.gcn3(f2_, lf2_, index)
            f3_ = self.maxpool(f3)
            f3 = Up(f3_)
            lf3_ = self.maxpool(lf3)
            lf3 = lup(lf3_)

        if self.scale_layer >= 4:
            f4, lf4 = self.gcn4(f3_, lf3_, index)
            f4_ = self.maxpool(f4)
            f4 = Up(f4_)
            lf4_ = self.maxpool(lf4)
            lf4 = lup(lf4_)

        if self.scale_layer == 1:
            final_class, l_fc = self.gcn(f1, lf1, index)
        if self.scale_layer == 2:
            final_class, l_fc = self.gcn(torch.cat((f1, f2), dim=1), torch.cat((lf1, lf2), dim=1), index)
        if self.scale_layer == 3:
            final_class, l_fc = self.gcn(torch.cat((f1, f2, f3), dim=1), torch.cat((lf1, lf2, lf3), dim=1), index)
        if self.scale_layer == 4:
            final_class, l_fc = self.gcn(torch.cat((f1, f2, f3, f4), dim=1), torch.cat((lf1, lf2, lf3, lf4), dim=1),
                                         index)
        return final_class, l_fc

class graphfusion(Module):


    def __init__(
            self,
            in_channels,
            out_channels,
            width,
            height,
            class_num,

    ):
        super(graphfusion, self).__init__()

        self.We = Parameter(torch.randn(out_channels - 1, out_channels - 1))
        self.w = Parameter(torch.randn(4, 1))
        self.ww = Parameter(torch.randn(17, 16)).cuda()
        self.weight = Parameter(torch.randn(17, 17)).cuda()
        self.w5 = Parameter(torch.randn(16, 16))
        self.w6 = Parameter(torch.randn(16, 16))
        self.w7 = Parameter(torch.randn(16, 16))
        self.w8 = Parameter(torch.randn(16, 16))
        self.w20 = Parameter(torch.randn(2, 1))

        self.height = height
        self.wl = Parameter(torch.randn(class_num - 1, class_num - 1))
        self.bn2 = nn.BatchNorm2d(1).cuda()
        self.bn = nn.BatchNorm2d(16).cuda()
        self.bn1 = nn.BatchNorm2d(out_channels).cuda()
        self.reset_parameters()
        self.in_features = in_channels
        self.out_features = out_channels
        self.block_num = out_channels
        self.linear = nn.Linear(2, 1).cuda()

    def reset_parameters(self):
        init.kaiming_uniform_(self.w, a=math.sqrt(5))
        init.kaiming_uniform_(self.ww, a=math.sqrt(5))
        init.kaiming_uniform_(self.We, a=math.sqrt(5))
        init.kaiming_uniform_(self.wl, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.w5, a=math.sqrt(5))
        init.kaiming_uniform_(self.w6, a=math.sqrt(5))
        init.kaiming_uniform_(self.w7, a=math.sqrt(5))
        init.kaiming_uniform_(self.w8, a=math.sqrt(5))
        init.kaiming_uniform_(self.w20, a=math.sqrt(5))


    def forward(self, input, lidar, index):

        input = input * 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        index = nn.UpsamplingNearest2d(size=(lidar.shape[2], lidar.shape[3]))(index.float()).long().to(device)
        batch_size = input.shape[0]
        channels = input.shape[1]
        channel_means = torch.mean(input, dim=(2, 3)).to(device)
        channel_means_ = channel_means.repeat(channels, 1, 1).permute(1, 2, 0)
        channel_sub = (channel_means_ - channel_means.unsqueeze(1)).permute(0, 2, 1)
        channel_sub = getf(channel_sub).to(device)
        fh = input.view(1,input.shape[1],-1)
        coh = cosine_similarities(fh)
        edgh = c(coh)
        channel_sub_ = channel_sub.repeat(channels, 1, 1, 1).permute(1, 2, 0, 3)
        channel_sub_ = (channel_sub_ - channel_sub.unsqueeze(1)).permute(0, 2, 1, 3)

        index_ex = torch.zeros(batch_size, self.block_num, lidar.shape[2], lidar.shape[3]).to(device)
        index_ex = index_ex.scatter_(1, index - 1, 1).to(device)
        block_value_sum = torch.sum(index_ex, dim=(2, 3))

        li_ = lidar.repeat(self.block_num, 1, 1, 1, 1).permute(1, 0, 2, 3, 4).to(device)

        index_ex = index_ex.unsqueeze(2)
        li_means = torch.sum(index_ex * li_, dim=(2, 3, 4)) / (block_value_sum + (block_value_sum == 0).float())
        li_means_ = li_means.repeat(self.block_num, 1, 1).permute(1, 2, 0)
        li_sub = (li_means_ - li_means.unsqueeze(1)).permute(0, 2, 1)
        li_sub = getf(li_sub).to(device)

        edgl = connections(li_means)
        li_sub_ = li_sub.repeat(self.block_num, 1, 1, 1).permute(1, 2, 0, 3)
        li_sub_ = (li_sub_ - li_sub.unsqueeze(1)).permute(0, 2, 1, 3)

        M = (self.We).mm(self.We.T).to(device)
        ad = channel_sub_.reshape(batch_size, -1, channels - 1).matmul(M)
        ad = torch.mean(ad * channel_sub_.reshape(batch_size, -1, channels - 1), dim=2).view(batch_size, channels,
                                                                                             channels)
        sig = 0.00000007
        ad = torch.exp(-1 / (2 * sig ** 2) * ad)
        ad = ad * edgh
        ad = nod(ad)
        lm = (self.wl).mm(self.wl.T).to(device)
        la = li_sub_.reshape(batch_size, -1, self.block_num - 1).matmul(lm)
        la = (
            torch.mean(la * li_sub_.reshape(batch_size, -1, self.block_num - 1), dim=2).view(batch_size, self.block_num,
                                                                                             self.block_num))
        si = 0.5
        la = torch.exp(-1 / (2 * si ** 2) * la)

        la = la * edgl
        la=nod(la)
        weights = torch.sigmoid(self.linear(torch.stack([ad, la], dim=-1))).cuda()  # 学习权重

        weight = weights.squeeze(3)

        aa = weight * ad + (1 - weight) * la

        ad_means = torch.matmul(aa, fh)
        ad_means = torch.matmul(ad_means.permute(0, 2, 1), self.w5.cuda())
        ad_means = torch.matmul(aa, ad_means.permute(0, 2, 1))
        ad_means = torch.matmul(ad_means.permute(0, 2, 1), self.w6.cuda())

        ad_means = ad_means.permute(0, 2, 1).view(1, input.shape[1], input.shape[2], input.shape[3])


        la_means = torch.matmul(aa, li_sub)
        la_means = torch.matmul(la_means.permute(0, 2, 1), self.w7.cuda())
        la_means = torch.matmul(aa, la_means.permute(0, 2, 1))
        la_means = torch.matmul(la_means.permute(0, 2, 1), self.w8.cuda())
        la_means = torch.mean(la_means, dim=1)

        index_ex = index_ex.squeeze(2)
        l_features = torch.sum(index_ex * (lidar + la_means.unsqueeze(2).unsqueeze(3)), dim=1)
        l_features = l_features.unsqueeze(0)
        l_features = self.bn2(l_features)

        h_feature = input + ad_means
        h_feature1 = self.bn(h_feature)
        h_feature = h_feature1

        feature = torch.cat((h_feature, l_features), dim=1)
        feature = feature.permute(0, 2, 3, 1).matmul(self.ww).permute(0, 3, 1, 2)

        return feature


# %%
def train_and_test():

    Net = SegNet(in_channel=channel_num,
                 block_num=Block_num,
                 width=data_width,
                 height=data_height,
                 class_num=class_num + 1,
                 adj_mask=adj_mask,
                 if_feature_update=if_up,
                 scale_layer=scale).cuda()

    FusionNet = graphfusion(in_channels=channel_num,
                            out_channels=Block_num,
                            width=data_width,
                            height=data_height,
                            class_num=class_num + 1)


    lossf = nn.CrossEntropyLoss()
    #lossf = nn.CrossEntropyLoss()
    EPOCHS = 1000
    FOUND_LR = 0.001
    opt = torch.optim.Adam(list(Net.parameters()) + list(FusionNet.parameters()), lr=FOUND_LR, weight_decay=0.0)

    scheduler = lr_scheduler.StepLR(opt, step_size=500, gamma=0.6)

    best_kappa = 0
    best_AA = 0
    losses = []

    hsimg = hsi_data.permute(2, 0, 1).unsqueeze(0).cuda()
    lida = lidar_data.permute(2, 0, 1).unsqueeze(0).cuda()

    for epoch in tqdm(range(EPOCHS)):
        Net.train()
        FusionNet.train()
        hf, lf = Net(hsimg, lida, seg_index.unsqueeze(0))
        classes = FusionNet(hf, lf, seg_index.unsqueeze(0))
        train_gt = combined_labels * train_mask
        pre_gt = torch.cat((train_gt.unsqueeze(0).cuda(), classes[0]), dim=0).view(class_num + 2, -1).permute(1, 0)
        pre_gt_ = pre_gt[torch.argsort(pre_gt[:, 0], descending=True)]
        pre_gt_ = pre_gt_[:int(train_sum)]
        OA, AA, kappa, ac_list, m = performance(pre_gt_[:, 1:], pre_gt_[:, 0].long(), class_num)
        print('epoch', epoch, ':', 'train_OA:', OA, 'train_AA:', AA, 'train_KAPPA:', kappa)
        loss = lossf(pre_gt_[:, 1:], pre_gt_[:, 0].long())
        losses.append(float(loss))
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                Net.eval()
                FusionNet.eval()
                hf, lf = Net(hsimg, lida, seg_index.unsqueeze(0))
                classes = FusionNet(hf, lf, seg_index.unsqueeze(0))
                test_gt = combined_labels * test_mask
                pre_gt = torch.cat((test_gt.unsqueeze(0).cuda(), classes[0]), dim=0).view(class_num + 2, -1).permute(1,
                                                                                                                     0)
                pre_gt_ = pre_gt[torch.argsort(pre_gt[:, 0], descending=True)]
                pre_gt_ = pre_gt_[:int(test_sum)]
                OA, AA, kappa, ac_list, m = performance(pre_gt_[:, 1:], pre_gt_[:, 0].long(), class_num)

                output_file_path = 'm/confusion_matrix.txt'
                with open(output_file_path, 'w') as f:
                    for row in m:
                        f.write('\t'.join(map(str, row)) + '\n')

                if best_kappa < kappa:
                    best_kappa = kappa
                    best_OA = OA
                    best_AA = AA
                    best_list = ac_list
                    torch.save(Net.state_dict(), 'dir_H/best_kappa.pth')
                    os.makedirs('immhh', exist_ok=True)

                    visualize_epoch(epoch, classes[0], combined_labels)

                    metrics_file_path = os.path.join('t41', 'metrics.txt')
                    with open(metrics_file_path, 'w') as f:
                        f.write(f'Epoch: {epoch}\n')
                        f.write(f'OA: {OA:.4f}\n')
                        f.write(f'AA: {AA:.4f}\n')
                        f.write(f'Kappa: {kappa:.4f}\n')
                        f.write('Accuracy List:\n')
                        for i, accuracy in enumerate(ac_list):
                            f.write(f'Class {i}: {accuracy:.4f}\n')
                    print('epoch', epoch, ':', 'test_OA:', OA, 'tset_AA:', AA, 'test_KAPPA:', kappa)
                    print('epoch', epoch, ':', 'Accuracy_list:', ac_list)

    return best_kappa, best_OA, best_AA, best_list


# %%
train_sum = torch.sum(train_mask)
test_sum = torch.sum(test_mask)

best_kappas = []
best_OAs = []
best_AAs = []
best_lists = []

list_best_OAs_mean = []
list_best_AAs_mean = []
list_best_kappas_mean = []

list_best_OAs_std = []
list_best_AAs_std = []
list_best_kappas_std = []


scale_list = [4]
feature_up_list = [True]

for scale in scale_list:
    for if_up in feature_up_list:
        for i in range(1):
            # Model
            best_kappa, best_OA, best_AA, best_list = train_and_test()
            best_kappas.append(best_kappa)
            best_OAs.append(best_OA)
            best_AAs.append(best_AA)
            best_lists.append(best_list)

        list_best_OAs_mean.append(np.mean(best_OAs))
        list_best_AAs_mean.append(np.mean(best_AAs))
        list_best_kappas_mean.append(np.mean(best_kappas))

        list_best_OAs_std.append(np.std(best_OAs))
        list_best_AAs_std.append(np.std(best_AAs))
        list_best_kappas_std.append(np.std(best_kappas))


# %%
# %15
print('OA:', np.mean(best_OAs), '+-', np.std(best_OAs))
print('AA:', np.mean(best_AAs), '+-', np.std(best_AAs))
print('kappa:', np.mean(best_kappas), '+-', np.std(best_kappas))
print('list:', np.mean(np.array(best_lists), axis=0), '+-', np.std(np.array(best_lists), axis=0))
# %%
# %5
folder_name = 're'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)


oa_mean = np.mean(best_OAs)
oa_std = np.std(best_OAs)
aa_mean = np.mean(best_AAs)
aa_std = np.std(best_AAs)
kappa_mean = np.mean(best_kappas)
kappa_std = np.std(best_kappas)
list_mean = np.mean(np.array(best_lists), axis=0)
list_std = np.std(np.array(best_lists), axis=0)

file_path = os.path.join(folder_name, 'results.txt')

with open(file_path, 'w') as file:
    file.write(f'OA: {oa_mean} +- {oa_std}\n')
    file.write(f'AA: {aa_mean} +- {aa_std}\n')
    file.write(f'kappa: {kappa_mean} +- {kappa_std}\n')
    file.write(f'list: {list_mean} +- {list_std}\n')
