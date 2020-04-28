import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def data_gen(image, window_size):
    size = image.shape
    mod0 = size[0] % window_size[0]
    mod1 = size[1] % window_size[1]
    mod2 = size[2] % window_size[2]
    tmp0 = np.delete(image, list(range(image.shape[0] - mod0,image.shape[0])), axis=0)
    tmp1 = np.delete(tmp0, list(range(image.shape[1] - mod1,image.shape[1])), axis=1)
    tmp2 = np.delete(tmp1, list(range(image.shape[2] - mod2,image.shape[2])), axis=2)
    data = tmp2.reshape((-1,1)+window_size)
    return data

def batch_data_gen(pet_imgs, mask_imgs, iter_step, batch_size=6):
    np.random.seed(seed=1)
    permutation_idxs = np.random.permutation(len(pet_imgs))
    pet_imgs = pet_imgs[permutation_idxs]
    mask_imgs = mask_imgs[permutation_idxs]
    step_count = batch_size * iter_step
    return pet_imgs[step_count: batch_size + step_count], mask_imgs[step_count: batch_size + step_count]

def mean_iou(outputs, labels):
    SMOOTH = 1e-6
    outputs = outputs.squeeze(1)
    labels = labels.squeeze(1)
    intersection = (outputs & labels).float().sum((1, 2, 3))
    union = (outputs | labels).float().sum((1, 2, 3))
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou.mean()

class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
    def forward(self, targets, logits):
        batch_size = targets.size(0)
        logits = logits.view(batch_size, -1).type(torch.FloatTensor)
        targets = targets.view(batch_size, -1).type(torch.FloatTensor)
        intersection = (logits * targets).sum(-1)
        dice_score = 2. * intersection / ((logits + targets).sum(-1) + self.epsilon)
        return torch.mean(1. - dice_score)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)
    def forward(self, x):
        x = self.batch_norm(self.conv3d(x))
        # x = self.conv3d(x)
        x = F.elu(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, model_depth=4, pool_size=2):
        super(EncoderBlock, self).__init__()
        self.root_feat_maps = 16
        self.num_conv_blocks = 2
        self.module_dict = nn.ModuleDict()
        for depth in range(model_depth):
            feat_map_channels = 2 ** (depth + 1) * self.root_feat_maps
            for i in range(self.num_conv_blocks):
                if depth == 0:
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
                else:
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
            if depth == model_depth - 1:
                break
            else:
                self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=2, padding=0)
                self.module_dict["max_pooling_{}".format(depth)] = self.pooling
    def forward(self, x):
        down_sampling_features = []
        for k, op in self.module_dict.items():
            if k.startswith("conv"):
                x = op(x)
                if k.endswith("1"):
                    down_sampling_features.append(x.to('cuda:1'))
            elif k.startswith("max_pooling"):
                x = op(x)
        return x, down_sampling_features

class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=2, padding=1, output_padding=1):
        super(ConvTranspose, self).__init__()
        self.conv3d_transpose = nn.ConvTranspose3d(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=k_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   output_padding=output_padding)

    def forward(self, x):
        return self.conv3d_transpose(x)

class DecoderBlock(nn.Module):
    def __init__(self, out_channels, model_depth=4):
        super(DecoderBlock, self).__init__()
        self.num_conv_blocks = 2
        self.num_feat_maps = 16
        self.module_dict = nn.ModuleDict()
        for depth in range(model_depth - 2, -1, -1):
            feat_map_channels = 2 ** (depth + 1) * self.num_feat_maps
            self.deconv = ConvTranspose(in_channels=feat_map_channels * 4, out_channels=feat_map_channels * 4)
            self.module_dict["deconv_{}".format(depth)] = self.deconv
            for i in range(self.num_conv_blocks):
                if i == 0:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 6, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
                else:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
            if depth == 0:
                self.final_conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=out_channels)
                self.module_dict["final_conv"] = self.final_conv
    def forward(self, x, down_sampling_features):
        for k, op in self.module_dict.items():
            if k.startswith("deconv"):
                x = op(x)
                x = torch.cat((down_sampling_features[int(k[-1])], x), dim=1)
            elif k.startswith("conv"):
                x = op(x)
            else:
                x = op(x)
        return x

class UnetModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, model_depth=4):
        super(UnetModel, self).__init__()
        self.encoder = EncoderBlock(in_channels=in_channels, model_depth=model_depth).to('cuda:0')
        self.decoder = DecoderBlock(out_channels=out_channels, model_depth=model_depth).to('cuda:1')
        self.sigmoid = nn.Sigmoid().to('cuda:1')
    def forward(self, x):
        x, downsampling_features = self.encoder(x)
        x = self.decoder(x.to('cuda:1'), downsampling_features)
        x = self.sigmoid(x)
        return x

class Trainer(object):
    def __init__(self, net, optimizer, criterion, no_epochs, batch_size=8):
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.no_epochs = no_epochs
        self.batch_size = batch_size
    def train(self, input_data, label_data, batch_data_loader):
        self.net.train()
        pets = input_data
        masks = label_data
        training_steps = len(pets) // self.batch_size
        for epoch in range(self.no_epochs):
            start_time = time.time()
            train_losses, train_iou = 0, 0
            for step in range(training_steps):
                x_batch, y_batch = batch_data_loader(pets, masks, iter_step=step, batch_size=self.batch_size)
                x_batch = torch.from_numpy(x_batch).float()
                y_batch = torch.from_numpy(y_batch).int()
                self.optimizer.zero_grad()
                logits = self.net(x_batch.to('cuda:0'))
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                train_iou += mean_iou(y_batch.to('cpu'), (logits.to('cpu') > 0.5).int())
                train_losses += loss.item()
            end_time = time.time()
            print("Epoch {}, training loss {:.4f}, time {:.2f}, IoU {:.2f}".format(epoch, train_losses / training_steps, end_time - start_time, train_iou / training_steps))
        torch.save(self.net.state_dict(), 'U-Net3d_model.pt')
    def predict(self, input_data):
        print('test prediction')
        self.net.eval()
        test_preds = np.zeros(input_data.shape)
        steps = len(test_preds) // self.batch_size
        for i in range(steps):
            x = torch.from_numpy(input_data[self.batch_size*i:self.batch_size*i+self.batch_size,:,:,:,:]).float().to('cuda:0')
            test_preds[self.batch_size*i:self.batch_size*i+self.batch_size,:,:,:,:] = self.net(x).int().detach().to('cpu')
        return test_preds