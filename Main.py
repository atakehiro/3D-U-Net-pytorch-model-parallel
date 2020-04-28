#!/usr/bin/env python
import sys
import time
import torch
import torch.nn as nn
import tifffile
import numpy as np
from functions import EncoderBlock, DecoderBlock, data_gen, DiceLoss, mean_iou, batch_data_gen

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

if __name__ == "__main__":
    image = tifffile.imread('traindata/training_input.tif')
    label = tifffile.imread('traindata/training_groundtruth.tif')
    window_size = (32, 32, 32)
    input_data = data_gen(image, window_size)
    label_data = data_gen(label, window_size)
    print('data size:', input_data.shape)
    model = UnetModel()
    print(model)
    learning_rate=0.01
    optim = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    criterion = DiceLoss()
    no_epochs = 10
    trainer = Trainer(net=model, optimizer=optim, criterion=criterion, no_epochs=no_epochs)
    trainer.train(input_data, label_data, batch_data_gen)
    pred = trainer.predict(input_data)
    print(pred.shape)
    np.save('prediction', pred)

