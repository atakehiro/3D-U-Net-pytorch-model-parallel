#!/usr/bin/env python
import torch
import tifffile
import numpy as np
from functions import UnetModel, Trainer, data_gen, DiceLoss,  batch_data_gen

if __name__ == "__main__":
    image = tifffile.imread('traindata/training_input.tif')
    label = tifffile.imread('traindata/training_groundtruth.tif')
    window_size = (64, 64, 128)
    input_data = data_gen(image, window_size)
    label_data = data_gen(label, window_size)
    print('image size:', image.shape)
    print('data size:', input_data.shape)
    model = UnetModel()
    print(model)
    learning_rate = 0.01
    optim = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    criterion = DiceLoss()
    no_epochs = 10
    trainer = Trainer(net=model, optimizer=optim, criterion=criterion, no_epochs=no_epochs)
    trainer.train(input_data, label_data, batch_data_gen)
    pred = trainer.predict(input_data)
    print(pred.shape)
    np.save('prediction', pred)

