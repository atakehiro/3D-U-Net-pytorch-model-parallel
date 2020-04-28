# 3D-U-Net-pytorch-model-parallel
このコードはhttps://github.com/JielongZ/3D-UNet-PyTorch-Implementation をModelParallel出来るように改変したものです。

Code is adapted from https://github.com/JielongZ/3D-UNet-PyTorch-Implementation for ModelParallel of 2 GPU.

エンコーダ部分をGPU1で計算し、デコーダ部分をGPU2で計算するようになっています。

<img src="u-net.jpg" width="1000" align="below">
以上のModelParallelによって、大きなネットワークの学習が可能です。

## Usage
Main.pyで画像のパスやwindow sizeなどのパラメータを調整して実行


## Author
Takehiro Ajioka

E-mail:1790651m@stu.kobe-u.ac.jp
