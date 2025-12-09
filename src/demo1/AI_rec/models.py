import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as td
import torchvision as tv
import torch.nn.init as init

# from utils import imshow, NNRegressor
# https://github.com/lychengr3x/Image-Denoising-with-Deep-CNNs



class CNN3(nn.Module):
    def __init__(self, num_channels):
        super(CNN3,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=9, padding=4);
        self.relu1 = nn.ReLU();
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1);
        self.relu2 = nn.ReLU();
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, padding=2);
        self.relu3 = nn.ReLU(); # added by ZT. Prevent negative pixel values in outputs

        #self._initialize_weights()

    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        return out

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))

class DnCNN(nn.Module):
    # Denoising CNN
    def __init__(self, D, C=64, num_channels=1):
        super(DnCNN, self).__init__()
        self.D = D

        # convolution layers
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(num_channels, C, 3, padding=1))
        self.conv.extend([nn.Conv2d(C, C, 3, padding=1) for _ in range(D)])
        self.conv.append(nn.Conv2d(C, num_channels, 3, padding=1))
        # apply He's initialization
        for i in range(len(self.conv[:-1])):
            nn.init.kaiming_normal_(
                self.conv[i].weight.data, nonlinearity='relu')

        # batch normalization
        self.bn = nn.ModuleList()
        self.bn.extend([nn.BatchNorm2d(C, C) for _ in range(D)])
        # initialize the weights of the Batch normalization layers
        for i in range(D):
            nn.init.constant_(self.bn[i].weight.data, 1.25 * np.sqrt(C))

    def forward(self, x):
        D = self.D
        h = F.relu(self.conv[0](x))
        for i in range(D):
            h = F.relu(self.bn[i](self.conv[i+1](h)))
        # y = self.conv[D+1](h) + x
        y = F.relu(self.conv[D+1](h) + x) # modified by ZT. Prevent negative pixel values in outputs
        return y

class UDnCNN(nn.Module):
    # u - shaped DnCNN
    def __init__(self, D, C=64, num_channels=1):
        super(UDnCNN, self).__init__()
        self.D = D

        # convolution layers
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(num_channels, C, 3, padding=1))
        self.conv.extend([nn.Conv2d(C, C, 3, padding=1) for _ in range(D)])
        self.conv.append(nn.Conv2d(C, num_channels, 3, padding=1))
        # apply He's initialization
        for i in range(len(self.conv[:-1])):
            nn.init.kaiming_normal_(
                self.conv[i].weight.data, nonlinearity='relu')

        # batch normalization
        self.bn = nn.ModuleList()
        self.bn.extend([nn.BatchNorm2d(C, C) for _ in range(D)])
        # initialize the weights of the Batch normalization layers
        for i in range(D):
            nn.init.constant_(self.bn[i].weight.data, 1.25 * np.sqrt(C))

    def forward(self, x):
        D = self.D
        h = F.relu(self.conv[0](x))
        h_buff = []
        idx_buff = []
        shape_buff = []
        for i in range(D//2-1):
            shape_buff.append(h.shape)
            h, idx = F.max_pool2d(F.relu(self.bn[i](self.conv[i+1](h))),
                                  kernel_size=(2, 2), return_indices=True)
            h_buff.append(h)
            idx_buff.append(idx)
        for i in range(D//2-1, D//2+1):
            h = F.relu(self.bn[i](self.conv[i+1](h)))
        for i in range(D//2+1, D):
            j = i - (D // 2 + 1) + 1
            h = F.max_unpool2d(F.relu(self.bn[i](self.conv[i+1]((h+h_buff[-j])/np.sqrt(2)))),
                               idx_buff[-j], kernel_size=(2, 2), output_size=shape_buff[-j])
        y = self.conv[D+1](h) + x
        return y

class DUDnCNN(nn.Module):
    # dialated U-shaped DnCNN
    def __init__(self, D, C=64, num_channels=1):
        super(DUDnCNN, self).__init__()
        self.D = D

        # compute k(max_pool) and l(max_unpool)
        k = [0]
        k.extend([i for i in range(D//2)])
        k.extend([k[-1] for _ in range(D//2, D+1)])
        l = [0 for _ in range(D//2+1)]
        l.extend([i for i in range(D+1-(D//2+1))])
        l.append(l[-1])

        # holes and dilations for convolution layers
        holes = [2**(kl[0]-kl[1])-1 for kl in zip(k, l)]
        dilations = [i+1 for i in holes]

        # convolution layers
        self.conv = nn.ModuleList()
        self.conv.append(
            nn.Conv2d(num_channels, C, 3, padding=dilations[0], dilation=dilations[0]))
        self.conv.extend([nn.Conv2d(C, C, 3, padding=dilations[i+1],
                                    dilation=dilations[i+1]) for i in range(D)])
        self.conv.append(
            nn.Conv2d(C, num_channels, 3, padding=dilations[-1], dilation=dilations[-1]))
        # apply He's initialization
        for i in range(len(self.conv[:-1])):
            nn.init.kaiming_normal_(
                self.conv[i].weight.data, nonlinearity='relu')

        # batch normalization
        self.bn = nn.ModuleList()
        self.bn.extend([nn.BatchNorm2d(C, C) for _ in range(D)])
        # initialize the weights of the Batch normalization layers
        for i in range(D):
            nn.init.constant_(self.bn[i].weight.data, 1.25 * np.sqrt(C))

    def forward(self, x):
        D = self.D
        h = F.relu(self.conv[0](x))
        h_buff = []

        for i in range(D//2 - 1):
            torch.backends.cudnn.benchmark = True
            h = self.conv[i+1](h)
            torch.backends.cudnn.benchmark = False
            h = F.relu(self.bn[i](h))
            h_buff.append(h)

        for i in range(D//2 - 1, D//2 + 1):
            torch.backends.cudnn.benchmark = True
            h = self.conv[i+1](h)
            torch.backends.cudnn.benchmark = False
            h = F.relu(self.bn[i](h))

        for i in range(D//2 + 1, D):
            j = i - (D//2 + 1) + 1
            torch.backends.cudnn.benchmark = True
            h = self.conv[i+1]((h + h_buff[-j]) / np.sqrt(2))
            torch.backends.cudnn.benchmark = False
            h = F.relu(self.bn[i](h))

        y = self.conv[D+1](h) + x
        return y

class REDcnn10(nn.Module):

    def __init__(self, d=96, s=5, batch_normalization=False, idmaps=3, bias=True):
        """ Pytorch implementation of Redcnn following the paper [1]_, [2]_.
        Notes
        -----
        In [1]_, authors have suggested three architectures:
        a. RED10, which has 10 layers and does not use any skip connection (hence skip_step = 0)
        b. RED20, which has 20 layers and uses skip_step = 2
        c. RED30, which has 30 layers and uses skip_step = 2
        d. It also shows that kernel size 7x7 & 9X9 yields better performance than (5x5) or (3x3)
           However, it argues that using large kernel size may lead to poor optimum for high-level tasks.
        e. Moreover, using filter size 64, while the kernel size is (3, 3), it shows 100x100 patch yeilds
           the best denoised result
        In [2]_, where RedNet was used for CT denoising, authors have suggested:
        a. Red10 with 3 skip connections
        b. patch-size (55x55)
        c. kernel-size (5X5) & no. of filters(96)

        Parameters
        ----------
        depth (hard coded to be 10, following [2]_)
            Number of fully convolutional layers in dncnn. In the original paper, the authors have used depth=17
            for non-blind denoising (same noise level) and depth=20 for blind denoising (different noise level).
        d : int
            Number of filters at each convolutional layer.
        s : kernel size (int)
            kernel window used to compute activations.

        Returns
        -------
        :class:`torch.nn.Module`
            Torch Model representing the Denoiser neural network
        References
        ----------
        .. [1] Mao, X., Shen, C., & Yang, Y. B. (2016). Image restoration using convolutional auto-encoders
            with symmetric skip connections. In Advances in neural information processing systems
        .. [2] Chen, Hu, et al. "Low-dose CT with a residual encoder-decoder convolutional neural network".
           IEEE transactions on medical imaging 36.12 (2017):
        """
        super(REDcnn10, self).__init__()
        if (s==9): pad = 4
        elif (s==5): pad = 2
        else: pad = 1
        self.batch_normalization=batch_normalization
        self.idmaps = idmaps

        ## Encoding layers ##
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=d, kernel_size=s, padding=pad, bias=bias)
        self.conv2 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=s, padding=pad, bias=bias)
        self.conv3 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=s, padding=pad, bias=bias)
        self.conv4 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=s, padding=pad, bias=bias)
        ## Decoder layers ##
        self.t_conv1 = nn.ConvTranspose2d(in_channels=(d), out_channels=d, kernel_size=s, stride=1, padding=pad, bias=bias)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=d, out_channels=d, kernel_size=s, stride=1, padding=pad, bias=bias)
        self.t_conv3 = nn.ConvTranspose2d(in_channels=d, out_channels=d, kernel_size=s, stride=1, padding=pad, bias=bias)
        self.t_conv4 = nn.ConvTranspose2d(in_channels=d, out_channels=d, kernel_size=s, stride=1, padding=pad, bias=bias)
        self.t_cout  = nn.ConvTranspose2d(in_channels=d, out_channels=1, kernel_size=s, stride=1, padding=pad, bias=bias)
        self.inbn    = nn.InstanceNorm2d(d, affine=True) #normalizes each batch independently
        self.sbn     = nn.BatchNorm2d(d)

        '''for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

        self.relu = nn.ReLU(inplace=True) #True:optimally does operation when inplace is true
        #self._initialize_weights()
        #self.relu = nn.LeakyReLU(0.2, inplace=True)
        '''

    def forward(self, x):
        if (self.batch_normalization==True):
            ''' This BN is my addition to the existing model.
            However a proper initilization for conv weights
            and bn layer is still required
            '''
            ## encode ##
            xinit = x
            x     = F.relu(self.sbn(self.conv1(x)))
            x2    = x.clone()
            x     = F.relu(self.sbn(self.conv2(x)))
            x     = F.relu(self.sbn(self.conv3(x)))
            x4    = x.clone()
            x     = F.relu(self.sbn(self.conv4(x)))
            ## decode ##
            x = F.relu(self.sbn(self.t_conv1(x))+x4)
            x = F.relu(self.sbn(self.t_conv2(x)))
            x = F.relu(self.sbn(self.t_conv3(x))+x2)
            x = F.relu(self.sbn(self.t_conv4(x)))
            x = (self.t_cout(x)+xinit)
        else:
            ## endcode #
            xinit = x
            x     = F.relu(self.conv1(x))
            x2    = x.clone()
            x     = F.relu(self.conv2(x))
            x     = F.relu(self.conv3(x))
            x4    = x.clone()
            x     = F.relu(self.conv4(x))
            ## decode ##
            if (self.idmaps==1):
                x = F.relu(self.t_conv1(x))
                x = F.relu(self.t_conv2(x))
                x = F.relu(self.t_conv3(x))
                x = F.relu(self.t_conv4(x))
                x = F.relu(self.t_cout(x)+xinit)
            else:
                x = F.relu(self.t_conv1(x)+x4)
                x = F.relu(self.t_conv2(x))
                x = F.relu(self.t_conv3(x)+x2)
                x = F.relu(self.t_conv4(x))
                x = F.relu(self.t_cout(x)+xinit)
        return x

class FSRCNN (torch.nn.Module):
    def __init__(self, num_channels, upscale_factor, d=64, s=12, m=4):
        super(FSRCNN, self).__init__()

        self.first_part = nn.Sequential(nn.Conv2d(in_channels=num_channels, out_channels=d, kernel_size=5, stride=1, padding=2),
                                        nn.PReLU())

        self.layers = []
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))
        for _ in range(m):
            self.layers.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))

        self.mid_part = torch.nn.Sequential(*self.layers)

        # Deconvolution
        if upscale_factor == 3:
            self.last_part = nn.ConvTranspose2d(in_channels=d, out_channels=num_channels, kernel_size=9, stride=upscale_factor, padding=3, output_padding=0)
        else:
            self.last_part = nn.ConvTranspose2d(in_channels=d, out_channels=num_channels, kernel_size=9, stride=upscale_factor, padding=3, output_padding=1)
    def forward(self, x):
        out = self.first_part(x)
        out = self.mid_part(out)
        out = self.last_part(out)
        return out

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.0001)
                if m.bias is not None:
                    m.bias.data.zero_()


# UNet used in Kaiyan's SPIE paper
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.layer(x)

class Down_DC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down_DC, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.layer(x)

class Up_DC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up_DC, self).__init__()
        self.dim_output = out_channels
        self.layer1 = F.interpolate
        self.layer2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.layer3 = DoubleConv(in_channels, out_channels)
    def forward(self, x, x_shortcut):
        x = self.layer1(x, scale_factor=2)
        x = self.layer2(x)
        x = torch.cat((x_shortcut, x), dim=1)
        x = self.layer3(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.inc = DoubleConv(1, 64)
        self.down1 = Down_DC(64, 128)
        self.down2 = Down_DC(128, 256)
        self.down3 = Down_DC(256, 512)
        self.down4 = Down_DC(512, 1024)
        self.up4 = Up_DC(1024, 512)
        self.up3 = Up_DC(512, 256)
        self.up2 = Up_DC(256, 128)
        self.up1 = Up_DC(128, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.out(x)
        return x