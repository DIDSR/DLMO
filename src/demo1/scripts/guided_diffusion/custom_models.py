
import torch
import torch.nn as nn



class VGG16_custom(nn.Module):
    def __init__(self, num_input_channel,image_size,num_classes=10,ngf=64):
        super(VGG16_custom, self).__init__()
        self.image_size=image_size
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_input_channel, ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.image_size = self.image_size/2
        self.layer3 = nn.Sequential(
            nn.Conv2d(ngf, 2*ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2*ngf),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(2*ngf, 2*ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2*ngf),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.image_size = self.image_size/2
        self.layer5 = nn.Sequential(
            nn.Conv2d(2*ngf, 4*ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*ngf),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(4*ngf, 4*ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*ngf),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(4*ngf, 4*ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*ngf),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.image_size = self.image_size/2
        self.layer8 = nn.Sequential(
            nn.Conv2d(4*ngf, 8*ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*ngf),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(8*ngf, 8*ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*ngf),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(8*ngf, 8*ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*ngf),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.image_size = self.image_size/2
        self.layer11 = nn.Sequential(
            nn.Conv2d(8*ngf, 8*ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*ngf),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(8*ngf, 8*ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*ngf),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(8*ngf, 8*ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*ngf),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.image_size = self.image_size/2
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(int(self.image_size*self.image_size)*8*ngf, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out