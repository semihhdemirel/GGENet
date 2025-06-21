import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CBAM import CBAM

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class GeLU_GRN_Conv_Module(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(GeLU_GRN_Conv_Module, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.grn1 = GRN(in_channels)
        self.grn2 = GRN(out_channels)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.grn1(x.permute(0, 2, 3, 1))
        x = x.permute(0, 3, 1, 2)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.grn2(x.permute(0, 2, 3, 1))
        x = x.permute(0, 3, 1, 2)
        x += residual
        x = self.gelu(x)
        
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return x

class GGENet_T(nn.Module):
    def __init__(self):
        super(GGENet_T, self).__init__()

        #Early convolution
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.grn = GRN(32)
        self.gelu = nn.GELU()

        #Block1
        self.geluGRN1 = GeLU_GRN_Conv_Module(32, 32)
        self.geluGRN2 = GeLU_GRN_Conv_Module(32, 32)
        self.downsample1 = Downsample(32, 64)

        #Attention module
        self.cbam1 = CBAM(64)

        #Block2
        self.geluGRN3 = GeLU_GRN_Conv_Module(64, 64)
        self.geluGRN4 = GeLU_GRN_Conv_Module(64, 64)
        self.downsample2 = Downsample(64, 128)
        self.geluGRN5 = GeLU_GRN_Conv_Module(128, 128)
        self.geluGRN6 = GeLU_GRN_Conv_Module(128, 128)
        self.downsample3 = Downsample(128, 256)

        #Attention module
        self.cbam2 = CBAM(256)

        #Block3
        self.geluGRN7 = GeLU_GRN_Conv_Module(256, 256)
        self.geluGRN8 = GeLU_GRN_Conv_Module(256, 256)
        self.geluGRN9 = GeLU_GRN_Conv_Module(256, 256)
        self.downsample4 = Downsample(256, 512)
        self.geluGRN10 = GeLU_GRN_Conv_Module(512, 512)
        self.geluGRN11 = GeLU_GRN_Conv_Module(512, 512)
        self.geluGRN12 = GeLU_GRN_Conv_Module(512, 512)
        self.downsample5 = Downsample(512, 1024)

        #Attention module
        self.cbam3 = CBAM(1024)

        #Block4
        self.geluGRN13 = GeLU_GRN_Conv_Module(1024, 1024)
        self.geluGRN14 = GeLU_GRN_Conv_Module(1024, 1024)
        self.geluGRN15 = GeLU_GRN_Conv_Module(1024, 1024)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 6)

    def forward(self, x):

        x = self.conv1(x)
        x = self.grn(x.permute(0, 2, 3, 1))
        x = x.permute(0, 3, 1, 2)
        x = self.gelu(x)
        x = self.geluGRN1(x)
        x = self.geluGRN2(x)
        x = self.downsample1(x)
        cbam1 = self.cbam1(x)
        x = self.geluGRN3(cbam1)
        x = self.geluGRN4(x)
        x = self.downsample2(x)
        x = self.geluGRN5(x)
        x = self.geluGRN6(x)
        x = self.downsample3(x)
        cbam2 = self.cbam2(x)
        x = self.geluGRN7(cbam2)
        x = self.geluGRN8(x)  
        x = self.geluGRN9(x)
        x = self.downsample4(x)
        x = self.geluGRN10(x)
        x = self.geluGRN11(x)
        x = self.geluGRN12(x)
        x = self.downsample5(x)
        cbam3 = self.cbam3(x)
        x = self.geluGRN13(cbam3)
        x = self.geluGRN14(x)
        x = self.geluGRN15(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, [cbam1, cbam2, cbam3]

class GGENet_S(nn.Module):
    def __init__(self):
        super(GGENet_S, self).__init__()

        #Early convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.grn = GRN(16)  # Adjusted to double the input channels
        self.gelu = nn.GELU()

        #Block1
        self.geluGRN1 = GeLU_GRN_Conv_Module(16, 16)  # Doubled the channels
        self.geluGRN2 = GeLU_GRN_Conv_Module(16, 16)
        self.downsample1 = Downsample(16, 32)  # Doubled the channels

        #Attention module
        self.cbam1 = CBAM(32)  # Adjusted to match the increased channels
        self.regressor1 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)

        #Block2
        self.geluGRN3 = GeLU_GRN_Conv_Module(32, 32)
        self.geluGRN4 = GeLU_GRN_Conv_Module(32, 32)
        self.downsample2 = Downsample(32, 64)  # Doubled the channels
        self.geluGRN5 = GeLU_GRN_Conv_Module(64, 64)
        self.geluGRN6 = GeLU_GRN_Conv_Module(64, 64)
        self.downsample3 = Downsample(64, 128)  # Doubled the channels

        #Attention module
        self.cbam2 = CBAM(128)  # Adjusted to match the increased channels
        self.regressor2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        
        #Block3
        self.geluGRN7 = GeLU_GRN_Conv_Module(128, 128)
        self.geluGRN8 = GeLU_GRN_Conv_Module(128, 128)
        self.geluGRN9 = GeLU_GRN_Conv_Module(128, 128)
        self.downsample4 = Downsample(128, 256)  # Doubled the channels
        self.geluGRN10 = GeLU_GRN_Conv_Module(256, 256)
        self.geluGRN11 = GeLU_GRN_Conv_Module(256, 256)
        self.geluGRN12 = GeLU_GRN_Conv_Module(256, 256)
        self.downsample5 = Downsample(256, 512)  # Doubled the channels

        #Attention module
        self.cbam3 = CBAM(512)  # Adjusted to match the increased channels
        self.regressor3 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0)

        #Block4
        self.geluGRN13 = GeLU_GRN_Conv_Module(512, 512)
        self.geluGRN14 = GeLU_GRN_Conv_Module(512, 512)
        self.geluGRN15 = GeLU_GRN_Conv_Module(512, 512)

        #Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 6)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.grn(x.permute(0, 2, 3, 1))
        x = x.permute(0, 3, 1, 2)
        x = self.gelu(x)
        x = self.geluGRN1(x)
        x = self.geluGRN2(x)
        x = self.downsample1(x)
        cbam1 = self.cbam1(x)
        regressor1 = self.regressor1(cbam1)
        x = self.geluGRN3(cbam1)
        x = self.geluGRN4(x)
        x = self.downsample2(x)
        x = self.geluGRN5(x)
        x = self.geluGRN6(x)
        x = self.downsample3(x)
        cbam2 = self.cbam2(x)
        regressor2 = self.regressor2(cbam2)
        x = self.geluGRN7(cbam2)
        x = self.geluGRN8(x)   
        x = self.geluGRN9(x)
        x = self.downsample4(x)
        x = self.geluGRN10(x)
        x = self.geluGRN11(x)
        x = self.geluGRN12(x)
        x = self.downsample5(x)
        cbam3 = self.cbam3(x)
        regressor3 = self.regressor3(cbam3)
        x = self.geluGRN13(cbam3)
        x = self.geluGRN14(x)
        x = self.geluGRN15(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, [regressor1, regressor2, regressor3]