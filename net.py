import torch, math
import torch.nn as nn
import torch.nn.functional as F

class CBA(nn.Module):
    conv:nn.Conv2d
    bn:nn.BatchNorm2d
    activation:nn.Module
    out_ch:int

    def __init__(self,
                 in_ch:int,
                 out_ch:int,
                 kernel:int,
                 stride:int=1,
                 padding:int=0,
                 activation=nn.SiLU(),
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv= nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.activation = activation
        self.out_ch = out_ch

        nn.init.xavier_uniform_(self.conv.weight)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        y = self.conv.forward(x)
        y_bn = self.bn.forward(y)
        return self.activation.forward(y_bn)

class GridNetBackbone(nn.Module):
    
    # M
    # conv3_ch = (32, 128, 256, 256) 
    # conv5_ch = (128,)
    # dconv_ch = (256, 512)

    # L
    conv3_ch = (48, 196, 384, 384) 
    conv5_ch = (196,)
    dconv_ch = (384, 640)

    conv0_3:CBA
    dconv0_4:CBA
    conv1_3:CBA
    conv1_5:CBA
    dconv1_2:CBA
    conv2_3:CBA
    conv3_3:CBA

    out_channels:int

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv0_3 = CBA(3, self.conv3_ch[0], 3, 1, 1)
        self.dconv0_4 = CBA(self.conv3_ch[0], self.dconv_ch[0], 4, 4)
        self.conv1_3 = CBA(self.dconv_ch[0], self.conv3_ch[1], 3, 1, 1)
        self.conv1_5 = CBA(self.dconv_ch[0], self.conv5_ch[0], 5, 1, 2)
        self.dconv1_2 = CBA(self.dconv_ch[0] + self.conv3_ch[1] + self.conv5_ch[0], self.dconv_ch[1], 2, 2)
        self.conv2_3 = CBA(self.dconv_ch[1], self.conv3_ch[2], 3, 1, 1)
        self.conv3_3 = CBA(self.conv3_ch[2], self.conv3_ch[3], 3, 1, 1)

        self.out_channels = self.conv3_ch[-1] + self.dconv_ch[-1] + self.conv3_ch[-2]


    def forward(self, x:torch.Tensor) -> torch.Tensor:

        x0_3 = self.conv0_3.forward(x)
        x0_4 = self.dconv0_4.forward(x0_3)
        x1_3 = self.conv1_3.forward(x0_4)
        x1_5 = self.conv1_5.forward(x0_4)
        x1_2 = self.dconv1_2.forward(torch.cat((x0_4, x1_3, x1_5), dim=-3))
        x2_3 = self.conv2_3.forward(x1_2)
        x3_3 = self.conv3_3.forward(x2_3)

        return torch.cat((x1_2, x2_3, x3_3), dim=-3)


class GridNetHead(nn.Module):

    # ch = (512,512) # M
    ch = (640,640) # L

    layers:nn.ModuleList
    activations:nn.ModuleList

    def __init__(self, in_channels:int, out_channels:int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        first_layer_out = out_channels
        if len(self.ch) > 0:
            first_layer_out = self.ch[0]

        self.layers.append(nn.Conv2d(in_channels, first_layer_out, 1))
        for i in range(len(self.ch)-1):
            self.layers.append(nn.Conv2d(self.ch[i], self.ch[i+1], 1))
            self.activations.append(nn.PReLU())

        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
        
        if len(self.ch) > 0:
            self.activations.append(nn.PReLU())
            self.layers.append(nn.Conv2d(self.ch[-1], out_channels, 1))

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        for i in range(len(self.layers)-1):
            x = self.layers[i].forward(x)
            x = self.activations[i](x)
        
        return self.layers[-1].forward(x)


class GridNet(nn.Module):
    abox_count:int
    backbone:GridNetBackbone
    head:GridNetHead
    downscaler:nn.AvgPool2d
    sigmoid_scaler:float

    def __init__(self, abox_count:int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.abox_count = abox_count
        out_channels = 10 + 3 * abox_count

        self.backbone = GridNetBackbone()
        self.head = GridNetHead(self.backbone.out_channels, out_channels)

        self.downscaler = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        grids_output = []
        while True:

            # Network layers pass
            y = self.backbone.forward(x)
            y = self.head.forward(y)

            # flatten output
            y = y.flatten(start_dim=-2, end_dim=-1)
            y = y.transpose(-2, -1)

            # save in list to merge at the end
            grids_output.insert(0,y)

            # exit condition
            if x.shape[-1] <= 32:
                break

            # dowscale image
            x = self.downscaler(x)
        
        netout = torch.cat(grids_output, dim=-2)

        netout_probs = netout[..., 0:2]
        netout_color_classes = netout[..., 2:8]
        netout_center_offset = netout[..., 8:10]
        netout_abox_probs = netout[..., 10:10+self.abox_count]
        netout_abox_offsets = netout[..., 10+self.abox_count:]

        netout_probs = F.sigmoid(netout_probs)
        netout_center_offset = F.sigmoid(netout_center_offset)
        netout_abox_offsets = F.sigmoid(netout_abox_offsets)

        if self.eval: # use softmax only during evaluation
            netout_color_classes = F.softmax(netout_color_classes, dim=-1)
            netout_abox_probs = F.softmax(netout_abox_probs, dim=-1)


        netout = torch.cat((netout_probs, netout_color_classes, netout_center_offset, netout_abox_probs, netout_abox_offsets), dim=-1)

        return netout
