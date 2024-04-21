import torch, math
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    channels:tuple[int]
    convs:nn.ModuleList
    convs_1d:nn.ModuleList
    adjust_channels_conv:nn.Conv2d

    def __init__(self, 
                 in_channels,
                 channels:tuple[int]= (  64, 64, 64, 32),
                 kernels:tuple[int]=  (   1,  3,  5,  7),
                 batch_norm_after:bool=False,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert len(channels) == len(kernels)
        assert any( k % 2 == 1 for k in kernels )

        self.channels = channels

        paddings = []
        for i in range(len(kernels)):
            paddings.append(int(float(kernels[i])/2))
        
        self.convs = nn.ModuleList()
        self.convs_1d = nn.ModuleList()

        for i in range(len(channels)):
            self.convs_1d.append(nn.Conv2d(in_channels=in_channels, out_channels=channels[i], kernel_size=1, stride=1, padding=0))
            if channels[i] != 1:
                self.convs.append(nn.Conv2d(in_channels=in_channels, out_channels=channels[i], kernel_size=kernels[i], stride=1, padding=paddings[i], bias=not batch_norm_after, groups=min(in_channels, channels[i])))
        
        self.adjust_channels_conv = nn.Conv2d(in_channels=sum(channels), out_channels=in_channels, kernel_size=1, stride=1, padding=0)

    
    def forward(self, x:torch.Tensor) -> torch.Tensor:

        # convolutions
        convs_1d_out, convs_out = [], []
        for i in range(len(self.convs_1d)):
            convs_1d_out.append(self.convs_1d[i](x))

        for i in range(len(convs_1d_out)):
            if self.channels[i] != 1:
                convs_out.append(self.convs[i](convs_1d_out[i]))

        y = torch.cat(convs_out, dim=-3) # concatenate along channels

        y = self.adjust_channels_conv(y)

        # residual connection
        y = y + x

        return y

def get_GridNet(abox_count:int, channels:tuple[int, ...], inc_block_channels:tuple[tuple[int,...],...], inc_block_kernels:tuple[tuple[int,...],...]) -> nn.Sequential:

    assert len(inc_block_channels) == len(inc_block_kernels)
    for i in range(len(inc_block_channels)):
        assert len(inc_block_channels[i]) == len(inc_block_kernels[i])
    assert all (ch > 0 for ch in channels)

    out_ch_count = 1+1+6+2+abox_count*(1+2)

    return nn.Sequential(
        nn.Conv2d       (in_channels=3,             out_channels=channels[0],       kernel_size=4, stride=4),
        nn.PReLU    (),
        InceptionBlock  (in_channels=channels[0],   channels=inc_block_channels[0], kernels=inc_block_kernels[0], batch_norm_after=True),
        nn.BatchNorm2d  (num_features=channels[0]),
        nn.PReLU    (),
        nn.Conv2d       (in_channels=channels[0],   out_channels=channels[1],       kernel_size=2, stride=2, groups=channels[0]),
        nn.PReLU    (),
        InceptionBlock  (in_channels=channels[1],   channels=inc_block_channels[1], kernels=inc_block_kernels[1], batch_norm_after=True),
        nn.BatchNorm2d  (num_features=channels[1]),
        nn.PReLU    (),
        nn.Conv2d       (in_channels=channels[1],   out_channels=channels[2],       kernel_size=1),
        nn.PReLU    (),
        InceptionBlock  (in_channels=channels[2],   channels=inc_block_channels[2], kernels=inc_block_kernels[2], batch_norm_after=False),
        nn.PReLU    (),
        InceptionBlock  (in_channels=channels[3],   channels=inc_block_channels[3], kernels=inc_block_kernels[3], batch_norm_after=True),
        nn.BatchNorm2d  (num_features=channels[3]),
        nn.PReLU    (),
        nn.Conv2d       (in_channels=channels[3],   out_channels=out_ch_count,      kernel_size=1)
    )



class DetektorNet(nn.Module):
    abox_count:int
    GridNet_module:nn.Sequential
    downscaler:nn.AvgPool2d

    def __init__(self, abox_count:int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        channels = ( 128, 256, 512, 512 )
        inc_block_channels = (
            (channels[0],channels[0],channels[0]),
            (channels[1],channels[1],channels[1]),
            (channels[2],channels[2]),
            (channels[3],)
        )
        inc_block_kernels = (
            (1,3,5),
            (1,3,5),
            (3,5),
            (3,)
        )

        self.abox_count = abox_count

        self.GridNet_module = get_GridNet(abox_count, channels=channels, inc_block_channels=inc_block_channels, inc_block_kernels=inc_block_kernels)
        self.downscaler = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        grids_output = []
        while True:

            # Network layers pass
            y = self.GridNet_module(x)

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
        netout_abox_offsets = F.sigmoid(netout_abox_offsets)

        if self.eval: # use softmax only during evaluation
            netout_color_classes = F.softmax(netout_color_classes, dim=-1)
            netout_abox_probs = F.softmax(netout_abox_probs, dim=-1)


        netout = torch.cat((netout_probs, netout_color_classes, netout_center_offset, netout_abox_probs, netout_abox_offsets), dim=-1)

        return netout
