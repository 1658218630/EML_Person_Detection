import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyYoloV2_Pruned(nn.Module):

    def __init__(self, conv_shapes, num_classes=1, anchors=None):
        """
        conv_shapes: The result returned by parse_pruned_conv_shapes(...), 
                     contains the in/out_channels of each layer of conv1~conv9, etc.

        """
        super().__init__()

        if anchors is None:
            anchors = ((1.08, 1.19),
                       (3.42, 4.41),
                       (6.63, 11.38),
                       (9.42, 5.11),
                       (16.62, 10.52))
        self.register_buffer("anchors", torch.tensor(anchors))
        self.num_classes = num_classes

        self.pad = nn.ReflectionPad2d((0, 1, 0, 1))

        # Dynamically create 9 convolutional layers
        conv_names = [f"conv{i}" for i in range(1, 10)]  # ["conv1", "conv2", ..., "conv9"]
        for i, conv_name in enumerate(conv_names):
            info = conv_shapes[i]
            conv = nn.Conv2d(
                in_channels=info['in_channels'],
                out_channels=info['out_channels'],
                kernel_size=info['kernel_size'],
                stride=info['stride'],
                padding=info['padding'],
                bias=info['has_bias']
            )
            setattr(self, conv_name, conv)

    def forward(self, x, yolo=True):

        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv4(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv5(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv6(x)
        x = self.pad(x)
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv7(x)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv8(x)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv9(x)

        # YOLO decode
        if yolo:
            nB, _, nH, nW = x.shape

            x = x.view(
                nB, self.anchors.shape[0], -1, nH, nW
            ).permute(0, 1, 3, 4, 2)

            anchors = self.anchors.to(dtype=x.dtype, device=x.device)
            range_y, range_x = torch.meshgrid(
                torch.arange(nH, dtype=x.dtype, device=x.device),
                torch.arange(nW, dtype=x.dtype, device=x.device)
            )
            anchor_x, anchor_y = anchors[:, 0], anchors[:, 1]

            x = torch.cat([
                (x[:, :, :, :, 0:1].sigmoid() + range_x[None, None, :, :, None]) / nW,  # x center
                (x[:, :, :, :, 1:2].sigmoid() + range_y[None, None, :, :, None]) / nH,  # y center
                (x[:, :, :, :, 2:3].exp() * anchor_x[None, :, None, None, None]) / nW, # width
                (x[:, :, :, :, 3:4].exp() * anchor_y[None, :, None, None, None]) / nH, # height
                x[:, :, :, :, 4:5].sigmoid(),  # confidence
                x[:, :, :, :, 5:].softmax(-1),
            ], dim=-1)
        return x
