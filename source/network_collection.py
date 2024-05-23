import torch
import torch.nn as nn


##################################################
# Basic Layers
##################################################


def get_norm_layer(norm_type, num_features):
    if norm_type == "batch":
        norm_layer = nn.BatchNorm2d(
            num_features=num_features,
            momentum=0.5,
        )
    elif norm_type == "instance":
        norm_layer = nn.InstanceNorm2d(num_features=num_features)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("Unsupported normalization: {}".format(norm_type))

    return norm_layer


def get_activation_layer(activation_type, num_channels=1):
    if activation_type == "relu":
        activation_layer = nn.ReLU(inplace=True)
    elif activation_type == "prelu":
        activation_layer = nn.PReLU(num_parameters=num_channels)
    elif activation_type == "tanh":
        activation_layer = nn.Tanh()
    elif activation_type == "none" or activation_type is None:
        activation_layer = None
    else:
        raise NotImplementedError("Unsupported activation: {}".format(activation_type))

    return activation_layer


def get_pool_layer(pool_type, kernel_size=2):
    if pool_type == "max":
        pool_layer = nn.MaxPool2d(kernel_size)
    elif pool_type == "avg":
        pool_layer = nn.AvgPool2d(kernel_size)
    else:
        raise NotImplementedError("Unsupported pooling: {}".format(pool_type))

    return pool_layer


def get_pad_layer(pad_type):
    if pad_type in ["refl", "reflect"]:
        pad_layer = nn.ReflectionPad2d
    elif pad_type in ["repl", "replicate"]:
        pad_layer = nn.ReplicationPad2d
    elif pad_type == "zero":
        pad_layer = nn.ZeroPad2d
    else:
        raise NotImplementedError("Unsupported padding: {}".format(pad_type))

    return pad_layer


##################################################
# Basic Blocks
##################################################


class ConvBlock(nn.Module):
    """
    Conv -> Norm -> Activation

    input: (N, C_in, H, W)
    output: (N, C_out, H, W)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        norm="batch",
        activation="relu",
    ):
        super(ConvBlock, self).__init__()

        # convolution layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        # normalization layer
        self.norm = get_norm_layer(norm_type=norm, num_features=out_channels)
        # activation layer
        self.activation = get_activation_layer(
            activation_type=activation, num_channels=out_channels
        )

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):
    """
    [Conv -> Norm -> Activation -> Conv -> Norm] + x -> Activation

    input: (N, C_in, H, W)
    output: (N, C_out, H, W)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_features=None,
        kernel_size=3,
        padding=1,
        norm="batch",
        activation="relu",
        activation_out="relu",
    ):
        super(ResBlock, self).__init__()

        if num_features is None:
            num_features = out_channels

        self.model = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=num_features,
                kernel_size=kernel_size,
                padding=padding,
                norm=norm,
                activation=activation,
            ),
            ConvBlock(
                in_channels=num_features,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                norm=norm,
                activation="none",
            ),
        )

        if out_channels != in_channels:
            self.resample = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                norm=norm,
                activation="none",
            )
        else:
            self.resample = None

        self.activation = get_activation_layer(
            activation_type=activation_out, num_channels=out_channels
        )

    def forward(self, x):
        residual = x
        out = self.model(x)
        if self.resample:
            residual = self.resample(residual)
        out += residual
        if self.activation:
            out = self.activation(out)
        return out


class SobelBlock(nn.Module):
    def __init__(self):
        super(SobelBlock, self).__init__()
        sobel_kernel = torch.tensor(
            [
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
            ],
            dtype=torch.float,
            requires_grad=False,
        )

        num_kernel = sobel_kernel.shape[0]
        sobel_kernel = torch.unsqueeze(sobel_kernel, dim=1)

        self.sobel_layer = nn.Conv2d(
            1,
            num_kernel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            padding_mode="reflect",
        )
        self.sobel_layer.weight.data = sobel_kernel

    def forward(self, x):
        return self.sobel_layer(x)


##################################################
# Network Modules
##################################################


class BaseModule(nn.Module):
    def __init__(self, args):
        super(BaseModule, self).__init__()
        self.parse_args(args)

    def parse_args(self, args):
        self.kernel_size = args.kernel_size
        self.stride = args.stride
        self.padding = args.padding
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.num_feature = args.num_feature
        self.num_blocks = args.num_blocks
        self.norm_type = args.norm_type
        self.activation_type = args.activation_type
        self.resblock_activation_out_type = args.resblock_activation_out_type


class DeepVIDv2(BaseModule):
    def __init__(self, args):
        super(DeepVIDv2, self).__init__(args)

        self.in_block = ConvBlock(
            in_channels=self.in_channels,
            out_channels=self.num_feature,
            kernel_size=self.kernel_size,
            padding=self.padding,
            norm="none",
            activation=self.activation_type,
        )

        self.model = nn.ModuleList()

        for _ in range(self.num_blocks):
            self.model.append(
                ResBlock(
                    in_channels=self.num_feature,
                    out_channels=self.num_feature,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    norm=self.norm_type,
                    activation=self.activation_type,
                    activation_out=self.resblock_activation_out_type,
                )
            )

        self.pre_out_block = ConvBlock(
            in_channels=self.num_feature,
            out_channels=self.num_feature,
            kernel_size=self.kernel_size,
            padding=self.padding,
            norm=self.norm_type,
            activation="none",
        )

        self.out_block = nn.Sequential(
            ConvBlock(
                in_channels=self.num_feature,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                norm="none",
                activation=self.activation_type,
            ),
            ConvBlock(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                padding=0,
                norm="none",
                activation="none",
            ),
        )

    def parse_args(self, args):
        super(DeepVIDv2, self).parse_args(args)

    def forward(self, x, encode=False):
        out = self.in_block(x)
        identity = out
        feat_list = []

        for i, block in enumerate(self.model):
            out = block(out)
            feat_list.append(out)

        out = self.pre_out_block(out)

        out += identity
        out = self.out_block(out)

        if encode:
            return out, feat_list
        else:
            return out
