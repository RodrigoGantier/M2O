import torch 
import torch.nn.functional as F
from torch import nn


def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)

def skip_sum(x1, x2):
    return x1 + x2


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu'):
        super(ConvLayer, self).__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.activation = None if activation is None else nn.ReLU()

    def forward(self, x):
        out = self.conv2d(x)
        if self.activation is not None:
            out = self.activation(out)
        return out
    

class ConvLSTM_s(nn.Module):
    """Adapted from: https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTM_s, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)
        self.state = None

    def forward(self, input_):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty state, if None is provided
        if self.state is None:

            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[state_size] = (
                    torch.zeros(state_size, dtype=input_.dtype).to(input_),
                    torch.zeros(state_size, dtype=input_.dtype).to(input_)
                )

            self.state = self.zero_tensors[tuple(state_size)]
        prev_hidden, prev_cell = self.state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)
        self.state = hidden, cell

        return hidden
    

class RecurrentConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                stride=1, padding=0, activation='relu'):
        super(RecurrentConvLSTM, self).__init__()
        
        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation)
        self.recurrent_block = ConvLSTM_s(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

    def forward(self, x):
        x = self.conv(x)
        state = self.recurrent_block(x)
        return state


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out
    

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(UpsampleConvLayer, self).__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.activation = nn.ReLU()

    def forward(self, x):
        x_upsampled = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)
        out = self.activation(out)
        return out
    

class UConvLSTM(nn.Module):
    """
    UConvLSTM architecture where every encoder is followed by a recurrent convolutional block,
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, args, channel_multiplier=2):
        super().__init__()
        self.num_bins = args.num_bins
        self.base_num_channels = args.base_num_channels
        self.kernel_size = args.kernel
        self.num_residual_blocks = args.num_residual_blocks
        self.num_encoders = args.num_encoders
        self.num_output_channels = args.num_output_channels

        self.encoder_input_sizes = [int(self.base_num_channels * pow(channel_multiplier, i)) for i in range(self.num_encoders)]
        self.encoder_output_sizes = [int(self.base_num_channels * pow(channel_multiplier, i + 1)) for i in range(self.num_encoders)]
        self.max_num_channels = self.encoder_output_sizes[-1]

        self.head = ConvLayer(self.num_bins, self.base_num_channels,
                              kernel_size=self.kernel_size, stride=1,
                              padding=self.kernel_size // 2)  # N x C x H x W 

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLSTM(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2))

        self.build_resblocks()
        self.decoders = self.build_decoders()
        self.pred = self.build_prediction_layer(num_output_channels=args.num_output_channels)

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels))
    
    def build_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for input_size, output_size in zip(decoder_input_sizes, decoder_output_sizes):
            decoders.append(UpsampleConvLayer(
                input_size, output_size, kernel_size=self.kernel_size,
                padding=self.kernel_size // 2))
        return decoders
    
    def build_prediction_layer(self, num_output_channels):
        return ConvLayer(self.base_num_channels,
                         num_output_channels, 1, activation=None)

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(skip_sum(x, blocks[self.num_encoders - i - 1]))

        # tail
        img = self.pred(skip_sum(x, head))
        
        return img