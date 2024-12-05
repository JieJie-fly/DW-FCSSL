import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        if x.dim() == 4 and x.size(2) == 1:
            x = x.squeeze(2)
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)

        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):  # (64,24,3)
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i - 1] if i > 0 else in_channels, channels[i], kernel_size=kernel_size, dilation=2 ** i, final=(i == len(channels) - 1))
            for i in range(len(channels))
        ])

    def forward(self, x):
        return self.net(x)


class BertInterpHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.activation = nn.ReLU()
        self.project = nn.Linear(4 * hidden_dim, input_dim)

    def forward(self, first_token_tensor):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.project(pooled_output)
        return pooled_output

class Encoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims, depth=5):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.input_fc = nn.Linear(input_dims, hidden_dims)

        self.feature_extractor = DilatedConvEncoder(
            # input_dims,
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        self.interphead = BertInterpHead(input_dims, output_dims)


    def forward(self, x1, x2):
        x1.to(device)
        x2.to(device)
        if self.training:
            x_whole = self.input_fc(x1.float())
            x_whole = x_whole.transpose(1, 2)
            x_whole = self.feature_extractor(x_whole)
            x_whole = x_whole.transpose(1, 2)
            x_whole = self.repr_dropout(x_whole)

        # recon mask part
        if self.training:
            x_interp = self.input_fc(x2.float())
            x_interp = x_interp.transpose(1, 2)
            x_interp = self.feature_extractor(x_interp)
            x_interp = x_interp.transpose(1, 2)
            x_interp = self.repr_dropout(x_interp)

        x = self.input_fc(x1.float())
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.repr_dropout(x)

        if self.training:

            x_recon = self.interphead(x_interp)
            return x_whole, x_recon
        else:
            return x, self.interphead(x)


# 4*4*64
class FCSSL(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims):
        super(FCSSL, self).__init__()
        self.extractor = Encoder(input_dims, output_dims, hidden_dims)
        self.flattern = nn.Flatten(1)
        self.predictor = nn.Sequential()
        self.predictor.add_module('1', nn.Linear(output_dims * 10, 64))
        # self.predictor.add_module('2', nn.Dropout(0.8))
        self.predictor.add_module('3', nn.Linear(64, 16))
        self.predictor.add_module('2', nn.ReLU())
        self.predictor.add_module('4', nn.Linear(16, 1))

    def forward(self, x, x2):
        # self.extractor.eval()
        if self.extractor.eval():
            out1, recon_fea = self.extractor(x, x2)
            out2 = self.flattern(out1)
            out3 = self.predictor(out2)

        else:
            out1, recon_fea = self.extractor(x, x2)
            out2 = self.flattern(out1)
            out3 = self.predictor(out2)

        return out3, out1, recon_fea