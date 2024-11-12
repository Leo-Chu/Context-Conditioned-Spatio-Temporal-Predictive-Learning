import torch
import torch.nn as nn

class temporal_atten1(nn.Module):
    def __init__(self, channel, reduction=16):
        super(temporal_atten1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class temporal_atten2(nn.Module):
    def __init__(self, channel, reduction=16):
        super(temporal_atten2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes// ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class STAttention(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(STAttention, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class CausalLSTMCell(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm, reduction=16):
        super(CausalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.tem_attenx = temporal_atten1(num_hidden * 7, reduction)
        self.tem_attenh = temporal_atten2(num_hidden * 4, reduction)
        self.tem_attenc = temporal_atten2(num_hidden * 3, reduction)
        self.st_atten = STAttention(num_hidden * 3)

        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_c = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, height, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
            self.conv_c2m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_om = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_c = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_c2m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_om = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.tem_attenx(self.conv_x(x_t))
        h_concat = self.tem_attenh(self.conv_h(h_t))
        c_concat = self.tem_attenc(self.conv_c(c_t))
        m_concat = self.st_atten(self.conv_m(m_t))
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = \
            torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, m_m = torch.split(m_concat, self.num_hidden, dim=1)
        i_c, f_c, g_c = torch.split(c_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h + i_c)
        f_t = torch.sigmoid(f_x + f_h + f_c + self._forget_bias)
        g_t = torch.tanh(g_x + g_h + g_c)

        c_new = f_t * c_t + i_t * g_t

        c2m = self.conv_c2m(c_new)
        i_c, g_c, f_c, o_c = torch.split(c2m, self.num_hidden, dim=1)

        i_t_prime = torch.sigmoid(i_x_prime + i_m + i_c)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + f_c + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_c)

        m_new = f_t_prime * torch.tanh(m_m) + i_t_prime * g_t_prime
        o_m = self.conv_om(m_new)

        o_t = torch.tanh(o_x + o_h + o_c + o_m)
        mem = torch.cat((c_new, m_new), 1)
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


class GHU(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size,
                 stride, layer_norm, initializer=0.001):
        super(GHU, self).__init__()

        self.filter_size = filter_size
        self.padding = filter_size // 2
        self.num_hidden = num_hidden
        self.layer_norm = layer_norm

        if layer_norm:
            self.z_concat = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 2, height, width])
            )
            self.x_concat = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 2, height, width])
            )
        else:
            self.z_concat = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.x_concat = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )

        if initializer != -1:
            self.initializer = initializer
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            nn.init.uniform_(m.weight, -self.initializer, self.initializer)

    def _init_state(self, inputs):
        return torch.zeros_like(inputs)

    def forward(self, x, z):
        if z is None:
            z = self._init_state(x)
        z_concat = self.z_concat(z)
        x_concat = self.x_concat(x)

        gates = x_concat + z_concat
        p, u = torch.split(gates, self.num_hidden, dim=1)
        p = torch.tanh(p)
        u = torch.sigmoid(u)
        z_new = u * p + (1-u) * z
        return z_new


class RNN(nn.Module):
    r"""PredRNN++ Model

    Implementation of `PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma
    in Spatiotemporal Predictive Learning <https://arxiv.org/abs/1804.06300>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(RNN, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()
        
        self.gradient_highway = GHU(num_hidden[0], num_hidden[0], height, width,
                                    configs.filter_size, configs.stride, configs.layer_norm)

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                CausalLSTMCell(in_channel, num_hidden[i], height, width,
                               configs.filter_size, configs.stride, configs.layer_norm))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        device = frames_tensor.device
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [batch, self.num_hidden[i], height, width], device=device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros(
            [batch, self.num_hidden[0], height, width], device=device)
        z_t = None

        for t in range(self.configs.total_length - 1):
            # reverse schedule sampling
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs.input_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                          (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)
            z_t = self.gradient_highway(h_t[0], z_t)
            h_t[1], c_t[1], memory = self.cell_list[1](z_t, h_t[1], c_t[1], memory)

            for i in range(2, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None

        return next_frames, loss
