import torch
import torch.nn.functional as F
import math
from vedacore.misc import registry


@registry.register_module('common_model')
class SpConAtt(torch.nn.Module):

    def __init__(
        self, in_channels, out_channels, out_kernel=7

    ) -> None:
        super(SpConAtt, self).__init__()

        self.out_channels = out_channels

        self.conv_q = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                      bias=True)
        self.conv_k = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                      bias=True)
        self.conv_v = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                      bias=True)
        self.dk_sqrt = math.sqrt(float(out_channels))

        self.out_kernel = out_kernel

        self.con_out = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=self.out_kernel, stride=self.out_kernel, padding=0, bias=True)

    def forward(self, x):
        # b,c,h,w
        b, c, h, w = x.shape
        qs = self.conv_q(x)
        qs = torch.permute(qs, (0, 2, 3, 1)).contiguous()
        qs = qs.view((b, h*w, self.out_channels))

        ks = self.conv_k(x)
        ks = ks.view((b, self.out_channels, h*w))

        vs = self.conv_v(x)
        vs = torch.permute(vs, (0, 2, 3, 1)).contiguous()
        vs = vs.view((b, h, w, self.out_channels))

        q_k = torch.matmul(qs, ks)/self.dk_sqrt

        # 自身注意力点赋零
        q_k[:, torch.arange(h*w), torch.arange(h*w)] = 0

        q_k = F.softmax(q_k, dim=-1)

        # 节省显存 仅对 对应的卷积区域进行计算
        q_k = q_k.view((b, h, w, h, w))
        q_k_select = []
        vs_select = []
        for h_i in range(-int(self.out_kernel/2), int(self.out_kernel/2)+1):
            for w_i in range(-int(self.out_kernel/2), int(self.out_kernel/2)+1):
                q_k_s_h = q_k[:, torch.arange(h), :, torch.clamp(
                    torch.arange(h)+h_i, 0, h-1), :]
                q_k_s_hw = q_k_s_h[:, :, torch.arange(
                    w), torch.clamp(torch.arange(w)+w_i, 0, w-1)]
                q_k_s_hw = torch.permute(q_k_s_hw, (1, 0, 2)).contiguous()
                q_k_s_hw = q_k_s_hw.view((b, h, w, 1))
                q_k_select.append(q_k_s_hw)

                vs_s_h = vs[:, torch.clamp(torch.arange(h)+h_i, 0, h-1), :, :]
                vs_s_hw = vs_s_h[:, :, torch.clamp(torch.arange(
                    w)+w_i, 0, w-1), :].view((b, h, w, 1, self.out_channels))
                vs_select.append(vs_s_hw)
        q_k_select = torch.cat(q_k_select, dim=3)[..., None]
        vs_select = torch.cat(vs_select, dim=3)
        qkv = q_k_select*vs_select
        qkv = qkv.view((b, h, w, self.out_kernel, self.out_kernel,
                       self.out_channels)).contiguous()
        qkv = torch.permute(qkv, (0, 1, 2, 5, 3, 4)).contiguous()
        qkv = qkv.view((b*h*w, self.out_channels,
                       self.out_kernel, self.out_kernel))
        # 对注意力结果进行卷积输出

        qkv_out = self.con_out(qkv)
        qkv_out = F.relu(qkv_out)
        qkv_out = qkv_out.view((b, h, w, self.out_channels))
        out = torch.permute(qkv_out, (0, 3, 1, 2)).contiguous()
        # # 筛选自身点的数值
        # qkv_out_h = qkv_out[:,torch.arange(h),:,:,torch.arange(h),:]

        # qkv_out_w = qkv_out_h[:,:,torch.arange(w),:,torch.arange(w)]

        # out = qkv_out_w.permute((2,3,1,0)).contiguous()

        return out


@registry.register_module('common_model')
class AttM_1(torch.nn.Module):

    def __init__(
        self, in_channels, out_channels, out_kernel, **kargs
    ) -> None:
        super(AttM_1, self).__init__()
        self.placeholder = kargs["placeholder"] if "placeholder" in kargs.keys() else False
        if self.placeholder == False:
            self.att_0 = SpConAtt(
                in_channels[0], out_channels[0], out_kernel=out_kernel[0])

            self.att_1 = SpConAtt(
                in_channels[1], out_channels[1], out_kernel=out_kernel[1])

            self.att_2 = SpConAtt(
                in_channels[2], out_channels[2], out_kernel=out_kernel[2])

    def forward(self, x):
        # b,c,h,w
        if self.placeholder:
            return x
        else:
            y = []
            x_0 = self.att_0(x[0])
            y.append(torch.cat((x[0], x_0), dim=1))
            x_1 = self.att_1(x[1])
            y.append(torch.cat((x[1], x_1), dim=1))
            x_2 = self.att_2(x[2])
            y.append(torch.cat((x[2], x_2), dim=1))

            return y


if __name__ == '__main__':

    x = torch.randn((2, 3, 64, 28))

    model = SpConAtt(3, 8, 7)

    out = model(x)

    print(out.shape)
