from vedacore.misc import registry
import torch
import torch.nn.functional as F
import math
# import sys
# sys.path.append("/home/cjj/workspace/vedat-dev_mask/")


@registry.register_module('common_model')
class SpConAtt(torch.nn.Module):

    def __init__(
        self, in_channels, out_channels, out_kernel=7, pos_dim=4,**kwargs
    ) -> None:
        super(SpConAtt, self).__init__()

        self.out_channels = out_channels
        
        self.pos_dim = pos_dim
        self.conv_q = torch.nn.Conv2d(in_channels, out_channels-self.pos_dim, kernel_size=1, stride=1, padding=0,
                                      bias=True)
        self.conv_k = torch.nn.Conv2d(in_channels, out_channels-self.pos_dim, kernel_size=1, stride=1, padding=0,
                                      bias=True)
        self.conv_v = torch.nn.Conv2d(in_channels, out_channels-self.pos_dim, kernel_size=1, stride=1, padding=0,
                                      bias=True)
        self.dk_sqrt = math.sqrt(float(out_channels))

        self.out_kernel = out_kernel

        self.hw_pose = None

        self.conv_pose = torch.nn.Conv2d(2, self.pos_dim, kernel_size=1, stride=1, padding=0,
                                         bias=True)
        self.select = kwargs["select"] if  "select" in kwargs.keys() else True
        self.con_out = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=self.out_kernel, stride=self.out_kernel, padding=0, bias=True)

    def get_pos(self, x):
        b, c, h, w = x.shape
        h_pose = torch.ones((h))
        h_pose = h_pose.cumsum(dim=-1)-1.0
        h_pose = h_pose[None, None, :, None]
        h_pose = h_pose.repeat((1, 1, 1, w))

        w_pose = torch.ones((w))
        w_pose = w_pose.cumsum(dim=-1)-1.0
        w_pose = w_pose[None, None, None, :]
        w_pose = w_pose.repeat((1, 1, h, 1))

        hw_pose = torch.cat((h_pose, w_pose), dim=1)
        hw_pose = hw_pose.repeat((b, 1, 1, 1))
        self.hw_pose = hw_pose.to(x.device)
        return

    def forward(self, x):
        # 位置编码
        if self.hw_pose == None:
            self.get_pos(x)
        pos = self.conv_pose(self.hw_pose[0:x.shape[0],...])

        # b,c,h,w
        b, c, h, w = x.shape
        qs = self.conv_q(x)
        qs = torch.cat((qs, pos), dim=1)
        qs = torch.permute(qs, (0, 2, 3, 1)).contiguous()
        qs = qs.view((b, h*w, self.out_channels))

        ks = self.conv_k(x)
        ks = torch.cat((ks, pos), dim=1)
        ks = ks.view((b, self.out_channels, h*w))

        vs = self.conv_v(x)
        vs = torch.cat((vs, pos), dim=1)
        vs = torch.permute(vs, (0, 2, 3, 1)).contiguous()
        if self.select:
            vs = vs.view((b, h, w, self.out_channels))
        else:
            vs = vs.view((b, h*w, self.out_channels))

        q_k = torch.matmul(qs, ks)/self.dk_sqrt

        # 自身注意力点赋零
        q_k[:, torch.arange(h*w), torch.arange(h*w)] = 0

        q_k = F.softmax(q_k, dim=-1)
        if self.select:
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
        else:
            qkv = torch.matmul(q_k,vs)
            qkv = qkv.view((b, h, w, self.out_channels)).contiguous()
            qkv = qkv.permute((0, 3, 1, 2)).contiguous()
            return qkv

@registry.register_module('common_model')
class ChanConAtt(torch.nn.Module):

    def __init__(
        self, in_channels, out_channels, out_kernel=7,
        **kwargs
    ) -> None:
        super(ChanConAtt, self).__init__()

        self.out_channels = out_channels

        self.conv_q = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                      bias=True)
        self.conv_k = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                      bias=True)
        self.conv_v = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                      bias=True)
        self.dk_sqrt = math.sqrt(float(out_channels))

        self.out_kernel = out_kernel
        self.select = kwargs["select"] if  "select" in kwargs.keys() else True
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
        if self.select:
            vs = vs.view((b, h, w, self.out_channels))
        else:
            vs = vs.view((b, h*w, self.out_channels))

        q_k = torch.matmul(qs, ks)/self.dk_sqrt

        # 自身注意力点赋零
        q_k[:, torch.arange(h*w), torch.arange(h*w)] = 0

        q_k = F.softmax(q_k, dim=-1)

        if self.select:
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
        else:
            qkv = torch.matmul((q_k,vs))
            qkv = qkv.view((b, h, w, self.out_channels)).contiguous()
            qkv = qkv.permute(qkv, (0, 3, 1, 2)).contiguous()
            return qkv


@registry.register_module('common_model')
class AttM_1(torch.nn.Module):

    def __init__(
        self, in_channels, out_channels, out_kernel, **kargs
    ) -> None:
        super(AttM_1, self).__init__()
        self.pos_dim = kargs["pos_dim"] if "pos_dim" in kargs.keys() else 4
        select_list = kargs["select"] if "select" in kargs.keys() else [True,True,True]
        if in_channels[0] == 0 :
            self.att_0 = torch.nn.Identity()
        else:
            self.att_0 = SpConAtt(
            in_channels[0], out_channels[0],pos_dim=self.pos_dim, out_kernel=out_kernel[0],select=select_list[0])
        if in_channels[1] == 0 :
            self.att_1 = torch.nn.Identity()
        else:
            self.att_1 = SpConAtt(
            in_channels[1], out_channels[1],pos_dim=self.pos_dim, out_kernel=out_kernel[1],select=select_list[1])
        if in_channels[2] == 0 :
            self.att_2 = torch.nn.Identity()
        else:
            self.att_2 = SpConAtt(
            in_channels[2], out_channels[2],pos_dim=self.pos_dim, out_kernel=out_kernel[2],select=select_list[2])

    def forward(self, x):
        # b,c,h,w
        y = []
        if type(self.att_0) is torch.nn.Identity:
            y.append(x[0])
        else:    
            x_0 = self.att_0(x[0])
            y.append(torch.cat((x[0], x_0), dim=1))
        
        if type(self.att_1) is torch.nn.Identity:
            y.append(x[1])
        else:   
            x_1 = self.att_1(x[1])
            y.append(torch.cat((x[1], x_1), dim=1))
        if type(self.att_2) is torch.nn.Identity:
            y.append(x[2])
        else:   
            x_2 = self.att_2(x[2])
            y.append(torch.cat((x[2], x_2), dim=1))
        return y



if __name__ == '__main__':

    x = torch.randn((2, 3, 64, 28))

    model = SpConAtt(3, 8, 7)

    for i in range(100):
        out = model(x)

    print(out.shape)