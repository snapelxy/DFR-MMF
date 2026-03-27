import torch
import torch.nn.functional as F


class UBlockUp(torch.nn.Module):
    def __init__(self, inChannels, inChannels_2, outChannels, useBilinear=False,with_bn_act=True):
        super(UBlockUp, self).__init__()
        self.con1 = torch.nn.Conv2d(
            in_channels=inChannels+inChannels_2, out_channels=outChannels, kernel_size=3, padding=1)

        self.con2 = torch.nn.Conv2d(
            in_channels=outChannels, out_channels=outChannels, kernel_size=3, padding=1)
        self.useBili = useBilinear
        if (useBilinear):
            pass
            # bilinear
            # self.xSpace = torch.linspace(-1, 1, height)
            # self.ySpace = torch.linspace(-1, 1, width)
            # self.msehX, self.meshY = torch.meshgrid((self.xSpace, self.ySpace))
            # self.grid = torch.stack((self.meshY, self.msehX), 2)
            # self.grid = torch.unsqueeze(self.grid, 0)
            # self.grid=self.grid.detach()
            # if (cuda_avi):
            # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
            # self.grid = self.grid.cuda(device=self.parameters().device)
        else:
           # stride 可以简单理解为放大倍数
           self.conTrans = torch.nn.ConvTranspose2d(in_channels=inChannels, out_channels=inChannels,
                                                    kernel_size=3, stride=2, padding=1, output_padding=1)
        self.b0 = torch.nn.BatchNorm2d(outChannels)

        # batchNorm  参数不需要训练
        # self.b1=torch.nn.BatchNorm2d(outChannels,affine=False)
        self.b2 = torch.nn.BatchNorm2d(outChannels)
        self.with_bn_act=with_bn_act

    def forward(self, xin1, xin2):
        if (self.useBili):
            # gridFinal = self.grid
            # for i in range((xin2.shape[0] - 1)):
            #     gridFinal = torch.cat((gridFinal, self.grid), 0)
            # x1 = F.grid_sample(xin1, gridFinal, align_corners=True)
            x1 = F.interpolate(input=xin1, scale_factor=2, mode='bilinear')
        else:
            x1 = self.conTrans(xin1)
        # 判断差值 并拼补

        if xin2 is not None:
            difx = x1.size(3)
            difx = xin2.size(3)-difx
            if (difx > 0):
                print(f"ublock x not match,padding...")
                difTx = torch.zeros((x1.size(0), x1.size(1), x1.size(2), difx))
                if (xin2.device.type == 'cuda'):
                    difTx = difTx.cuda()
                x1 = torch.cat((x1, difTx), dim=3)

            dify = x1.size(2)
            dify = xin2.size(2) - dify
            if (dify > 0):
                print(f"ublock y not match,padding...")
                difTy = torch.zeros((x1.size(0), x1.size(1), dify, x1.size(3)))
                if (xin2.device.type == 'cuda'):
                    difTy = difTy.cuda()
                x1 = torch.cat((x1, difTy), dim=2)

            x1 = torch.cat((x1, xin2), dim=1)
        x = self.con1(x1)
        ################################ BN#############
        x = self.b0(x)
        x = F.leaky_relu(x)
        x = self.con2(x)
        ################################ BN#############
        if self.with_bn_act:
            x = self.b2(x)
            x = F.leaky_relu(x)

        return x


class UPNet1_1(torch.nn.Module):
    def __init__(self,  useBilinear=False, **kargs):
        super(UPNet1_1, self).__init__()
        self.up1 = UBlockUp(inChannels=256, inChannels_2=1024, outChannels=768)
        self.up2 = UBlockUp(inChannels=768, inChannels_2=512, outChannels=384)
        self.up3 = UBlockUp(inChannels=384, inChannels_2=256, outChannels=192)
        self.up4 = UBlockUp(inChannels=192, inChannels_2=0, outChannels=96)
        self.up5 = UBlockUp(inChannels=96, inChannels_2=0, outChannels=3,with_bn_act=False)

    def gather_featrue(self, x_in, pos_idx, up_sacle):
        # tensor选取 用gather来实现
        b, c, h, w = x_in.shape
        f_roi = pow(2, up_sacle)
        p_h_idx = pos_idx[:, 0:1][:, :, None, None]*f_roi
        p_w_idx = pos_idx[:, 1:2][:, :, None, None]*f_roi
        p_h_idx_1 = p_h_idx.repeat((1, c, f_roi, w))
        p_add = torch.arange(f_roi).to(pos_idx.device)
        p_add_h = p_add[None, None, :, None].repeat((b, c, 1, w))
        p_h_idx_1 = p_h_idx_1 + p_add_h

        p_w_idx_1 = p_w_idx.repeat((1, c, f_roi, f_roi))
        p_add_w = p_add[None, None, None, :].repeat((b, c, f_roi, 1))
        p_w_idx_1 = p_w_idx_1+p_add_w

        x_s = x_in.gather(dim=2, index=p_h_idx_1)
        x_s = x_s.gather(dim=3, index=p_w_idx_1)
        return x_s

    def forward(self, x_in1, x_in2, pos_idx,**kwargs):
        x_s1 = self.gather_featrue(x_in2["2"], pos_idx, 1)
        x1 = self.up1(x_in1, x_s1)

        x_s2 = self.gather_featrue(x_in2["1"], pos_idx, 2)
        x2 = self.up2(x1, x_s2)

        x_s3 = self.gather_featrue(x_in2["0"], pos_idx, 3)
        x3 = self.up3(x2, x_s3)

        x4 = self.up4(x3, None)

        x5 = self.up5(x4, None)

        return x5


class UPNet1_2(torch.nn.Module):
    def __init__(self,  useBilinear=False, **kargs):
        super(UPNet1_2, self).__init__()
        self.up1 = UBlockUp(inChannels=256, inChannels_2=1024, outChannels=768)
        self.up2 = UBlockUp(inChannels=768, inChannels_2=512, outChannels=384)
        self.up3 = UBlockUp(inChannels=384, inChannels_2=256, outChannels=192)
        self.up4 = UBlockUp(inChannels=192, inChannels_2=0, outChannels=96)
        self.up5 = UBlockUp(inChannels=96, inChannels_2=0, outChannels=48,with_bn_act=True)
        self.conv_out1=torch.nn.Conv2d(in_channels=48+3,out_channels=32,kernel_size=3,padding=1)
        self.bn1=torch.nn.BatchNorm2d(32)
        self.conv_out2=torch.nn.Conv2d(in_channels=32,out_channels=3,kernel_size=1)

    def gather_featrue(self, x_in, pos_idx, up_sacle):
        # tensor选取 用gather来实现
        b, c, h, w = x_in.shape
        f_roi = pow(2, up_sacle)
        p_h_idx = pos_idx[:, 0:1][:, :, None, None]*f_roi
        p_w_idx = pos_idx[:, 1:2][:, :, None, None]*f_roi
        p_h_idx_1 = p_h_idx.repeat((1, c, f_roi, w))
        p_add = torch.arange(f_roi).to(pos_idx.device)
        p_add_h = p_add[None, None, :, None].repeat((b, c, 1, w))
        p_h_idx_1 = p_h_idx_1 + p_add_h

        p_w_idx_1 = p_w_idx.repeat((1, c, f_roi, f_roi))
        p_add_w = p_add[None, None, None, :].repeat((b, c, f_roi, 1))
        p_w_idx_1 = p_w_idx_1+p_add_w

        x_s = x_in.gather(dim=2, index=p_h_idx_1)
        x_s = x_s.gather(dim=3, index=p_w_idx_1)
        return x_s

    def forward(self, x_in1, x_in2, pos_idx,x_tb,**kwargs):
        x_s1 = self.gather_featrue(x_in2["2"], pos_idx, 1)
        x1 = self.up1(x_in1, x_s1)

        x_s2 = self.gather_featrue(x_in2["1"], pos_idx, 2)
        x2 = self.up2(x1, x_s2)

        x_s3 = self.gather_featrue(x_in2["0"], pos_idx, 3)
        x3 = self.up3(x2, x_s3)

        x4 = self.up4(x3, None)

        x5 = self.up5(x4, None)
        x5=torch.cat([x5,x_tb],dim=1)
        out = self.conv_out1(x5)
        out=self.bn1(out)
        out=F.relu(out)
        out=self.conv_out2(out)

        return out


class UPNet1_3(torch.nn.Module):
    def __init__(self,  useBilinear=False, **kargs):
        super(UPNet1_3, self).__init__()
        hidd_dims=kargs["hidd_dims"]
        self.up1 = UBlockUp(inChannels=hidd_dims[0][0], inChannels_2=hidd_dims[0][1], outChannels=hidd_dims[0][2])
        self.up2 = UBlockUp(inChannels=hidd_dims[1][0], inChannels_2=hidd_dims[1][1], outChannels=hidd_dims[1][2])
        self.up3 = UBlockUp(inChannels=hidd_dims[2][0], inChannels_2=hidd_dims[2][1], outChannels=hidd_dims[2][2])
        self.up4 = UBlockUp(inChannels=hidd_dims[3][0], inChannels_2=hidd_dims[3][1], outChannels=hidd_dims[3][2])
        self.up5 = UBlockUp(inChannels=hidd_dims[4][0], inChannels_2=hidd_dims[4][1], outChannels=hidd_dims[4][2],with_bn_act=True)
        self.conv_out1=torch.nn.Conv2d(in_channels=hidd_dims[4][2]+3,out_channels=32,kernel_size=3,padding=1)
        self.bn1=torch.nn.BatchNorm2d(32)
        self.conv_out2=torch.nn.Conv2d(in_channels=32,out_channels=3,kernel_size=1)

    def forward(self, x_in1, x_in2, x_tb,**kwargs):
        if x_in2==None:
            x1 = self.up1(x_in1,None)

            x2 = self.up2(x1, None)

            x3 = self.up3(x2,None)
        else:
            x1 = self.up1(x_in1, x_in2[0])

            x2 = self.up2(x1, x_in2[1])

            x3 = self.up3(x2, x_in2[2])

        x4 = self.up4(x3, None)

        x5 = self.up5(x4, None)
        x5=torch.cat([x5,x_tb],dim=1)
        out = self.conv_out1(x5)
        out=self.bn1(out)
        out=F.relu(out)
        out=self.conv_out2(out)

        return out


if __name__ == "__main__":
    x_in2 = {
        "0": torch.rand(2, 256, 256, 64),
        "1": torch.rand(2, 512, 128, 32),
        "2": torch.rand(2, 1024, 64, 16),
        "3": torch.rand(2, 2048, 32, 8),
    }
    x_in1 = torch.rand(2, 256, 32, 8)
    pos_idx = torch.ones((2, 2)).to(torch.int64)
    pos_idx[0, 0] = 16
    pos_idx[0, 1] = 4
    m = UPNet1_3()
    imgblur=torch.randn(2, 3, 1024, 256)
    out = m(x_in1, x_in2, imgblur)

    # from torchstat import stat
    # # #SummaryModel(m)
    # # # from thop import profile
    # # # fs,ps=profile(m,(img,))
    # stat(m,(3,960,960))
    # y=m(img)
    # print('qw')
