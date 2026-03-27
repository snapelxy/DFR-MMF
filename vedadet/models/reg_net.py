import copy
import torch
from torchvision.models import regnet_x_1_6gf,regnet_y_16gf,regnet_y_800mf,regnet_x_800mf,RegNet_Y_800MF_Weights,regnet_x_3_2gf,regnet_y_3_2gf
from vedacore.misc import registry

reg_net_dict = {
    "regnet_x_800mf":regnet_x_800mf,
    "regnet_y_800mf":regnet_y_800mf,
    "regnet_x_1_6gf":regnet_x_1_6gf,
    "regnet_y_16gf":regnet_y_16gf,
    "regnet_x_3_2gf":regnet_x_3_2gf,
    "regnet_y_3_2gf":regnet_y_3_2gf
}


@registry.register_module('common_model')
class regnet(torch.nn.Module):

    def __init__(
        self,regnet_type,pretrained=True,**kargs

    ) -> None:
        super(regnet, self).__init__()
        
        model_pre = reg_net_dict[regnet_type](pretrained=pretrained)
        self.stem = copy.deepcopy(model_pre.stem)
        self.block_1 = copy.deepcopy(model_pre.trunk_output[0])
        self.block_2 = copy.deepcopy(model_pre.trunk_output[1])
        self.block_3 = copy.deepcopy(model_pre.trunk_output[2])
        self.block_4 = copy.deepcopy(model_pre.trunk_output[3])


    def forward(self, x):
        # b,c,h,w
        x = self.stem(x)
        out1 = self.block_1(x)
        out2 = self.block_2(out1)
        out3 = self.block_3(out2)
        out4 = self.block_4(out3)

        return [out2,out3,out4]


if __name__ == '__main__':
    
    x1 = torch.randn((1,3,256,256)) #.cuda()
    model = regnet("regnet_y_800mf",weights=RegNet_Y_800MF_Weights.IMAGENET1K_V2)
    out = model(x1) 
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save(r"C:\Pro\yuanAI\configs\regnet_y_800mf.pt")
    
    print("finished")
    