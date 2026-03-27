import torch
from torchvision.models import mobilenetv3
from torchvision.models._utils import IntermediateLayerGetter
from vedacore.misc import registry

class MobNetv3(torch.nn.Module):

    def __init__(
        self,pretrained=True,**kargs

    ) -> None:
        super(MobNetv3, self).__init__()
        
        backbone = mobilenetv3.__dict__["mobilenet_v3_large"](pretrained=pretrained, dilated=True).features

        return_layers={
            "3": "l_1",
            "6": "l_2",
            "16": "l_3"
        }
        
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)


    def forward(self, x):
        # b,c,h,w
        out = self.backbone(x)

        return [out["l_1"],out["l_2"],out["l_3"]]


if __name__ == '__main__':
    
    x1 = torch.randn((1,3,256,256)) #.cuda()
    
    model = MobNetv3(pretrained=True)
    out = model(x1)
    
    print("finished")
    