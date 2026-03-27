from vedacore.misc import registry
import torch



@registry.register_module('common_model')
class ProjectionNet(torch.nn.Module):
    def __init__(self, in_channels,**kargs):
        super(ProjectionNet, self).__init__()
        self.modules_list=[]
        for i in range(len(in_channels)):
            setattr(self,"l"+str(i),torch.nn.Conv2d(in_channels=in_channels[i], out_channels=in_channels[i],kernel_size=1))
            self.modules_list.append("l"+str(i))
    
    def forward(self,inputs):
        assert len(self.modules_list) == len(inputs)
        out = []
        for m,inp in zip(self.modules_list,inputs):
            obj = getattr(self,m)
            out.append(obj(inp))
        return out

if __name__ == '__main__':

    x = torch.randn((2, 3, 64, 28))

