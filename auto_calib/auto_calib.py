import torch

class AutoCalib(torch.nn.Module):
    def __init__(self, **kargs):
        self.layer1 = torch.nn.Linear(2,32)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.layer2 = torch.nn.Linear(32,32)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.layer3 = torch.nn.Linear(32,32)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.layer4 = torch.nn.Linear(32,2)

    def forward(self,x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.layer4(x)
        
        return x


if __name__ == '__main__':
    
    x1 = torch.randn((1,3,256,256)) #.cuda()
    model = AutoCalib()
    out = model(x1)
    
    print("finished")