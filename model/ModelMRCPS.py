import torch
from .MSUnetHub import MSUnetHub

class ModelMRCPS():
    def __init__(self, lrscale=8):
        super().__init__()
        #Resnest + Segformer (CNN branch)
        self.branch1 = MSUnetHub(encoder_name="resnest26d", lrbackbone="nvidia/mit-b1", lrscale=lrscale)
        #Segformer + Segformer (Transformer branch)
        self.branch2 = MSUnetHub(encoder_name="nvidia/mit-b1", lrbackbone="nvidia/mit-b1", lrscale=lrscale)
    
    def forward(self, x, lrx, step=1):
        # for inference output not suitable for training (traning need used branch output)
        """
        Args:
            x       input tensor
            step    predict branch
        """
        if step == 0:
            return torch.argmax(
                self.branch1(x, lrx).softmax(1) + self.branch2(x, lrx).softmax(1)
                , dim=1)
        elif step == 3:
            p1 = self.branch1(x, lrx).softmax(1)
            p2 = self.branch2(x, lrx).softmax(1)
            return torch.argmax(p1+p2, dim=1), torch.argmax(p1, dim=1), torch.argmax(p2, dim=1)
        else:
            return torch.argmax(getattr(self, f'branch{step}')(x, lrx), dim=1)

