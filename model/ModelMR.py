import torch
from .MSUnetHub import MSUnetHub

class ModelMRCNN():
    def __init__(self, lrscale=8):
        super().__init__()
        #Resnest + Segformer (CNN branch)
        self.model = MSUnetHub(encoder_name="resnest26d", lrbackbone="nvidia/mit-b1", lrscale=lrscale)

    def forward(self, x, lrx):
        # for inference output not suitable for training
        # (traning need used model output)
        predmask = self.model(x, lrx)
        return torch.argmax(predmask, dim=1)

class ModelMRTrans():
    def __init__(self, lrscale=8):
        super().__init__()
        #Segformer + Segformer (Transformer branch)
        self.model = MSUnetHub(encoder_name="nvidia/mit-b1", lrbackbone="nvidia/mit-b1", lrscale=lrscale)
    
    def forward(self, x, lrx):
        # for inference output not suitable for training
        # (traning need used self.model output)
        predmask = self.model(x, lrx)
        return torch.argmax(predmask, dim=1)

