import os
import numpy as np
import torch
from torch import nn
from torchsummary import summary
from model.ModelMRCPS import ModelMRCPS


def main(preweight):
    ## model prepare
    model = ModelMRCPS()
    if os.path.isfile(preweight):
        state_dict = torch.load(preweight, map_location=device)
        model.load_state_dict(state_dict)

    #weight確認
    model_var_dict = model.state_dict()
    all_keys = model_var_dict.keys()
    print(model_var_dict)
    print(all_keys)

    branch1_keys = [k for k in all_keys if k.startswith('branch1')]
    print(branch1_keys)
    


if __name__ == '__main__':
    preweight=""
    main(preweight)