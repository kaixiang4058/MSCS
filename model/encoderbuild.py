from torch import nn
from timm import create_model
from transformers import SegformerModel, SegformerConfig

# timm build
class timm(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.encoder = create_model(name, features_only=True, pretrained=True)

    def forward(self, inputs):
        return self.encoder(inputs)

    def hidden_size(self):
        return self.encoder.feature_info.channels()

# mit
class mit(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.encoder = SegformerModel.from_pretrained(
                name,config=SegformerConfig.from_pretrained(
                    name, output_hidden_states=True
                ))

    def forward(self, inputs):
        return list(self.encoder(inputs).hidden_states[-4:])

    def hidden_size(self):
        return self.encoder.config.hidden_sizes



# model building table
modeldict = {
    "nvidia/mit-b1" : mit, #segformer
    "resnest26d" : timm,
    "resnest50d" : timm,
    "resnet50d" : timm,
    "efficientnet_b3" : timm,
}


def encoder_select(encoder_name):
    if encoder_name in modeldict.keys():
       encoder = modeldict[encoder_name](encoder_name)
    else:
        raise KeyError(f"Key '{encoder_name}' not found in the model table.")
    return encoder