from utils import transformsgpu

def _strongTransform(parameters, data=None, lrdata=None, target=None,
                    cutaug = transformsgpu.cutout, coloraug = transformsgpu.colorJitter, flipaug = transformsgpu.flip,
                    isaugsym=True):
    assert ((data is not None) or (target is not None))
    data, target = cutaug(mask = parameters.get("Cut", None), data = data, target = target)
    lrdata, _ = cutaug(mask = parameters.get("LRCut", None), data = lrdata)
    data, lrdata, target = coloraug(
        colorJitter=parameters["ColorJitter"], data=data, lrdata=lrdata, target=target,
        Threshold=0.4,saturation=0.04, hue=0.08, issymetric=isaugsym
        )
    data, lrdata, target = flipaug(flip=parameters["flip"], data=data, lrdata=lrdata, target=target)

    if not (lrdata is None):
        return data, lrdata, target
    else:
        return data, target