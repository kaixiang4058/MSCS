import random
import torch
from utils.training_package import _strongTransform
import torch.nn.functional as F

def dice_loss(logits, targets, eps=1e-6):
    """
    logits: (B, 1, H, W)  raw output (no sigmoid)
    targets: (B, H, W) or (B, 1, H, W)  {0,1}
    """
    num_classes = logits.shape[1]

    probs = F.softmax(logits, dim=1)

    targets_onehot = F.one_hot(targets, num_classes=num_classes)
    targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets_onehot, dims)
    union = torch.sum(probs + targets_onehot, dims)

    dice = (2. * intersection + eps) / (union + eps)

    return 1 - dice.mean()

    # probs = torch.sigmoid(logits)
    # if targets.dim() == 3:
    #     targets = targets.unsqueeze(1)

    # probs = probs.contiguous().view(probs.size(0), -1)
    # targets = targets.contiguous().view(targets.size(0), -1)

    # intersection = (probs * targets).sum(dim=1)
    # dice = (2. * intersection + eps) / \
    #        (probs.sum(dim=1) + targets.sum(dim=1) + eps)

    # return 1 - dice.mean()


def compute_supervised_loss(model, batch, criterion, loss_record_temp, dice_weight=1.0):
    image, mask, lrimage = batch

    y_pred_1_sup = model.branch1(image, lrimage)
    y_pred_2_sup = model.branch2(image, lrimage)
    
    ce_1  = criterion(y_pred_1_sup, mask)
    ce_2  = criterion(y_pred_2_sup, mask)

    dice_1 = dice_loss(y_pred_1_sup, mask)
    dice_2 = dice_loss(y_pred_2_sup, mask)
    
    sup_loss_1 = ce_1 + dice_weight * dice_1
    sup_loss_2 = ce_2 + dice_weight * dice_2

    loss_record_temp['b1_sup_ce'].append(ce_1.item())
    loss_record_temp['b2_sup_ce'].append(ce_2.item())
    loss_record_temp['b1_sup_dice'].append(dice_1.item())
    loss_record_temp['b2_sup_dice'].append(dice_2.item())
    loss_record_temp['b1_sup'].append(sup_loss_1.item())
    loss_record_temp['b2_sup'].append(sup_loss_2.item())

    return sup_loss_1 + sup_loss_2
    
def compute_consistency_loss(model, batch, criterion, loss_record_temp):
    image, lrimage = batch
    
    with torch.no_grad():
        y_pred_un_1 = model.branch1(image, lrimage)
        y_pred_un_2 = model.branch2(image, lrimage)
        pseudomask_un_1 = torch.argmax(y_pred_un_1, dim=1)
        pseudomask_un_2 = torch.argmax(y_pred_un_2, dim=1)

        pseudomask_cat = torch.cat(\
                    (torch.unsqueeze(pseudomask_un_1, dim=1), torch.unsqueeze(pseudomask_un_2, dim=1)), dim=1)
                
        strong_parameters = {}
        strong_parameters["flip"] = random.randint(0, 7)
        strong_parameters["ColorJitter"] = random.uniform(0, 1)
        mix_un_img, mix_un_lrimg, mix_un_mask = _strongTransform(
                                                            parameters=strong_parameters,
                                                            data=image,
                                                            lrdata=lrimage,
                                                            target=pseudomask_cat,
                                                            isaugsym=True
                                                            )
        mix_un_mask_1 = torch.squeeze(mix_un_mask[:, 0:1], dim=1).long()
        mix_un_mask_2 = torch.squeeze(mix_un_mask[:, 1:2], dim=1).long()

    mix_pred_1 = model.branch1(mix_un_img, mix_un_lrimg)
    mix_pred_2 = model.branch2(mix_un_img, mix_un_lrimg)

    cps_loss_1 = criterion(mix_pred_1, mix_un_mask_2)
    cps_loss_2 = criterion(mix_pred_2, mix_un_mask_1)

    loss_record_temp['b1_cps'].append(cps_loss_1.item())
    loss_record_temp['b2_cps'].append(cps_loss_2.item())

    return cps_loss_1 + cps_loss_2