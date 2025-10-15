import os
import random
import json
import numpy as np
import torch
from torch import nn
from itertools import cycle
import segmentation_models_pytorch.utils as smputils
import torchvision.transforms.functional as TF

from dataprocess.datamodule import DataModule
from model.ModelMRCPS import ModelMRCPS
from utils.training_package import _strongTransform

def compute_supervised_loss(model, batch, criterion, loss_record_temp):
    image, mask, lrimage, lrnmask = batch

    y_pred_1_sup = model.branch1(image, lrimage)
    y_pred_2_sup = model.branch2(image, lrimage)
    
    sup_loss_1 = criterion(y_pred_1_sup, mask)
    sup_loss_2 = criterion(y_pred_2_sup, mask)

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
                                                            strong_parameters,
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


def train_one_epoch(model, train_loader, optimizer1, optimizer2, criterion, 
                    consistencyratio, device):
    model.train()
    loss_record_temp = {'total':[], 'b1_sup':[], 'b2_sup':[],
                        'b1_cps':[], 'b2_cps':[]}

    #unlabel step
    label_iter = cycle(train_loader['label'])
    for unlabel_batch in train_loader['unlabel']:
        label_batch = next(label_iter)

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        # 計算 label_loss 與 unlabel_loss
        loss_label = compute_supervised_loss(model, label_batch, criterion, loss_record_temp)
        loss_unlabel = compute_consistency_loss(model, unlabel_batch, criterion, loss_record_temp)
        
        semi_loss = loss_label + consistencyratio * loss_unlabel
        semi_loss.backward()

        optimizer1.step()
        optimizer2.step()

        loss_record_temp['total'].append(semi_loss.item())

    # -------- epoch 平均統計 --------
    loss_record = {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in loss_record_temp.items()}

    return loss_record



def _evaluate(predmask, y, metrics, stage: str = "valid"):
    """
    計算模型在一批資料上的評估指標（不依賴 Lightning）
    
    Args:
        predmask: list of predicted masks (每個分支一個)
        y: ground truth tensor
        stage: 'train' | 'valid' | 'test'
    
    Returns:
        results_dict: 包含 sensitivity / specificity / 各 metric 的平均值
    """
    results = {}
    results_list = []

    # 單模型
    if len(predmask) == 1:
        pm = predmask[0]

        tp = (torch.eq(pm, y) & (y == 1)).sum()
        fp = (torch.eq(pm, 1) & (y == 0)).sum()
        fn = (torch.eq(pm, 0) & (y == 1)).sum()
        tn = (torch.eq(pm, y) & (y == 0)).sum()

        sensitivity = (tp + 1e-6) / (tp + fn + 1e-6)
        specificity = (tn + 1e-6) / (tn + fp + 1e-6)

        results["sensitivity"] = sensitivity.item()
        results["specificity"] = specificity.item()

        # segmentation metrics
        for metric_fn in metrics:
            metric_value = metric_fn(pm, y)
            results[metric_fn.__name__] = metric_value.item()

        results_list.append(results)
        print(f"[{stage}] sensitivity: {sensitivity:.4f}, specificity: {specificity:.4f}")

    # 多模型
    else:
        sensitivity_values = []
        specificity_values = []
        metric_results = {m.__name__: [] for m in metrics}

        for idx, pm in enumerate(predmask):
            tp = (torch.eq(pm, y) & (y == 1)).sum()
            fp = (torch.eq(pm, 1) & (y == 0)).sum()
            fn = (torch.eq(pm, 0) & (y == 1)).sum()
            tn = (torch.eq(pm, y) & (y == 0)).sum()

            sensitivity = (tp + 1e-6) / (tp + fn + 1e-6)
            specificity = (tn + 1e-6) / (tn + fp + 1e-6)
            sensitivity_values.append(sensitivity.item())
            specificity_values.append(specificity.item())

            for metric_fn in metrics:
                metric_value = metric_fn(pm, y)
                metric_results[metric_fn.__name__].append(metric_value.item())

        # 平均結果
        results["sensitivity"] = float(torch.tensor(sensitivity_values).mean())
        results["specificity"] = float(torch.tensor(specificity_values).mean())
        for name, vals in metric_results.items():
            results[name] = float(torch.tensor(vals).mean())

        results_list.append(results)
        print(f"[{stage}] avg sensitivity: {results['sensitivity']:.4f}, "
              f"avg specificity: {results['specificity']:.4f}")

    return results



def save_validation_images(model, image_batch, mask_batch, predensem_list, epoch, valid_imageSavePath, max_num=4):
    """
    儲存 validation 的幾張影像：
     - 原圖 (image)
     - gt (mask)
     - inference (ensemble / 第一個 branch)
     - overlay / overlayGT

    參數:
      model: 可能含有 model.valid_imageSavePath 屬性
      image_batch: tensor [B, C, H, W] (0..1 float)
      mask_batch: tensor [B, H, W] (class indices)
      predensem_list: list like [predensem], 其中 predensem 是 [B, H, W]
      epoch: 當前 epoch，用於檔名
      max_num: 最多儲存幾張
    """
    save_dir = valid_imageSavePath
    os.makedirs(save_dir, exist_ok=True)

    # 取要儲存的張數
    B = image_batch.shape[0]
    nsave = min(B, max_num)

    # 有可能 predensem_list[0] 是一個 tensor list
    pred_tensor = predensem_list[0] if isinstance(predensem_list, (list, tuple)) else predensem_list

    for img_idx in range(nsave):
        img = image_batch[img_idx].cpu()  # [C, H, W], 0..1
        gt = mask_batch[img_idx].cpu()  # [H, W]
        pred = pred_tensor[img_idx].cpu()  # [H, W]

        # 轉 PIL
        try:
            pil_img = TF.to_pil_image(img)  # original image
        except Exception:
            # 如果 channel=1
            pil_img = TF.to_pil_image(img.squeeze(0))

        # GT: convert to RGB visualization by mapping classes -> gray-scale
        gt_vis = (gt.unsqueeze(0).float() * (255.0 / max(1, gt.max().item()))).to(dtype=torch.uint8)
        try:
            pil_gt = TF.to_pil_image(gt_vis)
        except Exception:
            pil_gt = TF.to_pil_image(gt_vis.squeeze(0))

        # prediction visualization
        pred_vis = (pred.unsqueeze(0).float() * (255.0 / max(1, pred.max().item()))).to(dtype=torch.uint8)
        try:
            pil_pred = TF.to_pil_image(pred_vis)
        except Exception:
            pil_pred = TF.to_pil_image(pred_vis.squeeze(0))

        # overlay (簡單 alpha blending)：需將 pred/gt 轉為三通道以便 overlay
        try:
            pred_rgb = TF.to_pil_image(torch.stack([pred_vis.squeeze(0)]*3))
            gt_rgb = TF.to_pil_image(torch.stack([gt_vis.squeeze(0)]*3))
            overlay = TF.to_pil_image((0.6 * img + 0.4 * TF.to_tensor(pred_rgb)).clamp(0, 1))
        except Exception:
            overlay = pil_img

        # 檔名可帶 client_id / round_n / current_epoch 若 model 有這些屬性
        client_id = getattr(model, "client_id", "c")
        round_n = getattr(model, "round_n", "r")
        current_epoch = getattr(model, "current_epoch", epoch)

        pil_img.save(os.path.join(save_dir, f'valid_{client_id}_round{round_n}_epoch{current_epoch}_img{img_idx}_img.png'), format='PNG')
        pil_gt.save(os.path.join(save_dir, f'valid_{client_id}_round{round_n}_epoch{current_epoch}_img{img_idx}_gt.png'), format='PNG')
        pil_pred.save(os.path.join(save_dir, f'valid_{client_id}_round{round_n}_epoch{current_epoch}_img{img_idx}_inf.png'), format='PNG')
        overlay.save(os.path.join(save_dir, f'valid_{client_id}_round{round_n}_epoch{current_epoch}_img{img_idx}_overlay.png'), format='PNG')




def validate(model, val_loader, criterion, metrics, device, epoch, imageSavePath):
    model.eval()
    total_loss = 0
    evaRecords = []  # 儲存每個 batch 的結果

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            image, mask, lrimage = [b.to(device) for b in batch]
            
            # --- forward ---
            y_pred_1 = model.branch1(image, lrimage)
            y_pred_2 = model.branch2(image, lrimage)
            
            # --- loss ---
            loss_1 = criterion(y_pred_1, mask)
            loss_2 = criterion(y_pred_2, mask)
            total_loss += ((loss_1 + loss_2) / 2).item()
            
            # --- evaluation ---
            predensem_b1 = [torch.argmax(y_pred_1.softmax(1), dim=1)]
            predensem_b2 = [torch.argmax(y_pred_2.softmax(1), dim=1)]
            voting = y_pred_1.softmax(1) + y_pred_2.softmax(1)
            predensem = [torch.argmax(voting, dim=1)]

            evaRecords.append({
                "b1": _evaluate(predensem_b1, mask, metrics, "valid b1"),
                "b2": _evaluate(predensem_b2, mask, metrics, "valid b2"),
                "ens": _evaluate(predensem, mask, metrics, "valid ens"),
            })

            # --- 儲存影像 (僅第一個 batch) ---
            if batch_idx == 0:
                save_validation_images(model, image, mask, predensem, epoch, imageSavePath)

    # --- 平均 loss 與評估記錄 ---
    mean_loss = total_loss / len(val_loader)

    # --- 將所有 batch 的結果平均 ---
    mean_eval = {}
    for key in ["b1", "b2", "ens"]:
        # 取出每個 batch 對應的 metric 平均
        all_keys = evaRecords[0][key].keys()
        mean_eval[key] = {k: np.mean([batch[key][k] for batch in evaRecords]) for k in all_keys}

    return mean_loss, mean_eval




def main():
    ## train setting / config read
    num_epochs = 1
    consistencyratio = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_baseName = 'MSCPS1009_liver'
    valid_imageSavePath = f'./{save_baseName}/results/valid_img'
    test_imageSavePath = f'./{save_baseName}/results/test_img'
    os.makedirs(valid_imageSavePath, exist_ok=True)
    os.makedirs(test_imageSavePath, exist_ok=True)

    ## data prepare
    data_argpath = './dataprocess/cfgs/data_config.yaml'
    data_module = DataModule(data_argpath)

    train_loader = data_module.train_dataloader()   # [label, lunlabel]
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    ## model prepare
    model = ModelMRCPS().to(device)

    opt1 = torch.optim.Adam(model.branch1.parameters(), lr=1e-4)
    opt2 = torch.optim.Adam(model.branch2.parameters(), lr=1e-4)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt1, T_0=10)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt2, T_0=10)
    criterion = nn.CrossEntropyLoss()
    
    metrics = [
            smputils.metrics.IoU(),
            smputils.metrics.Fscore(),
            smputils.metrics.Recall(),
            smputils.metrics.Precision(),
            ]

    evaRecords = []
    best_val_loss = float("inf")

    ## training
    for epoch in range(num_epochs):
        train_loss_dict  = train_one_epoch(model, train_loader, opt1, opt2, criterion, consistencyratio, device)
        val_loss, evaRecord = validate(model, val_loader, criterion, metrics, device, epoch, valid_imageSavePath)
        evaRecords.append(evaRecord)

        sch1.step()
        sch2.step()

        print(f"[Epoch {epoch}] Train total: {train_loss_dict['total']:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'./{save_baseName}/results/best_model_epoch{epoch}.pth')

    # save final weight
    torch.save(model.state_dict(), f'./{save_baseName}/results/final_model.pth')

    #save valid records as json
    with open(f'./{save_baseName}/results/val_records.json', 'w') as f:
        json.dump(evaRecords, f, indent=2)

    ## testing (optional)
    test_loss, testEvaRecord = validate(model, test_loader, criterion, metrics, device, 0, test_imageSavePath) 
    print(f"Test avg loss: {test_loss:.4f}")

    #save test record as json
    with open(f'./{save_baseName}/results/test_records.json', 'w') as f:
        json.dump(testEvaRecord, f, indent=2)

if __name__ == '__main__':
    main()