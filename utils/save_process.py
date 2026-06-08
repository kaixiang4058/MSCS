import os
import torch
import torchvision.transforms.functional as TF

def save_validation_images(model, image_batch, mask_batch, predensem_list, epoch, imageSavePath, max_num=4):
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

        if not (imageSavePath is None):
            pil_img.save(os.path.join(imageSavePath, f'valid_{client_id}_round{round_n}_epoch{current_epoch}_img{img_idx}_img.png'), format='PNG')
            pil_gt.save(os.path.join(imageSavePath, f'valid_{client_id}_round{round_n}_epoch{current_epoch}_img{img_idx}_gt.png'), format='PNG')
            pil_pred.save(os.path.join(imageSavePath, f'valid_{client_id}_round{round_n}_epoch{current_epoch}_img{img_idx}_inf.png'), format='PNG')
            overlay.save(os.path.join(imageSavePath, f'valid_{client_id}_round{round_n}_epoch{current_epoch}_img{img_idx}_overlay.png'), format='PNG')
