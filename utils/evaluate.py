import torch

def _evaluate(predmask, y, metrics=None, stage: str = "valid"):
    """
    計算模型在一批資料上的評估指標（不依賴 Lightning）
    
    Args:
        predmask: list of predicted masks (每個分支一個)
        y: ground truth tensor
        metrics: list of metric functions (可為 smputils.Metric 類)
        stage: 'train' | 'valid' | 'test'
    
    Returns:
        results_dict: 包含 sensitivity / specificity / 各 metric 的平均值
    """
    results = {}
    results_list = []
    if metrics is None:
        metrics = []  # 可傳入 smputils.metrics.IoU() 之類的 metric
    
    # 定義本地 IoU 函數（若沒給 metric）
    def _iou(pred, target, eps=1e-6):
        """自定義 IoU 計算"""
        pred = (pred > 0.5).float()  # 確保是 0/1 mask
        target = (target > 0.5).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return (intersection + eps) / (union + eps)

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

        # smputils metric + 自動 iou
        has_iou = False
        for metric_fn in metrics:
            name = metric_fn.__class__.__name__ if hasattr(metric_fn, '__class__') else metric_fn.__name__
            metric_value = metric_fn(pm, y)
            results[name] = metric_value.item()
            if "IoU" in name or "Jaccard" in name:
                has_iou = True

        if not has_iou:
            results["IoU"] = _iou(pm, y).item()

        results_list.append(results)

    # 多模型
    else:
        sensitivity_values = []
        specificity_values = []
        metric_results = {m.__class__.__name__ if hasattr(m, '__class__') else m.__name__: [] for m in metrics}
        iou_values = []

        for pm in predmask:
            tp = (torch.eq(pm, y) & (y == 1)).sum()
            fp = (torch.eq(pm, 1) & (y == 0)).sum()
            fn = (torch.eq(pm, 0) & (y == 1)).sum()
            tn = (torch.eq(pm, y) & (y == 0)).sum()

            sensitivity = (tp + 1e-6) / (tp + fn + 1e-6)
            specificity = (tn + 1e-6) / (tn + fp + 1e-6)
            sensitivity_values.append(sensitivity.item())
            specificity_values.append(specificity.item())

            # smputils metric
            for metric_fn in metrics:
                name = metric_fn.__class__.__name__ if hasattr(metric_fn, '__class__') else metric_fn.__name__
                metric_value = metric_fn(pm, y)
                metric_results[name].append(metric_value.item())

            # 自動 IoU
            iou_values.append(_iou(pm, y).item())

        # 平均結果
        results["sensitivity"] = float(torch.tensor(sensitivity_values).mean())
        results["specificity"] = float(torch.tensor(specificity_values).mean())

        for name, vals in metric_results.items():
            results[name] = float(torch.tensor(vals).mean())

        # 若沒傳入 IoU 類 metric，自動補上
        if not any("IoU" in k or "Jaccard" in k for k in metric_results.keys()):
            results["IoU"] = float(torch.tensor(iou_values).mean())

        results_list.append(results)

    return results
