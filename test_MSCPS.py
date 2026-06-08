import os
import random
import json
import argparse
import numpy as np
import torch
from torch import nn
from itertools import cycle
import segmentation_models_pytorch.utils as smputils
import torchvision.transforms.functional as TF
from tqdm import tqdm

from dataprocess.datamodule import DataModule
from model.ModelMRCPS import ModelMRCPS
from utils.training_package import _strongTransform
from utils.compute_loss import compute_supervised_loss, compute_consistency_loss
from utils.evaluate import _evaluate
from utils.save_process import save_validation_images

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    print(f"[Seed] {seed}")
    

def validate(model, val_loader, metrics, device, epoch, imageSavePath):
    model.eval()
    total_loss = 0
    evaRecords = []  # 儲存每個 batch 的結果

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader)):
            image, mask, lrimage = [b.to(device) for b in batch]
            
            # --- forward ---
            y_pred_1 = model.branch1(image, lrimage)
            y_pred_2 = model.branch2(image, lrimage)
            
            # --- loss ---
            # loss_1 = criterion(y_pred_1, mask)
            # loss_2 = criterion(y_pred_2, mask)
            # total_loss += ((loss_1 + loss_2) / 2).item()
            
            # --- evaluation ---
            predensem_b1 = [torch.argmax(y_pred_1.softmax(1), dim=1)]
            predensem_b2 = [torch.argmax(y_pred_2.softmax(1), dim=1)]
            voting = y_pred_1.softmax(1) + y_pred_2.softmax(1)
            predensem = [torch.argmax(voting, dim=1)]

            evaRecords.append({
                "b1": _evaluate(predensem_b1, mask, 4),
                "b2": _evaluate(predensem_b2, mask, 4),
                "ens": _evaluate(predensem, mask, 4),
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


def main(args):
    ## train setting / config read
    weight = args.weight
    save_baseName = args.save_base
    save_testImg = args.save_testimg
    data_argpath = args.data_cfg
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    SaveBasePath = f'./{save_baseName}/'
    os.makedirs(SaveBasePath, exist_ok=True)

    ## img save path
    if save_testImg:
        test_imageSavePath = f'./{save_baseName}/test_img'
        os.makedirs(test_imageSavePath, exist_ok=True)
    else:
        test_imageSavePath = None

    ## data prepare
    data_module = DataModule(data_argpath)

    train_loader = data_module.train_dataloader()   # [label, lunlabel]
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    ## model prepare
    model = ModelMRCPS()

    print(weight)
    print(os.path.isfile(weight))
    if os.path.isfile(weight):
        print('===load weight===')
        state_dict = torch.load(weight, map_location=device)
        model.load_state_dict(state_dict)

    model = model.to(device)
    # model = nn.DataParallel(model).to(device)

    metrics = [
            smputils.metrics.IoU(),
            smputils.metrics.Fscore(),
            smputils.metrics.Recall(),
            smputils.metrics.Precision(),
            ]

    evaRecords = []
    lossRecords = []
    best_val_loss = float("inf")

    ## testing (optional)
    test_loss, testEvaRecord = validate(model, test_loader, metrics, device, 0, test_imageSavePath) 
    print(f"Test avg loss: {test_loss:.4f}")
    print(f"Test avg acc: {testEvaRecord}")

    #save test record as json
    with open(f'./{save_baseName}/test_records.json', 'w') as f:
        json.dump(testEvaRecord, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MSCPS Training Script")

    # Training parameters
    parser.add_argument('--weight', type=str, default="", help='pretrained weight for training')

    # Data / Paths
    parser.add_argument('--save_base', type=str, default='results/MSCPS1015_tiger', help='Base path to save results')
    parser.add_argument('--data_cfg', type=str, default='./dataprocess/cfg/datacfg_MRCPS_tiger.yaml', help='Path to data config YAML file')

    # Save options
    parser.add_argument('--save_testimg', action='store_true', help='Save test images')

    # GPU / device
    parser.add_argument('--device', type=str, default='cuda', help='cuda / cpu')

    args = parser.parse_args()

    set_seed(42)
    main(args)