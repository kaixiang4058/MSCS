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


def train_one_epoch(model, train_loader, optimizer1, optimizer2, criterion, 
                    consistencyratio, device):
    model.train()
    loss_record_temp = {'total':[], 'b1_sup':[], 'b2_sup':[],
                        'b1_cps':[], 'b2_cps':[]}

    #unlabel step
    for label_batch in tqdm(train_loader['label']):

        #problem batch is list
        label_batch = [t.to(device, non_blocking=True) for t in label_batch]

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        # 計算 label_loss 與 unlabel_loss
        loss_label = compute_supervised_loss(model, label_batch, criterion, loss_record_temp)
        
        semi_loss = loss_label
        semi_loss.backward()

        optimizer1.step()
        optimizer2.step()

        loss_record_temp['total'].append(semi_loss.item())

    # -------- epoch 平均統計 --------
    loss_record = {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in loss_record_temp.items()}

    return loss_record


def train_one_epoch_semi(model, train_loader, optimizer1, optimizer2, criterion, 
                    consistencyratio, device):
    model.train()
    loss_record_temp = {'total':[], 'b1_sup':[], 'b2_sup':[],
                        'b1_cps':[], 'b2_cps':[], 
                        'b1_sup_ce':[],'b1_sup_dice':[],
                        'b2_sup_ce':[],'b2_sup_dice':[]}

    #unlabel step
    label_iter = cycle(train_loader['label'])
    for unlabel_batch in tqdm(train_loader['unlabel']):
        label_batch = next(label_iter)  

        #problem batch is list
        label_batch = [t.to(device, non_blocking=True) for t in label_batch]
        unlabel_batch = [t.to(device, non_blocking=True) for t in unlabel_batch]

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


def validate(model, val_loader, criterion, metrics, device, epoch, imageSavePath):
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


def main(args):
    ## train setting / config read
    num_epochs = args.num_epochs
    consistencyratio = args.consistency_ratio
    save_baseName = args.save_base
    save_validImg = args.save_valimg
    save_testImg = args.save_testimg
    data_argpath = args.data_cfg
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    preweight = args.preweight

    SaveBasePath = f'./{save_baseName}/'
    os.makedirs(SaveBasePath, exist_ok=True)

    ## img save path
    if save_validImg:
        valid_imageSavePath = f'./{save_baseName}/valid_img'
        os.makedirs(valid_imageSavePath, exist_ok=True)
    else:
        valid_imageSavePath = None
    
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

    if os.path.isfile(preweight):
        state_dict = torch.load(preweight, map_location=device)
        model.load_state_dict(state_dict)

    model = model.to(device)
    # model = nn.DataParallel(model).to(device)

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
    lossRecords = []
    best_val_loss = float("inf")

    ## --- first valid ---
    print(f"Valid before training")
    val_loss, evaRecord = validate(model, val_loader, criterion, metrics, device, 0, valid_imageSavePath)
    evaRecords.append(evaRecord)

    print(f"[before training] Val loss: {val_loss:.4f}")
    print(f"[before training] Val acc:{evaRecord}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss


    ## --- training --- 
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} Train")
        train_loss_dict  = train_one_epoch_semi(model, train_loader, opt1, opt2, criterion, consistencyratio, device)
        lossRecords.append(train_loss_dict)
        print(f"Epoch {epoch+1} Valid")
        val_loss, evaRecord = validate(model, val_loader, criterion, metrics, device, epoch+1, valid_imageSavePath)
        evaRecords.append(evaRecord)

        sch1.step()
        sch2.step()

        print(f"[Epoch {epoch+1}] Train total loss: {train_loss_dict['total']:.4f} | Val loss: {val_loss:.4f}")
        print(f"[Epoch {epoch+1}] Val acc:{evaRecord}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'./{save_baseName}/best_model_epoch.pth')

    # save final weight
    torch.save(model.state_dict(), f'./{save_baseName}/final_model.pth')

    #save loss records as json
    with open(f'./{save_baseName}/loss_records.json', 'w') as f:
        json.dump(lossRecords, f, indent=2)
    #save valid records as json
    with open(f'./{save_baseName}/val_records.json', 'w') as f:
        json.dump(evaRecords, f, indent=2)

    ## testing (optional)
    test_loss, testEvaRecord = validate(model, test_loader, criterion, metrics, device, 0, test_imageSavePath) 
    print(f"Test avg loss: {test_loss:.4f}")
    print(f"Test avg acc: {testEvaRecord}")

    #save test record as json
    with open(f'./{save_baseName}/test_records.json', 'w') as f:
        json.dump(testEvaRecord, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MSCPS Training Script")

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=3, help='Total number of training epochs')
    parser.add_argument('--consistency_ratio', type=float, default=0.5, help='Weight for semi-supervised consistency loss')
    # parser.add_argument('--batch_size', type=int, default=8, help='Training batch size') #define in dataprocess cfg
    parser.add_argument('--preweight', type=str, default="", help='pretrained weight for training')

    # Data / Paths
    parser.add_argument('--save_base', type=str, default='results/MSCPS1015_tiger', help='Base path to save results')
    parser.add_argument('--data_cfg', type=str, default='./dataprocess/cfg/datacfg_MRCPS_tiger.yaml', help='Path to data config YAML file')

    # Save options
    parser.add_argument('--save_valimg', action='store_true', help='Save validation images')
    parser.add_argument('--save_testimg', action='store_true', help='Save test images')

    # GPU / device
    parser.add_argument('--device', type=str, default='cuda', help='cuda / cpu')

    args = parser.parse_args()

    set_seed(42)
    main(args)