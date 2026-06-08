## MSCPS implement
This script trains a multi-branch semi-supervised segmentation model (MSCPS) with optional validation/test image saving and evaluation.

MSCPS paper: https://ieeexplore.ieee.org/abstract/document/10637254

---
### Usage
```
python train_MSCPS.py [OPTIONS]
```

或直接使用 `train.sh` 執行相同訓練指令：
```
bash train.sh
```

**Options:**

| Argument            | Type   | Default                       | Description                                |
|--------------------|--------|-------------------------------|--------------------------------------------|
| --num_epochs        | int    | 3                             | Total training epochs                      |
| --consistency_ratio | float  | 0.5                           | Weight for semi-supervised consistency loss|
| --batch_size        | int    | 8                             | Training batch size                        |
| --save_base         | str    | results/MSCPS1015_tiger       | Base path to save results                  |
| --data_cfg          | str    | ./dataprocess/cfg/datacfg_MRCPS_tiger.yaml | Data config YAML path           |
| --save_valimg       | flag   | False                         | Save validation images                     |
| --save_testimg      | flag   | False                         | Save test images                           |
| --device            | str    | cuda                          | Device: cuda or cpu                        |

**Example:**
```
python train_MSCPS.py --num_epochs 10 --consistency_ratio 0.7 --save_base "results/MSCS_labBreast" --data_cfg "./dataprocess/cfg/datacfg_MRCPS_labBreast.yaml" --save_valimg --save_testimg
```

`train.sh` 中的實際指令：
```
python train_MSCPS.py --num_epochs 20 --consistency_ratio 0.5 --save_base "results/MSCS_labBreast" --data_cfg "./dataprocess/cfg/datacfg_MRCPS_labBreast.yaml" --save_valimg --save_testimg
```

- 驗證影像會輸出到： `./results/MSCS_labBreast/valid_img/`
- 測試影像會輸出到： `./results/MSCS_labBreast/test_img/`

---

### Output
- Best and final model weights:
  - `./{save_base}/results/best_model_epoch.pth`
  - `./{save_base}/results/final_model.pth`
- Validation records:
  - `val_records.json`
- Test records:
  - `test_records.json`