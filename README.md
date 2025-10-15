## MSCPS implement
This script trains a multi-branch semi-supervised segmentation model (MSCPS) with optional validation/test image saving and evaluation.

MSCPS paper: https://ieeexplore.ieee.org/abstract/document/10637254

---
### Usage
```
python train_MSCPS.py [OPTIONS]
```

**Options:**

| Argument            | Type   | Default                       | Description                                |
|--------------------|--------|-------------------------------|--------------------------------------------|
| --num_epochs        | int    | 3                             | Total training epochs                       |
| --consistency_ratio | float  | 0.5                           | Weight for semi-supervised consistency loss|
| --batch_size        | int    | 8                             | Training batch size                          |
| --save_base         | str    | results/MSCPS1015_tiger       | Base path to save results                    |
| --data_cfg          | str    | ./dataprocess/cfg/datacfg_MRCPS_tiger.yaml | Data config YAML path           |
| --save_valimg       | flag   | False                         | Save validation images                        |
| --save_testimg      | flag   | False                         | Save test images                              |
| --device            | str    | cuda                          | Device: cuda or cpu                           |

**Example:**
```
python train_MSCPS.py --num_epochs 10 --consistency_ratio 0.7 --save_valimg --save_testimg
```

This will save validation images in `./results/MSCPS1015_tiger/`valid_img

And test images in `./results/MSCPS1015_tiger/test_img`

---

### Output
- Best and final model weights:
  - `./{save_base}/results/best_model_epoch.pth`
  - `./{save_base}/results/final_model.pth`
- Validation records:
  - `val_records.json`
- Test records:
  - `test_records.json`