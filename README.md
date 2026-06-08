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

### Test 使用說明

可使用 `test.sh` 或直接執行 `test_MSCPS.py` 進行測試：
```
bash test.sh
```

或直接執行：
```
python test_MSCPS.py --weight "./results/MSCS_labBreast_Mackey_0601/best_model_epoch.pth" --save_base "results/MSCS_labBreast_Mackey_0601" --data_cfg "./dataprocess/cfg/datacfg_MRCPS_labBreast.yaml" --save_testimg
```

**Options:**

| Argument        | Type   | Default                               | Description                                 |
|----------------|--------|----------------------------------------|---------------------------------------------|
| --weight       | str    | ""                                     | 欲載入的模型權重檔案                          |
| --save_base    | str    | results/MSCPS1015_tiger                | 結果儲存根目錄                               |
| --data_cfg     | str    | ./dataprocess/cfg/datacfg_MRCPS_tiger.yaml | 資料配置 YAML 路徑                 |
| --save_testimg | flag   | False                                  | 儲存測試推論影像                             |
| --device       | str    | cuda                                   | 裝置: cuda 或 cpu                            |

測試結果會輸出為：
- `./{save_base}/test_records.json`
- 若啟用 `--save_testimg`，則會在 `./{save_base}/test_img/` 保存測試影像

---

### Output
- Best and final model weights:
  - `./{save_base}/results/best_model_epoch.pth`
  - `./{save_base}/results/final_model.pth`
- Validation records:
  - `val_records.json`
- Test records:
  - `test_records.json`

---

### 其他工具程式說明

#### `inf_wsi.py`
- 實作 `WSIPatchDataset`，用於從 whole-slide image（WSI）讀取 patch。
- 使用 `pyvips` 讀取 TIFF/WSI 檔案，並根據 patch size、stride、背景閾值自動生成 patch 位置。
- 支援多尺度分支（`lrratio`）與低解析度 patch 輸入。
- 提供影像轉換與輔助函式，例如 `numpy2vips()`、`img_tensor2pillow()`、`img_tensor2pillow_mask()`。

#### `model_info.py`
- 建立 `ModelMRCPS()` 模型實例。
- 若提供 `preweight` 權重檔，會嘗試載入 checkpoint。
- 列印模型參數字典與分支權重鍵值（例如 `branch1` 相關參數），方便檢查模型結構與權重匹配。

#### `plt_curve.py`
- 讀取結果資料夾中的 `val_records.json` 並繪製訓練指標曲線。
- 預設繪製 `IoU`，可修改為 `Fscore`、`Recall`、`Precision`、`sensitivity`、`specificity`。
- 產生的圖檔預設儲存為 `./results/otherset_1141029/{metric.lower()}_over_epochs.png`，可修改 `filefolder` 路徑以對應自己的實驗資料夾。