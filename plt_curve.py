import os
import json
import matplotlib.pyplot as plt

filefolder = './results/otherset_1141029/'
filename = "val_records.json"
filepath = os.path.join(filefolder, filename)

# === 讀取 JSON 檔案 ===
with open(filepath, "r", encoding="utf-8") as f:
    data = json.load(f)

# === 可選指標 ===
metric = "IoU"  # 可改成 "Fscore", "Recall", "Precision", "sensitivity", "specificity"

# === 取出各分支的資料 ===
epochs = range(1, len(data) + 1)
b1_vals = [entry["b1"][metric] for entry in data]
b2_vals = [entry["b2"][metric] for entry in data]
ens_vals = [entry["ens"][metric] for entry in data]

# === 繪圖 ===
plt.figure(figsize=(8, 5))
plt.plot(epochs, b1_vals, marker='o', label='Branch 1')
plt.plot(epochs, b2_vals, marker='s', label='Branch 2')
plt.plot(epochs, ens_vals, marker='^', label='Ensemble')

plt.title(f'{metric} over Epochs')
plt.xlabel('Epoch')
plt.ylabel(metric)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()


# === 儲存圖片 ===
output_path = os.path.join(filefolder, f"{metric.lower()}_over_epochs.png")

plt.savefig(output_path, dpi=300)
plt.close()  # 關閉圖表以節省記憶體


# # === 顯示 ===
# plt.show()
