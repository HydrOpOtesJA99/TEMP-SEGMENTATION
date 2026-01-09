import torch

# 1) PyTorch 버전 확인
print(torch.__version__)  # 예: 2.1.0+cu124

# 2) CUDA(GPU) 사용 가능 여부 확인
print(torch.cuda.is_available())  # True면 GPU 사용 가능
print(torch.cuda.get_device_name())

import torch
import segmentation_models_pytorch as smp # v0.5.0
import os
import random
import pandas as pd
import cv2
import torch
import torch.nn.utils
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd
from datetime import datetime

import json
import glob

import ssl

import random
import time

# =========================
# 0. 기본 설정
# =========================

# =========================
# 전체 학습 시작 시간 기록
# =========================
start_time = time.time()

seed = 18
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

NUM_EPOCHS = 200
learning_rate = 1e-4

# base_dir = "/BiO/Sangjin_Ko_work_2022/51_25.12.22_AI_MODEL/"
# train_folder = "000_Dataset/000_26.01.05_NO_AUG_DATASET"
base_dir = "./"
train_folder = "00000_ALL_JSON_LABELED-FINAL-NEW"


image_size_value = 512
encoder_model_value = "resnet50"
encoder_weights_value = "imagenet"
in_channel_value = 3
classes_value = 2


#original_context = ssl._create_default_https_context
#ssl._create_default_https_context = ssl._create_unverified_context

MODEL = "UnetPlusPlus"
model = smp.UnetPlusPlus(encoder_name=encoder_model_value, encoder_weights=encoder_weights_value, in_channels=in_channel_value, classes=classes_value).cuda()
print(f"Download completed")

#ssl._create_default_https_context = original_context
#print("SSL verification restored")

PARAMETER = MODEL + "_Leaf_and_Disease_" + str(NUM_EPOCHS) + "epoch"
#earlystop = EarlyStopping(patience=10, min_delta=0.00001)

TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train / Validation / Test 경로
train_images_dir = os.path.join(base_dir, train_folder, "000_TRAIN_images_NEW")
train_json_dir   = os.path.join(base_dir, train_folder, "000_TRAIN_json_NEW")
valid_images_dir = os.path.join(base_dir, train_folder, "000_VALID_images_NEW")
valid_json_dir   = os.path.join(base_dir, train_folder, "000_VALID_json_NEW")
test_images_dir  = os.path.join(base_dir, train_folder, "000_TEST_images_NEW")
test_json_dir    = os.path.join(base_dir, train_folder, "000_TEST_json_NEW")

checkpoint_dir = base_dir + "/" + train_folder + "/" + MODEL +"_LESION_SEG_MODEL_checkpoints_" + current_time_str
os.makedirs(checkpoint_dir, exist_ok=True)


# =========================
# 1. Dataset 준비 (image / json 폴더 분리)
# =========================
def prepare_data(images_dir, json_dir):
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    data_list = []
    for img_name in image_files:
        json_name = os.path.splitext(img_name)[0] + ".json"
        img_path = os.path.join(images_dir, img_name)
        json_path = os.path.join(json_dir, json_name)

        if os.path.exists(json_path):
            data_list.append({
                "image": img_path,
                "annotation": json_path
            })
    return data_list
    
train_data = prepare_data(train_images_dir, train_json_dir)
valid_data = prepare_data(valid_images_dir, valid_json_dir)
test_data  = prepare_data(test_images_dir,  test_json_dir)


# =========================
# 2. Preprocessing & Dataset
# =========================
preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_model_value, encoder_weights_value)

def json_to_mask_leaf_lesion(json_path, shape):
    leaf_mask = np.zeros(shape, dtype=np.uint8)
    lesion_mask = np.zeros(shape, dtype=np.uint8)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for shape_obj in data.get("shapes", []):
        label = shape_obj.get("label", "").lower()
        points = np.array(shape_obj.get("points", []), dtype=np.int32)
        if points.size == 0:
            continue
        if label == "leaf":
            cv2.fillPoly(leaf_mask, [points], 1)
        elif label in ["disease", "lesion"]:   # sangjin update
            cv2.fillPoly(lesion_mask, [points], 1)

    return leaf_mask, lesion_mask  # sangjin update


class ModelDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, image_size=image_size_value):
        self.data_list = data_list
        self.image_size = image_size
        self.preprocessing_fn = preprocessing_fn
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        ent = self.data_list[idx]
        Img = cv2.imread(ent["image"])[..., ::-1] # BGR > RGB convert
        H, W = Img.shape[:2]

        leaf_mask, lesion_mask = json_to_mask_leaf_lesion(ent["annotation"], (H, W))

        Img = cv2.resize(Img, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        leaf_mask = cv2.resize(leaf_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        leaf_mask = np.round(leaf_mask).astype(np.uint8)
        
        lesion_mask = cv2.resize(lesion_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        lesion_mask = np.round(lesion_mask).astype(np.uint8)

        valid_region = (leaf_mask == 1) | (lesion_mask == 1)

        mask = np.full((self.image_size, self.image_size), 255, dtype=np.uint8)
        mask[valid_region] = 0
        mask[lesion_mask == 1] = 1
        
        Img = torch.from_numpy(self.preprocessing_fn(Img)).permute(2,0,1).float()
        mask = torch.from_numpy(mask).long()

        return Img, mask        


train_dataset = ModelDataset(train_data, image_size=image_size_value)
valid_dataset = ModelDataset(valid_data, image_size=image_size_value)
test_dataset  = ModelDataset(test_data, image_size=image_size_value)

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 전처리/데이터 확인 코드
for imgs, masks in train_loader:
    print("mask shape:", masks.shape, "dtype:", masks.dtype, "unique:", torch.unique(masks))
    break

# IoU 계산
def compute_iou_per_image(preds, targets, cls):
    #preds = torch.argmax(preds, dim=1)
    ious = []
    for p, t in zip(preds, targets):
        valid = (t != 255) # ignore pixel 제외
        p_cls = (p == cls) & valid
        t_cls = (t == cls) & valid

        pred_has = p_cls.sum() > 0
        gt_has   = t_cls.sum() > 0

        if not pred_has and not gt_has:
            ious.append(1.0)  # case 1
        elif not pred_has and gt_has:
            ious.append(0.0)  # case 2
        elif pred_has and not gt_has:
            ious.append(0.0)  # case 3
        else:
            intersection = (p_cls & t_cls).sum().float()
            union = (p_cls | t_cls).sum().float() + 1e-6
            ious.append((intersection / union).item())  # case 4
    return ious        


def compute_dice_per_image(preds, targets, cls):
    #preds = torch.argmax(preds, dim=1)
    dices = []
    for p, t in zip(preds, targets):
        valid = (t != 255)
        preds_cls = ((p == cls) & valid).float()
        targets_cls = ((t == cls) & valid).float()
        intersection = (preds_cls * targets_cls).sum()
        
        if targets_cls.sum() == 0 and preds_cls.sum() == 0:  # target, pred 모두 없음
            dices.append(1.0)
        elif targets_cls.sum() == 0:  # target 없음, pred 있음
            dices.append(0.0)
        else:
            dice = (2.0 * intersection + 1e-6) / (preds_cls.sum() + targets_cls.sum() + 1e-6)
            dices.append(dice.item())
    return dices


def compute_pixel_accuracy_per_image(preds, targets, cls):
    #preds = torch.argmax(preds, dim=1)
    accs = []
    for p, t in zip(preds, targets):
        valid = (t != 255)
        mask = (t == cls) & valid
        pred_mask = (p == cls) & valid
        
        if mask.sum() == 0 and pred_mask.sum() == 0:  # target, pred 모두 없음
            accs.append(1.0)
        elif mask.sum() == 0:  # target 없음, pred 있음
            accs.append(0.0)
        else:
            correct = (p == cls) & mask
            accs.append(correct.float().sum().item() / mask.sum().item())
    return accs


def compute_precision_per_image(preds, targets, cls):
    #preds = torch.argmax(preds, dim=1)
    precisions = []
    for p, t in zip(preds, targets):
        valid = (t != 255)
        preds_cls = ((p == cls) & valid).float()
        targets_cls = ((t == cls) & valid).float()
        TP = (preds_cls * targets_cls).sum()
        FP = ((preds_cls == 1) & (targets_cls == 0)).sum()

        if targets_cls.sum() == 0 and preds_cls.sum() == 0:
            precisions.append(1.0)
        elif preds_cls.sum() == 0:  # pred 없음
            precisions.append(1.0 if targets_cls.sum() == 0 else 0.0)
        else:
            precisions.append((TP + 1e-6) / (TP + FP + 1e-6))
    return precisions


def compute_recall_per_image(preds, targets, cls):
    #preds = torch.argmax(preds, dim=1)
    recalls = []
    for p, t in zip(preds, targets):
        valid = (t != 255)
        preds_cls = ((p == cls) & valid).float()
        targets_cls = ((t == cls) & valid).float()
        TP = (preds_cls * targets_cls).sum()
        FN = ((preds_cls == 0) & (targets_cls == 1)).sum()

        if targets_cls.sum() == 0 and preds_cls.sum() == 0:
            recalls.append(1.0)
        elif targets_cls.sum() == 0:  # target 없음, pred 있음
            recalls.append(0.0)
        else:
            recalls.append((TP + 1e-6) / (TP + FN + 1e-6))
    return recalls


def compute_f1_from_prec_recall(precision_list, recall_list):
    f1_list = []
    for p, r in zip(precision_list, recall_list):
        # p, r가 Tensor이면 float로 변환
        if isinstance(p, torch.Tensor):
            p = p.item()
        if isinstance(r, torch.Tensor):
            r = r.item()
        f1 = (2 * p * r + 1e-6) / (p + r + 1e-6)
        f1_list.append(f1)
    return f1_list


def to_cpu_list(tensor_list):
    return [t.item() if torch.is_tensor(t) else t for t in tensor_list]

def compute_metrics_per_batch(outputs, targets, classes=[0,1]):
    preds = torch.argmax(outputs, dim=1)
    batch_metrics = {cls:{k: [] for k in ["iou","dice","acc","prec","rec","f1"]} for cls in classes}
    
    for cls in classes:
        iou = compute_iou_per_image(preds, targets, cls)
        dice = compute_dice_per_image(preds, targets, cls)
        acc = compute_pixel_accuracy_per_image(preds, targets, cls)
        prec = compute_precision_per_image(preds, targets, cls)
        rec = compute_recall_per_image(preds, targets, cls)
        f1  = compute_f1_from_prec_recall(prec, rec)
        
        batch_metrics[cls]["iou"].extend(to_cpu_list(iou))
        batch_metrics[cls]["dice"].extend(to_cpu_list(dice))
        batch_metrics[cls]["acc"].extend(to_cpu_list(acc))
        batch_metrics[cls]["prec"].extend(to_cpu_list(prec))
        batch_metrics[cls]["rec"].extend(to_cpu_list(rec))
        batch_metrics[cls]["f1"].extend(to_cpu_list(f1))
    return batch_metrics

def average_metrics(metrics_dict):
    avg_metrics = {}
    for cls, cls_metrics in metrics_dict.items():
        avg_metrics[cls] = {k: np.mean(v) if len(v) > 0 else 0.0 for k,v in cls_metrics.items()}
    return avg_metrics

def print_gpu_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU 메모리 사용: {allocated:.2f} GB, 캐시 사용: {cached:.2f} GB")


# =========================
# Checkpoint tracking 변수
# =========================
best_val_loss = float("inf")
best_val_iou  = -1.0
best_disease_iou = -1.0
global_step = 0


log_metrics = {
    "train": {cls: {k: [] for k in ["iou","dice","acc","prec","rec","f1"]} for cls in [0,1]},
    "val":   {cls: {k: [] for k in ["iou","dice","acc","prec","rec","f1"]} for cls in [0,1]},
    "train_avg": {k: [] for k in ["mIoU","mDice","mPA","mPrecision","mRecall","mF1"]},
    "val_avg":   {k: [] for k in ["mIoU","mDice","mPA","mPrecision","mRecall","mF1"]},
    "loss": {"train": [], "val": []}
}

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    
    train_metrics = {cls: {k: [] for k in ["iou","dice","acc","prec","rec","f1"]} for cls in [0,1]}
    
    for images, masks in train_loader:
        images, masks = images.cuda(), masks.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        global_step += 1

        batch_metrics = compute_metrics_per_batch(outputs, masks)
        for cls in list(range(classes_value)):
            for k in ["iou","dice","acc","prec","rec","f1"]:
                train_metrics[cls][k].extend(batch_metrics[cls][k])

    # epoch 평균 계산
    avg_train_metrics = average_metrics(train_metrics)
    avg_train_loss = epoch_loss / len(train_loader)
    
    log_metrics["loss"]["train"].append(avg_train_loss)
    
    for cls in list(range(classes_value)):
        for k in ["iou","dice","acc","prec","rec","f1"]:
            log_metrics["train"][cls][k].append(avg_train_metrics[cls][k])

    # 전체 평균 metric
    log_metrics["train_avg"]["mIoU"].append(np.mean([avg_train_metrics[cls]["iou"] for cls in [0,1]]))
    log_metrics["train_avg"]["mDice"].append(np.mean([avg_train_metrics[cls]["dice"] for cls in [0,1]]))
    log_metrics["train_avg"]["mPA"].append(np.mean([avg_train_metrics[cls]["acc"] for cls in [0,1]]))
    log_metrics["train_avg"]["mPrecision"].append(np.mean([avg_train_metrics[cls]["prec"] for cls in [0,1]]))
    log_metrics["train_avg"]["mRecall"].append(np.mean([avg_train_metrics[cls]["rec"] for cls in [0,1]]))
    log_metrics["train_avg"]["mF1"].append(np.mean([avg_train_metrics[cls]["f1"] for cls in [0,1]]))
    
    # =========================
    # Validation
    # =========================
    model.eval()
    val_metrics = {cls: {k: [] for k in ["iou","dice","acc","prec","rec","f1"]} for cls in [0,1]}
    val_loss = 0.0
    
    with torch.no_grad():
        for images, masks in valid_loader:
            images, masks = images.cuda(), masks.cuda()
            outputs = model(images)
            val_loss += loss_fn(outputs, masks).item()

            batch_metrics = compute_metrics_per_batch(outputs, masks)
            for cls in [0,1]:
                for k in ["iou","dice","acc","prec","rec","f1"]:
                    val_metrics[cls][k].extend(batch_metrics[cls][k])

    avg_val_metrics = average_metrics(val_metrics)
    avg_val_loss = val_loss / len(valid_loader)
    
    log_metrics["loss"]["val"].append(avg_val_loss)
    

    for cls in list(range(classes_value)):
        for k in ["iou","dice","acc","prec","rec","f1"]:
            log_metrics["val"][cls][k].append(avg_val_metrics[cls][k])

    log_metrics["val_avg"]["mIoU"].append(np.mean([avg_val_metrics[cls]["iou"] for cls in [0,1]]))
    log_metrics["val_avg"]["mDice"].append(np.mean([avg_val_metrics[cls]["dice"] for cls in [0,1]]))
    log_metrics["val_avg"]["mPA"].append(np.mean([avg_val_metrics[cls]["acc"] for cls in [0,1]]))
    log_metrics["val_avg"]["mPrecision"].append(np.mean([avg_val_metrics[cls]["prec"] for cls in [0,1]]))
    log_metrics["val_avg"]["mRecall"].append(np.mean([avg_val_metrics[cls]["rec"] for cls in [0,1]]))
    log_metrics["val_avg"]["mF1"].append(np.mean([avg_val_metrics[cls]["f1"] for cls in [0,1]]))

    
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} Metrics Summary")
    print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    for cls, name in zip(list(range(classes_value)), ["BG","Disease"]):
        print(f"Class {name}:")
        print(f"Train IoU: {avg_train_metrics[cls]['iou']:.4f}, Dice: {avg_train_metrics[cls]['dice']:.4f}, "
              f"Acc: {avg_train_metrics[cls]['acc']:.4f}, Prec: {avg_train_metrics[cls]['prec']:.4f}, "
              f"Recall: {avg_train_metrics[cls]['rec']:.4f}, F1: {avg_train_metrics[cls]['f1']:.4f}")
        print(f"Val IoU: {avg_val_metrics[cls]['iou']:.4f}, Dice: {avg_val_metrics[cls]['dice']:.4f}, "
              f"Acc: {avg_val_metrics[cls]['acc']:.4f}, Prec: {avg_val_metrics[cls]['prec']:.4f}, "
              f"Recall: {avg_val_metrics[cls]['rec']:.4f}, F1: {avg_val_metrics[cls]['f1']:.4f}")

    # 전체 평균 출력
    print(f"Overall Average:")
    print(f"Train mIoU: {log_metrics['train_avg']['mIoU'][-1]:.4f}, mDice: {log_metrics['train_avg']['mDice'][-1]:.4f}, "
          f"mPA: {log_metrics['train_avg']['mPA'][-1]:.4f}, mPrecision: {log_metrics['train_avg']['mPrecision'][-1]:.4f}, "
          f"mRecall: {log_metrics['train_avg']['mRecall'][-1]:.4f}, mF1: {log_metrics['train_avg']['mF1'][-1]:.4f}")
    print(f"Val mIoU: {log_metrics['val_avg']['mIoU'][-1]:.4f}, mDice: {log_metrics['val_avg']['mDice'][-1]:.4f}, "
          f"mPA: {log_metrics['val_avg']['mPA'][-1]:.4f}, mPrecision: {log_metrics['val_avg']['mPrecision'][-1]:.4f}, "
          f"mRecall: {log_metrics['val_avg']['mRecall'][-1]:.4f}, mF1: {log_metrics['val_avg']['mF1'][-1]:.4f}\n")
    
    
    # =========================
    # Checkpoint 공통 dict
    # =========================
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "epoch": epoch+1,
        "metrics": log_metrics
    }

    # =========================
    # 1) 최소 validation loss
    # =========================
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        loss_ckpt_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_best_loss_{current_time_str}.pth"
        )
        torch.save(checkpoint, loss_ckpt_path)
        print(f"Best-loss checkpoint saved (val_loss={best_val_loss:.4f})")

    # =========================
    # 2) 최고 validation mIoU
    # =========================
    avg_val_mIoU = np.mean([avg_val_metrics[cls]["iou"] for cls in [0,1]])
    if avg_val_mIoU > best_val_iou:
        best_val_iou = avg_val_mIoU
        iou_ckpt_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_best_mIoU_{current_time_str}.pth"
        )
        torch.save(checkpoint, iou_ckpt_path)
        print(f"Best-mIoU checkpoint saved (val_mIoU={best_val_iou:.4f})")
        
    # =========================
    # 3) 최고 Disease IoU 기준 checkpoint 저장
    # =========================
    disease_class_iou = avg_val_metrics[1]["iou"]  # 클래스 1 = disease
    if disease_class_iou > best_disease_iou:
        best_disease_iou = disease_class_iou
        disease_ckpt_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_best_diseaseIoU_{current_time_str}.pth"
        )
        torch.save(checkpoint, disease_ckpt_path)
        print(f"Best-disease-IoU checkpoint saved (val_disease_IoU={best_disease_iou:.4f}")


#     # =========================
#     # EarlyStopping
#     # =========================
#     earlystop.step(avg_iou_disease, model)
#     if earlystop.should_stop:
#         print(f"Early stopping triggered at epoch {epoch+1}")
#         break

    print_gpu_usage()

last_ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_last_epoch_{NUM_EPOCHS}_{current_time_str}.pth")
torch.save(checkpoint, last_ckpt_path)
print(f"Last checkpoint saved at {last_ckpt_path}")


# =========================
# 학습 종료 후 시각화
# =========================
import matplotlib.pyplot as plt

epochs_range = range(1, NUM_EPOCHS+1)

plt.figure(figsize=(16,5))

# Loss
plt.subplot(1,3,1)
plt.plot(epochs_range, log_metrics["loss"]["train"], label="Train Loss")
plt.plot(epochs_range, log_metrics["loss"]["val"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss"); plt.legend()

# Disease IoU
plt.subplot(1,3,2)
plt.plot(epochs_range, log_metrics["train"][1]["iou"], label="Train IoU")
plt.plot(epochs_range, log_metrics["val"][1]["iou"], label="Val IoU")
plt.xlabel("Epoch"); plt.ylabel("IoU"); plt.title("Disease IoU"); plt.legend()
plt.ylim(0, 1)

# Disease Dice
plt.subplot(1,3,3)
plt.plot(epochs_range, log_metrics["train"][1]["dice"], label="Train Dice")
plt.plot(epochs_range, log_metrics["val"][1]["dice"], label="Val Dice")
plt.xlabel("Epoch"); plt.ylabel("Dice"); plt.title("Disease Dice"); plt.legend()
plt.ylim(0, 1)

plt.tight_layout()
metrics_image_path = os.path.join(checkpoint_dir, "train_val_metrics.png")
plt.savefig(metrics_image_path, dpi=300)
plt.show()
plt.close()


print(f"✅ Epoch별 학습/검증 지표 이미지 저장 완료: {metrics_image_path}")

# =========================
# 2. Training/Validation metrics CSV 저장
# =========================
metrics_df = pd.DataFrame({
    "Epoch": list(range(1, NUM_EPOCHS+1)),
    "Train_Loss": log_metrics["loss"]["train"],
    "Val_Loss": log_metrics["loss"]["val"],

    # 클래스별
    "Train_Disease_IoU": log_metrics["train"][1]["iou"],
    "Val_Disease_IoU": log_metrics["val"][1]["iou"],
    "Train_Disease_Dice": log_metrics["train"][1]["dice"],
    "Val_Disease_Dice": log_metrics["val"][1]["dice"],
    "Train_Disease_Accuracy": log_metrics["train"][1]["acc"],
    "Val_Disease_Accuracy": log_metrics["val"][1]["acc"],
    "Train_Disease_Precision": log_metrics["train"][1]["prec"],
    "Val_Disease_Precision": log_metrics["val"][1]["prec"],
    "Train_Disease_Recall": log_metrics["train"][1]["rec"],
    "Val_Disease_Recall": log_metrics["val"][1]["rec"],
    "Train_Disease_F1": log_metrics["train"][1]["f1"],
    "Val_Disease_F1": log_metrics["val"][1]["f1"],

    "Train_BG_IoU": log_metrics["train"][0]["iou"],
    "Val_BG_IoU": log_metrics["val"][0]["iou"],
    "Train_BG_Dice": log_metrics["train"][0]["dice"],
    "Val_BG_Dice": log_metrics["val"][0]["dice"],
    "Train_BG_Accuracy": log_metrics["train"][0]["acc"],
    "Val_BG_Accuracy": log_metrics["val"][0]["acc"],
    "Train_BG_Precision": log_metrics["train"][0]["prec"],
    "Val_BG_Precision": log_metrics["val"][0]["prec"],
    "Train_BG_Recall": log_metrics["train"][0]["rec"],
    "Val_BG_Recall": log_metrics["val"][0]["rec"],
    "Train_BG_F1": log_metrics["train"][0]["f1"],
    "Val_BG_F1": log_metrics["val"][0]["f1"],

    # sangjin update: 전체 평균 metric
    "Train_mIoU": log_metrics["train_avg"]["mIoU"],
    "Val_mIoU": log_metrics["val_avg"]["mIoU"],
    "Train_mDice": log_metrics["train_avg"]["mDice"],
    "Val_mDice": log_metrics["val_avg"]["mDice"],
    "Train_mPA": log_metrics["train_avg"]["mPA"],
    "Val_mPA": log_metrics["val_avg"]["mPA"],
    "Train_mPrecision": log_metrics["train_avg"]["mPrecision"],
    "Val_mPrecision": log_metrics["val_avg"]["mPrecision"],
    "Train_mRecall": log_metrics["train_avg"]["mRecall"],
    "Val_mRecall": log_metrics["val_avg"]["mRecall"],
    "Train_mF1": log_metrics["train_avg"]["mF1"],
    "Val_mF1": log_metrics["val_avg"]["mF1"],
})

metrics_csv_path = os.path.join(checkpoint_dir, f"train_val_metrics_{current_time_str}.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"Metrics CSV saved: {metrics_csv_path}")


# =========================
# 전체 학습 종료 시간 기록
# =========================
end_time = time.time()  # sangjin update
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print()
print()
print("======================================================")
print(f"⏱️ 전체 학습 시간: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
print("======================================================")


# # =========================
# # 0. 기본 설정
# # =========================

# # =========================
# # 전체 학습 시작 시간 기록
# # =========================
# start_time = time.time()

# seed = 18
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# NUM_EPOCHS = 200
# learning_rate = 1e-4

# # base_dir = "/BiO/Sangjin_Ko_work_2022/51_25.12.22_AI_MODEL/"
# # train_folder = "000_Dataset/000_26.01.05_NO_AUG_DATASET"
# base_dir = "./"
# train_folder = "00000_ALL_JSON_LABELED-FINAL-NEW"


# image_size_value = 512
# encoder_model_value = "resnet50"
# encoder_weights_value = "imagenet"
# in_channel_value = 3
# classes_value = 2


# #original_context = ssl._create_default_https_context
# #ssl._create_default_https_context = ssl._create_unverified_context

# MODEL = "PSPNet"
# model = smp.PSPNet(encoder_name=encoder_model_value, encoder_weights=encoder_weights_value, in_channels=in_channel_value, classes=classes_value).cuda()
# print(f"Download completed")

# #ssl._create_default_https_context = original_context
# #print("SSL verification restored")

# PARAMETER = MODEL + "_Leaf_and_Disease_" + str(NUM_EPOCHS) + "epoch"
# #earlystop = EarlyStopping(patience=10, min_delta=0.00001)

# TRAIN_BATCH_SIZE = 8
# VALID_BATCH_SIZE = 8
# TEST_BATCH_SIZE = 8

# loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Train / Validation / Test 경로
# train_images_dir = os.path.join(base_dir, train_folder, "000_TRAIN_images_NEW")
# train_json_dir   = os.path.join(base_dir, train_folder, "000_TRAIN_json_NEW")
# valid_images_dir = os.path.join(base_dir, train_folder, "000_VALID_images_NEW")
# valid_json_dir   = os.path.join(base_dir, train_folder, "000_VALID_json_NEW")
# test_images_dir  = os.path.join(base_dir, train_folder, "000_TEST_images_NEW")
# test_json_dir    = os.path.join(base_dir, train_folder, "000_TEST_json_NEW")

# checkpoint_dir = base_dir + "/" + train_folder + "/" + MODEL +"_LESION_SEG_MODEL_checkpoints_" + current_time_str
# #os.makedirs(checkpoint_dir, exist_ok=True)


# # =========================
# # 1. Dataset 준비 (image / json 폴더 분리)
# # =========================
# def prepare_data(images_dir, json_dir):
#     image_files = sorted([
#         f for f in os.listdir(images_dir)
#         if f.lower().endswith((".jpg", ".png", ".jpeg"))
#     ])

#     data_list = []
#     for img_name in image_files:
#         json_name = os.path.splitext(img_name)[0] + ".json"
#         img_path = os.path.join(images_dir, img_name)
#         json_path = os.path.join(json_dir, json_name)

#         if os.path.exists(json_path):
#             data_list.append({
#                 "image": img_path,
#                 "annotation": json_path
#             })
#     return data_list
    
# train_data = prepare_data(train_images_dir, train_json_dir)
# valid_data = prepare_data(valid_images_dir, valid_json_dir)
# test_data  = prepare_data(test_images_dir,  test_json_dir)


# # =========================
# # 2. Preprocessing & Dataset
# # =========================
# preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_model_value, encoder_weights_value)

# def json_to_mask_leaf_lesion(json_path, shape):
#     leaf_mask = np.zeros(shape, dtype=np.uint8)
#     lesion_mask = np.zeros(shape, dtype=np.uint8)

#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     for shape_obj in data.get("shapes", []):
#         label = shape_obj.get("label", "").lower()
#         points = np.array(shape_obj.get("points", []), dtype=np.int32)
#         if points.size == 0:
#             continue
#         if label == "leaf":
#             cv2.fillPoly(leaf_mask, [points], 1)
#         elif label in ["disease", "lesion"]:   # sangjin update
#             cv2.fillPoly(lesion_mask, [points], 1)

#     return leaf_mask, lesion_mask  # sangjin update


# class ModelDataset(torch.utils.data.Dataset):
#     def __init__(self, data_list, image_size=image_size_value):
#         self.data_list = data_list
#         self.image_size = image_size
#         self.preprocessing_fn = preprocessing_fn
#     def __len__(self):
#         return len(self.data_list)
#     def __getitem__(self, idx):
#         ent = self.data_list[idx]
#         Img = cv2.imread(ent["image"])[..., ::-1] # BGR > RGB convert
#         H, W = Img.shape[:2]

#         leaf_mask, lesion_mask = json_to_mask_leaf_lesion(ent["annotation"], (H, W))

#         Img = cv2.resize(Img, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
#         leaf_mask = cv2.resize(leaf_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
#         leaf_mask = np.round(leaf_mask).astype(np.uint8)
        
#         lesion_mask = cv2.resize(lesion_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
#         lesion_mask = np.round(lesion_mask).astype(np.uint8)

#         valid_region = (leaf_mask == 1) | (lesion_mask == 1)

#         mask = np.full((self.image_size, self.image_size), 255, dtype=np.uint8)
#         mask[valid_region] = 0
#         mask[lesion_mask == 1] = 1
        
#         Img = torch.from_numpy(self.preprocessing_fn(Img)).permute(2,0,1).float()
#         mask = torch.from_numpy(mask).long()

#         return Img, mask        


# train_dataset = ModelDataset(train_data, image_size=image_size_value)
# valid_dataset = ModelDataset(valid_data, image_size=image_size_value)
# test_dataset  = ModelDataset(test_data, image_size=image_size_value)

# train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
# valid_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# # 전처리/데이터 확인 코드
# for imgs, masks in train_loader:
#     print("mask shape:", masks.shape, "dtype:", masks.dtype, "unique:", torch.unique(masks))
#     break

# # IoU 계산
# def compute_iou_per_image(preds, targets, cls):
#     #preds = torch.argmax(preds, dim=1)
#     ious = []
#     for p, t in zip(preds, targets):
#         valid = (t != 255) # ignore pixel 제외
#         p_cls = (p == cls) & valid
#         t_cls = (t == cls) & valid

#         pred_has = p_cls.sum() > 0
#         gt_has   = t_cls.sum() > 0

#         if not pred_has and not gt_has:
#             ious.append(1.0)  # case 1
#         elif not pred_has and gt_has:
#             ious.append(0.0)  # case 2
#         elif pred_has and not gt_has:
#             ious.append(0.0)  # case 3
#         else:
#             intersection = (p_cls & t_cls).sum().float()
#             union = (p_cls | t_cls).sum().float() + 1e-6
#             ious.append((intersection / union).item())  # case 4
#     return ious        


# def compute_dice_per_image(preds, targets, cls):
#     #preds = torch.argmax(preds, dim=1)
#     dices = []
#     for p, t in zip(preds, targets):
#         valid = (t != 255)
#         preds_cls = ((p == cls) & valid).float()
#         targets_cls = ((t == cls) & valid).float()
#         intersection = (preds_cls * targets_cls).sum()
        
#         if targets_cls.sum() == 0 and preds_cls.sum() == 0:  # target, pred 모두 없음
#             dices.append(1.0)
#         elif targets_cls.sum() == 0:  # target 없음, pred 있음
#             dices.append(0.0)
#         else:
#             dice = (2.0 * intersection + 1e-6) / (preds_cls.sum() + targets_cls.sum() + 1e-6)
#             dices.append(dice.item())
#     return dices


# def compute_pixel_accuracy_per_image(preds, targets, cls):
#     #preds = torch.argmax(preds, dim=1)
#     accs = []
#     for p, t in zip(preds, targets):
#         valid = (t != 255)
#         mask = (t == cls) & valid
#         pred_mask = (p == cls) & valid
        
#         if mask.sum() == 0 and pred_mask.sum() == 0:  # target, pred 모두 없음
#             accs.append(1.0)
#         elif mask.sum() == 0:  # target 없음, pred 있음
#             accs.append(0.0)
#         else:
#             correct = (p == cls) & mask
#             accs.append(correct.float().sum().item() / mask.sum().item())
#     return accs


# def compute_precision_per_image(preds, targets, cls):
#     #preds = torch.argmax(preds, dim=1)
#     precisions = []
#     for p, t in zip(preds, targets):
#         valid = (t != 255)
#         preds_cls = ((p == cls) & valid).float()
#         targets_cls = ((t == cls) & valid).float()
#         TP = (preds_cls * targets_cls).sum()
#         FP = ((preds_cls == 1) & (targets_cls == 0)).sum()

#         if targets_cls.sum() == 0 and preds_cls.sum() == 0:
#             precisions.append(1.0)
#         elif preds_cls.sum() == 0:  # pred 없음
#             precisions.append(1.0 if targets_cls.sum() == 0 else 0.0)
#         else:
#             precisions.append((TP + 1e-6) / (TP + FP + 1e-6))
#     return precisions


# def compute_recall_per_image(preds, targets, cls):
#     #preds = torch.argmax(preds, dim=1)
#     recalls = []
#     for p, t in zip(preds, targets):
#         valid = (t != 255)
#         preds_cls = ((p == cls) & valid).float()
#         targets_cls = ((t == cls) & valid).float()
#         TP = (preds_cls * targets_cls).sum()
#         FN = ((preds_cls == 0) & (targets_cls == 1)).sum()

#         if targets_cls.sum() == 0 and preds_cls.sum() == 0:
#             recalls.append(1.0)
#         elif targets_cls.sum() == 0:  # target 없음, pred 있음
#             recalls.append(0.0)
#         else:
#             recalls.append((TP + 1e-6) / (TP + FN + 1e-6))
#     return recalls


# def compute_f1_from_prec_recall(precision_list, recall_list):
#     f1_list = []
#     for p, r in zip(precision_list, recall_list):
#         # p, r가 Tensor이면 float로 변환
#         if isinstance(p, torch.Tensor):
#             p = p.item()
#         if isinstance(r, torch.Tensor):
#             r = r.item()
#         f1 = (2 * p * r + 1e-6) / (p + r + 1e-6)
#         f1_list.append(f1)
#     return f1_list


# def to_cpu_list(tensor_list):
#     return [t.item() if torch.is_tensor(t) else t for t in tensor_list]

# def compute_metrics_per_batch(outputs, targets, classes=[0,1]):
#     preds = torch.argmax(outputs, dim=1)
#     batch_metrics = {cls:{k: [] for k in ["iou","dice","acc","prec","rec","f1"]} for cls in classes}
    
#     for cls in classes:
#         iou = compute_iou_per_image(preds, targets, cls)
#         dice = compute_dice_per_image(preds, targets, cls)
#         acc = compute_pixel_accuracy_per_image(preds, targets, cls)
#         prec = compute_precision_per_image(preds, targets, cls)
#         rec = compute_recall_per_image(preds, targets, cls)
#         f1  = compute_f1_from_prec_recall(prec, rec)
        
#         batch_metrics[cls]["iou"].extend(to_cpu_list(iou))
#         batch_metrics[cls]["dice"].extend(to_cpu_list(dice))
#         batch_metrics[cls]["acc"].extend(to_cpu_list(acc))
#         batch_metrics[cls]["prec"].extend(to_cpu_list(prec))
#         batch_metrics[cls]["rec"].extend(to_cpu_list(rec))
#         batch_metrics[cls]["f1"].extend(to_cpu_list(f1))
#     return batch_metrics

# def average_metrics(metrics_dict):
#     avg_metrics = {}
#     for cls, cls_metrics in metrics_dict.items():
#         avg_metrics[cls] = {k: np.mean(v) if len(v) > 0 else 0.0 for k,v in cls_metrics.items()}
#     return avg_metrics

# def print_gpu_usage():
#     if torch.cuda.is_available():
#         allocated = torch.cuda.memory_allocated(0) / 1024**3
#         cached = torch.cuda.memory_reserved(0) / 1024**3
#         print(f"GPU 메모리 사용: {allocated:.2f} GB, 캐시 사용: {cached:.2f} GB")

########################################################################################################################################################################

# 만약 kernel memory 날아간 경우 위 스크립트 실행 # 


# =========================
# Validation 전체 prediction & overlay 시각화
# =========================
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 기본 설정
# =========================
alpha = 0.5
classes = [0, 1]          # 0: BG(Leaf), 1: Disease
ignore_index = 255

class_colors = {
    0: [0, 0, 0],       # BG: Red
    1: [255, 0, 0],       # Disease: Green
}

prefixes = [
    "Bellpepper_BS", "Tomato_BS", "Cherry_PM", "Pumpkin_PM",
    "Potato_EB", "Tomato_EB", "Potato_LB", "Tomato_LB"
]

# Prefix별 결과 저장
prefix_results = {
    p: {c: {"iou": [], "acc": [], "dice": []} for c in classes}
    for p in prefixes
}

path_parameter = "_LESION_SEG_MODEL_checkpoints_20260107_194311"

# =========================
# Checkpoint 로드
# =========================
best_loss_ckpt_path = os.path.join(
    base_dir,
    train_folder,
    f"{MODEL}{path_parameter}/"
    "checkpoint_best_diseaseIoU_20260107_194311.pth"
)

save_vis_path = os.path.join(
    base_dir,
    train_folder,
    f"{MODEL}{path_parameter}/{PARAMETER}_prediction_vis_output"
)
os.makedirs(save_vis_path, exist_ok=True)

checkpoint = torch.load(best_loss_ckpt_path, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print(f"✅ Loaded checkpoint: {best_loss_ckpt_path}")

# =========================
# Metric 저장용 (Dice 추가)
# =========================
class_iou_list = {c: [] for c in classes}
class_acc_list = {c: [] for c in classes}
class_dice_list = {c: [] for c in classes}  # sangjin update

# =========================
# Validation Loop
# =========================
with torch.no_grad():
    for batch_idx, (images, masks) in enumerate(valid_loader):
        images = images.to(device)
        masks  = masks.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        images_np = images.cpu().numpy()
        preds_np  = preds.cpu().numpy()
        masks_np  = masks.cpu().numpy()

        batch_size = images.shape[0]

        for i in range(batch_size):
            global_idx = batch_idx * valid_loader.batch_size + i
            if global_idx >= len(valid_dataset):
                continue

            img_path = valid_dataset.data_list[global_idx]["image"]
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            # 이미지 복원
            image = images_np[i].transpose(1, 2, 0)
            if preprocessing_fn is not None:
                mean = np.array([0.485, 0.456, 0.406])
                std  = np.array([0.229, 0.224, 0.225])
                image = (image * std + mean) * 255
            image = np.clip(image, 0, 255).astype(np.uint8)

            gt_mask   = masks_np[i]
            pred_mask = preds_np[i]
            H, W = gt_mask.shape
            valid_region = (gt_mask != ignore_index)

            # Prefix 매칭
            matched_prefix = next((p for p in prefixes if img_name.startswith(p)), None)

            # GT Mask 색상화
            gt_color = np.zeros((H, W, 3), dtype=np.uint8)
            for c in classes:
                gt_color[gt_mask == c] = class_colors[c]

            # Metric 계산 (IoU, Acc, Dice 추가)
            for c in classes:
                pred_c = (pred_mask == c) & valid_region
                gt_c   = (gt_mask == c) & valid_region

                inter = np.logical_and(pred_c, gt_c).sum()
                union = np.logical_or(pred_c, gt_c).sum()
                iou = inter / union if union > 0 else 0.0

                acc = (pred_c == gt_c).sum() / valid_region.sum() if valid_region.sum() > 0 else 0.0

                pred_sum = pred_c.sum()
                gt_sum   = gt_c.sum()
                dice = (2 * inter) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0  # sangjin update

                class_iou_list[c].append(iou)
                class_acc_list[c].append(acc)
                class_dice_list[c].append(dice)  # sangjin update

                if matched_prefix is not None:
                    prefix_results[matched_prefix][c]["iou"].append(iou)
                    prefix_results[matched_prefix][c]["acc"].append(acc)
                    prefix_results[matched_prefix][c].setdefault("dice", []).append(dice)  # sangjin update

            # 이미지별 metric 출력
            print(
                f"[{global_idx+1}] {img_name} | "
                f"BG IoU: {class_iou_list[0][-1]:.4f}, Dice: {class_dice_list[0][-1]:.4f}, "
                f"Disease IoU: {class_iou_list[1][-1]:.4f}, Dice: {class_dice_list[1][-1]:.4f}"
            )

            # Prediction Overlay 생성 (lesion 빨강)
            overlay = image.copy()
            mask_region = (pred_mask == 1) & valid_region
            color_mask = np.zeros((H, W, 3), dtype=np.uint8)
            color_mask[mask_region] = class_colors[1]
            overlay = cv2.addWeighted(overlay, 1-alpha, color_mask, alpha, 0)

            save_path = os.path.join(save_vis_path, f"{img_name}_pred_overlay.png")
            cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            # =========================
            # 시각화: Original / GT / Prediction Overlay (1x3)
            # =========================            
            plt.figure(figsize=(9, 3))  # 9,3 크기

            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(gt_color)
            plt.title("GT Mask")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title("Prediction Overlay")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

# =========================
# 전체 평균 Metric
# =========================
print("\n======================")
print(" Validation Set Evaluation ")
print("======================")
for c, name in zip(classes, ["BG", "Disease"]):
    print(
        f"{name} - Avg IoU: {np.mean(class_iou_list[c]):.4f}, "
        f"Avg Dice: {np.mean(class_dice_list[c]):.4f}, "
        f"Pixel Acc: {np.mean(class_acc_list[c]):.4f}"
    )

# =========================
# Prefix별 Metric
# =========================
print("\n======================")
print(" Prefix별 Validation Set Evaluation ")
print("======================")
for p in prefixes:
    for c, name in zip(classes, ["BG", "Disease"]):
        iou_avg = np.mean(prefix_results[p][c]["iou"]) if prefix_results[p][c]["iou"] else 0
        acc_avg = np.mean(prefix_results[p][c]["acc"]) if prefix_results[p][c]["acc"] else 0
        dice_avg = np.mean(prefix_results[p][c]["dice"]) if "dice" in prefix_results[p][c] else 0
        print(f"[{p}] {name} - IoU: {iou_avg:.4f}, Dice: {dice_avg:.4f}, Acc: {acc_avg:.4f}")

