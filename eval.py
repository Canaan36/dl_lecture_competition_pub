import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
from scipy.signal import resample, butter, filtfilt

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed

def preprocess_data(X, resample_rate=250, lowcut=0.5, highcut=40.0, fs=1000.0):
    # 重采样
    X_resampled = resample(X, int(X.shape[1] * resample_rate / fs), axis=1)
    
    # 滤波
    nyquist = 0.5 * resample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype='band')
    X_filtered = filtfilt(b, a, X_resampled, axis=1)
    
    # 缩放
    X_scaled = (X_filtered - np.mean(X_filtered, axis=1, keepdims=True)) / np.std(X_filtered, axis=1, keepdims=True)
    
    # 基线校正
    baseline = np.mean(X_scaled[:, :int(0.2 * resample_rate)], axis=1, keepdims=True)
    X_corrected = X_scaled - baseline
    
    # 转换数据类型为 Float
    X_corrected = X_corrected.astype(np.float32)
    
    return X_corrected

def preprocess_dataset(dataset, has_subject_idxs=True):
    processed_data = []
    for data in tqdm(dataset, desc="Preprocessing data"):
        if has_subject_idxs and len(data) == 2:
            X, subject_idxs = data
            X = preprocess_data(X)
            processed_data.append((X, subject_idxs))
        else:
            X = data
            X = preprocess_data(X)
            processed_data.append((X,))
    return processed_data

@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    savedir = os.path.dirname(args.model_path)
    
    # 手动设置 data_dir 绝对路径
    args.data_dir = "/content/drive/MyDrive/DL/最終課題/dl_lecture_competition_pub-MEG-competition/data"
    
    # ------------------
    #    Dataloader
    # ------------------   
    print("Loading test set...")
    test_set = ThingsMEGDataset("test", args.data_dir)
    print(f"Test set loaded with {len(test_set)} samples")
    
    num_classes = test_set.num_classes
    seq_len = test_set.seq_len
    num_channels = test_set.num_channels
    num_subjects = len(torch.unique(test_set.subject_idxs))
    
    test_set = preprocess_dataset(test_set, has_subject_idxs=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    model = BasicConvClassifier(
        num_classes, seq_len, num_channels, num_subjects
    ).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    # ------------------
    #  Start evaluation
    # ------------------ 
    preds = [] 
    model.eval()
    #for X, _, subject_idxs in tqdm(test_loader, desc="Validation"):        
    #    preds.append(model(X.to(args.device)).detach().cpu())
    for batch in tqdm(test_loader, desc="Validation"):
        if len(batch) == 2:
            X, subject_idxs = batch
            subject_idxs = subject_idxs.to(args.device)
        else:
            X, = batch
            subject_idxs = None
            
        if subject_idxs is not None:
            preds.append(model(X.to(args.device), subject_idxs).detach().cpu())
        else:
            preds.append(model(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(savedir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {savedir}", "cyan")


if __name__ == "__main__":
    run()