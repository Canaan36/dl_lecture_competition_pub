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

os.environ["WANDB_API_KEY"] = "7800d3aef9bafe37f184678f13dd93b8b703a567"

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
        if has_subject_idxs:
            X, y, subject_idxs = data
            X = preprocess_data(X)
            processed_data.append((X, y, subject_idxs))
        else:
            X, y = data
            X = preprocess_data(X)
            processed_data.append((X, y))
    return processed_data

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    # 打印 logdir 的值
    print(f"模型保存路径: {logdir}")
    
    # 手动设置 data_dir 绝对路径
    args.data_dir = "/content/drive/MyDrive/DL/最終課題/dl_lecture_competition_pub-MEG-competition/data"
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")
        wandb.config.update({
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size
        })

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    print("Loading train set...")
    train_set = ThingsMEGDataset("train", args.data_dir)
    print(f"Train set loaded with {len(train_set)} samples")
    #print("Train set sample:", train_set[0])
    num_classes = train_set.num_classes
    seq_len = train_set.seq_len
    num_channels = train_set.num_channels
    train_set = preprocess_dataset(train_set, has_subject_idxs=True)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    
    print("Loading validation set...")
    val_set = ThingsMEGDataset("val", args.data_dir)
    print(f"Validation set loaded with {len(val_set)} samples")
    #print("Validation set sample:", val_set[0])
    val_set = preprocess_dataset(val_set, has_subject_idxs=True)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    
    print("Loading test set...")
    test_set = ThingsMEGDataset("test", args.data_dir)
    print(f"Test set loaded with {len(test_set)} samples")
    #print("Test set sample:", test_set[0])
    test_set = preprocess_dataset(test_set, has_subject_idxs=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    #model = BasicConvClassifier(
    #    train_set.num_classes, train_set.seq_len, train_set.num_channels
    #).to(args.device)
    #model = BasicConvClassifier(train_set[0][0].shape[0], train_set[0][0].shape[1], len(train_set)).to(device)
    model = BasicConvClassifier(num_classes, seq_len, num_channels).to(device)
    print("Model initialized")

    # ------------------
    #     Optimizer
    # ------------------
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 添加 L2 正则化
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5) 
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    #accuracy = Accuracy(
    #    task="multiclass", num_classes=train_set.num_classes, top_k=10
    #).to(args.device)
    accuracy = Accuracy(task="multiclass", num_classes=num_classes, top_k=10).to(device)
    
    # 设置早停策略参数
    best_val_loss = float('inf')
    patience, trials = 30, 0
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)

            y_pred = model(X)
            
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        avg_val_loss = np.mean(val_loss)
        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if avg_val_loss < best_val_loss:
            cprint("New best (val loss).", "cyan")
            #torch.save(model.state_dict(), os.path.join(logdir, "model_best_loss.pt"))
            best_val_loss = avg_val_loss
            trials = 0
        else:
            trials += 1
            if trials >= patience:
                print("Early stopping triggered")
                break
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)
            
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
