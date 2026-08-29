"""Part B 학습 use case: train/val 분리 + early stopping + best 체크포인트 저장.

학습셋(train_txt)에서 val_split 비율만큼 떼어 early stopping 적용.
WFLW 테스트 분할은 Part A-D 평가 전용이라 여기서는 사용하지 않음. val_loss가
patience epoch 연속 개선되지 않으면 멈추고, 매 개선 시점마다 즉시 best
체크포인트 저장(중간에 프로세스가 죽어도 그 시점까지의 최선
가중치는 남도록).

실행 진입점: `src/train.py`(CLI) 또는 `src/notebooks/part_b.ipynb`(직접 호출).
"""
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import WFLWDataset


def subset_(ds, idx):
    """WFLWDataset 인스턴스를 idx 기준으로 제자리 필터링

    Args:
        ds: WFLWDataset 인스턴스
        idx: 선택할 인덱스 배열

    Returns:
        필터링된 동일 ds 인스턴스(제자리 수정)
    """
    ds.lms, ds.boxes = ds.lms[idx], ds.boxes[idx]
    ds.attrs = ds.attrs[idx]
    ds.paths = [ds.paths[i] for i in idx]
    ds.masks = {k: v[idx] for k, v in ds.masks.items()}
    return ds


def split_train_val(train_txt, images, size, seed, val_split=0.1, subsample=1.0):
    """train_txt 전체를 로드한 뒤 val_split만큼 먼저 떼어내고,
    남은 쪽에만 subsample을 적용(검증셋이 학습 subsample에 새지 않도록).

    Args:
        train_txt: WFLW train 어노테이션 텍스트 경로
        images: 이미지 루트 디렉터리
        size: crop 정사각 한 변 길이
        seed: 분할/subsample에 쓰이는 random seed
        val_split: 검증셋 비율
        subsample: val을 뗀 나머지(학습셋)에 적용할 subsample 비율

    Returns:
        (train_ds, val_ds) WFLWDataset 튜플
    """
    full = WFLWDataset(train_txt, images, size=size, train=False,
                       subsample=1.0, seed=seed)
    n = len(full)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_split))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    if subsample < 1.0:  # Part 지시: 서브샘플 시 반드시 명시. val을 뗀 뒤에 적용.
        k = int(len(tr_idx) * subsample)
        tr_idx = rng.choice(tr_idx, k, replace=False)

    val_ds = subset_(copy.deepcopy(full), val_idx)
    train_ds = subset_(full, tr_idx)
    train_ds.train = True  # augmentation은 학습 쪽에만
    return train_ds, val_ds


def run_epoch(net, dl, crit, opt, dev, train):
    """한 epoch을 학습 또는 평가 모드로 실행

    Args:
        net: 모델
        dl: DataLoader
        crit: 손실 함수
        opt: optimizer(train=False면 미사용)
        dev: 실행 디바이스
        train: True면 역전파 수행, False면 평가만 수행

    Returns:
        해당 epoch의 평균 손실
    """
    net.train(train)
    tot = 0.0
    with torch.set_grad_enabled(train):
        for x, y, _ in dl:
            x, y = x.to(dev), y.to(dev)
            if train:
                opt.zero_grad()
            loss = crit(net(x), y)
            if train:
                loss.backward(); opt.step()
            tot += loss.item() * x.size(0)
    return tot / len(dl.dataset)


def train_model(net, train_ds, val_ds, crit, dev, epochs, lr, batch_size,
                 ckpt_path, patience=8, log=print):
    """전체 학습 루프. best val_loss 체크포인트를 ckpt_path에 저장.

    Args:
        net: 모델
        train_ds: 학습 WFLWDataset
        val_ds: 검증 WFLWDataset
        crit: 손실 함수
        dev: 실행 디바이스
        epochs: 최대 epoch 수
        lr: 학습률
        batch_size: 배치 크기
        ckpt_path: best 체크포인트 저장 경로
        patience: early stopping patience
        log: 로그 출력 함수

    Returns:
        best_val_loss
    """
    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                    num_workers=4, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=4)
    log(f"[split] train={len(train_ds)}  val={len(val_ds)}")

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val, bad_epochs = float("inf"), 0
    for ep in range(epochs):
        train_loss = run_epoch(net, dl, crit, opt, dev, train=True)
        val_loss = run_epoch(net, val_dl, crit, opt, dev, train=False)
        sched.step()
        log(f"ep {ep+1:3d}/{epochs}  train_loss={train_loss:.5f}  "
            f"val_loss={val_loss:.5f}  lr={sched.get_last_lr()[0]:.2e}")

        if val_loss < best_val - 1e-5:
            best_val, bad_epochs = val_loss, 0
            torch.save(net.state_dict(), ckpt_path)
            log(f"  [best] val_loss={val_loss:.5f} -> saved {ckpt_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log(f"[early stop] val_loss 개선 없음({patience} epoch 연속) "
                    f"-> ep {ep+1}에서 중단, best val_loss={best_val:.5f}")
                break
    return best_val
