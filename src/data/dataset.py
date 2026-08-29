"""WFLW torch Dataset. 학습/평가가 동일한 crop 로직을 공유하도록 여기 모음.

좌표계 규약(전 파이프라인 일관):
  - 모델 타깃/출력: crop 기준 [0,1] 정규화 좌표.
  - NME 계산: [0,1] -> *size -> crop affine 역변환 -> 원본 픽셀 좌표.
"""
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from src.data.wflw_annotation_repository import parse_annotation
from src.shared.crop_geometry import crop_and_normalize, apply_affine


class WFLWDataset(Dataset):
    def __init__(self, txt, images_root, size=128, train=False, subsample=1.0,
                 seed=42):
        """WFLW 어노테이션을 읽어 학습/평가용 Dataset을 구성

        Args:
            txt: list_98pt_rect_attr_{train,test}.txt 경로
            images_root: 이미지 루트 디렉터리
            size: crop 정사각 한 변 길이
            train: True면 __getitem__에서 augmentation 적용
            subsample: 전체 대비 사용할 비율(1.0이면 전체 사용)
            seed: subsample에 쓰이는 random seed
        """
        ann = parse_annotation(txt, images_root)
        self.lms = ann["landmarks"]; self.boxes = ann["bboxes"]
        self.paths = ann["paths"]; self.attrs = ann["attrs"]
        self.masks = ann["subset_masks"]
        self.size = size; self.train = train
        if subsample < 1.0:  # Part 지시: 서브샘플 명시. seed 고정.
            rng = np.random.default_rng(seed)
            n = int(len(self.paths) * subsample)
            idx = rng.choice(len(self.paths), n, replace=False)
            self.lms, self.boxes = self.lms[idx], self.boxes[idx]
            self.attrs = self.attrs[idx]
            self.paths = [self.paths[i] for i in idx]
            self.masks = {k: v[idx] for k, v in self.masks.items()}

    def __len__(self):
        return len(self.paths)

    def _augment(self, crop, pts01):
        """가벼운 증강: 밝기 jitter + random cutout(occlusion 모사).
        좌우 flip은 점 인덱스 교환이 필요해 여기선 안 함.

        Args:
            crop: crop 이미지(HWC uint8)
            pts01: (98,2) crop 기준 [0,1] 정규화 좌표

        Returns:
            (증강된 crop, pts01). 좌표 자체는 변형하지 않으므로 그대로 반환
        """
        if np.random.rand() < 0.5:
            f = 0.7 + 0.6 * np.random.rand()
            crop = np.clip(crop.astype(np.float32) * f, 0, 255).astype(np.uint8)
        if np.random.rand() < 0.3:  # random cutout(가림 모사)
            h, w = crop.shape[:2]
            cw = int(w * (0.1 + 0.2 * np.random.rand()))
            x = np.random.randint(0, w - cw); y = np.random.randint(0, h - cw)
            crop[y:y+cw, x:x+cw] = 0
        return crop, pts01

    def __getitem__(self, i):
        """crop을 로드하고 좌표를 crop 기준 [0,1]로 변환

        Args:
            i: 인덱스

        Returns:
            (x, y, i): x는 crop 텐서(CHW float [0,1]), y는 (98,2) crop 기준
            [0,1] 정규화 좌표, i는 원본 인덱스(평가 시 역변환/subset 참조용)
        """
        img = cv2.imread(self.paths[i])
        crop, M = crop_and_normalize(img, self.boxes[i], self.size)
        pts_crop = apply_affine(self.lms[i], M)      # crop 픽셀 좌표
        pts01 = pts_crop / self.size                 # [0,1]
        if self.train:
            crop, pts01 = self._augment(crop, pts01)
        x = torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) \
                 .permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(pts01).float()
        return x, y, i
