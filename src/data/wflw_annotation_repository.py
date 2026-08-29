"""WFLW 어노테이션 파일(외부 포맷) 파싱. infrastructure 계층.

디스크의 텍스트 포맷을 읽어 numpy 배열로 변환하는 어댑터; application/shared는
이 함수가 반환하는 dict만 알면 됨. 라인 포맷(list_98pt_rect_attr_{train,test}.txt):
좌표 196 + bbox 4 + attribute 6 + 파일명 1 = 207 필드. attribute subset은
여기서 그대로 나옴(손으로 분할 불필요).
"""
import os
import numpy as np

ATTR_NAMES = ["pose", "expression", "illumination",
              "make-up", "occlusion", "blur"]


def parse_annotation(txt_path, images_root):
    """WFLW 어노테이션 텍스트를 읽어 landmark/bbox/attribute/경로로 변환

    Args:
        txt_path: list_98pt_rect_attr_{train,test}.txt 경로
        images_root: 이미지 루트 디렉터리 (라인의 image_name과 조합)

    Returns:
        dict(landmarks (N,98,2), bboxes (N,4), attrs (N,6) bool,
             paths list, subset_masks dict)
    """
    lms, boxes, attrs, paths = [], [], [], []
    with open(txt_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 207:
                continue
            coords = np.array(p[:196], float).reshape(98, 2)
            rect = np.array(p[196:200], float)
            a = np.array(p[200:206], int).astype(bool)
            name = p[206]
            lms.append(coords); boxes.append(rect); attrs.append(a)
            paths.append(os.path.join(images_root, name))
    lms = np.stack(lms); boxes = np.stack(boxes); attrs = np.stack(attrs)
    masks = {ATTR_NAMES[i]: attrs[:, i] for i in range(6)}
    return dict(landmarks=lms, bboxes=boxes, attrs=attrs,
                paths=paths, subset_masks=masks)
