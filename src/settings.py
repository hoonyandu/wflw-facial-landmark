import os

class DataConfig:
    root = f"{os.path.abspath('../..')}/data/WFLW"
    train_txt = f"{root}/annotations/list_98pt_rect_attr_train.txt"
    test_txt = f"{root}/annotations/list_98pt_rect_attr_test.txt"
    images = f"{root}/images"
    # 서브샘플 시 반드시 명시
    subsample = 1.0

class EvalConfig:
    # inter_ocular | inter_pupil | bbox_diag
    norm = "inter_ocular"
    failure_threshold = 0.10

class ModelConfig:
    input_size = 128
    pretrained = True

class TrainConfig:
    epochs = 60
    batch_size = 64
    lr = 1.0e-3
    loss = "wing"  # wing | smooth_l1
    val_split = 0.1
    patience = 8