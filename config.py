import torch


class TrainConfig:
    num_workers = 2
    batch_size = 1
    n_epochs = 40
    lr = 0.0002

    folder = 'effdet-checkpoints'
    verbose = True
    verbose_step = 1
    step_scheduler = False
    validation_scheduler = True

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08
    )
    TRAIN_ROOT_PATH = r"data\train"
    TRAIN_CSV_PATH = r'data\train.csv'
    SEED = 42
    WEIGHT = r'weights\efficientdet_d5-ef44aea8.pth'
    EFFICIENTDET_CONFIG = 'tf_efficientdet_d5'
