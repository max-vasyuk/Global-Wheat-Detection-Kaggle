from config import TrainConfig
from dataset import get_datasets
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
import torch
from train import run_training
from utils import seed_everything


def get_net():
    config = get_efficientdet_config(TrainConfig.EFFICIENTDET_CONFIG)
    net = EfficientDet(config, pretrained_backbone=False)
    checkpoint = torch.load(TrainConfig.WEIGHT)
    net.load_state_dict(checkpoint)
    config.num_classes = 1
    config.image_size = 512
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    return DetBenchTrain(net, config)


if __name__ == '__main__':
    seed_everything(TrainConfig.SEED)
    net = get_net()
    train_dataset, validation_dataset = get_datasets()
    run_training(net, train_dataset, validation_dataset)