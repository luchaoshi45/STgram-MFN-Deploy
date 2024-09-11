import os
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from net import STgramMFN
from dataset import ASDDataset
import utils

from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import sklearn

from sklearn.mixture import GaussianMixture
from loss import ASDLoss
from trainer import Trainer

sep = os.sep

def infer(args):
    # set random seed
    utils.setup_seed(args.random_seed)
    # set device
    cuda = args.cuda
    device_ids = args.device_ids
    args.dp = False
    if not cuda or device_ids is None:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{device_ids[0]}')
        if len(device_ids) > 1: args.dp = True
    # load data
    train_dir = './data/valve/test'
    args.meta2label = {'fan-id_00': 0, 'fan-id_02': 1, 'fan-id_04': 2, 'fan-id_06': 3, 'pump-id_00': 4, 'pump-id_02': 5, 'pump-id_04': 6,
     'pump-id_06': 7, 'slider-id_00': 8, 'slider-id_02': 9, 'slider-id_04': 10, 'slider-id_06': 11, 'ToyCar-id_01': 12,
     'ToyCar-id_02': 13, 'ToyCar-id_03': 14, 'ToyCar-id_04': 15, 'ToyConveyor-id_01': 16, 'ToyConveyor-id_02': 17,
     'ToyConveyor-id_03': 18, 'valve-id_00': 19, 'valve-id_02': 20, 'valve-id_04': 21, 'valve-id_06': 22,
     'fan-id_01': 23, 'fan-id_03': 24, 'fan-id_05': 25, 'pump-id_01': 26, 'pump-id_03': 27, 'pump-id_05': 28,
     'slider-id_01': 29, 'slider-id_03': 30, 'slider-id_05': 31, 'ToyCar-id_05': 32, 'ToyCar-id_06': 33,
     'ToyCar-id_07': 34, 'ToyConveyor-id_04': 35, 'ToyConveyor-id_05': 36, 'ToyConveyor-id_06': 37, 'valve-id_01': 38,
     'valve-id_03': 39, 'valve-id_05': 40}
    args.label2meta = {0: 'fan-id_00', 1: 'fan-id_02', 2: 'fan-id_04', 3: 'fan-id_06', 4: 'pump-id_00', 5: 'pump-id_02', 6: 'pump-id_04',
     7: 'pump-id_06', 8: 'slider-id_00', 9: 'slider-id_02', 10: 'slider-id_04', 11: 'slider-id_06', 12: 'ToyCar-id_01',
     13: 'ToyCar-id_02', 14: 'ToyCar-id_03', 15: 'ToyCar-id_04', 16: 'ToyConveyor-id_01', 17: 'ToyConveyor-id_02',
     18: 'ToyConveyor-id_03', 19: 'valve-id_00', 20: 'valve-id_02', 21: 'valve-id_04', 22: 'valve-id_06',
     23: 'fan-id_01', 24: 'fan-id_03', 25: 'fan-id_05', 26: 'pump-id_01', 27: 'pump-id_03', 28: 'pump-id_05',
     29: 'slider-id_01', 30: 'slider-id_03', 31: 'slider-id_05', 32: 'ToyCar-id_05', 33: 'ToyCar-id_06',
     34: 'ToyCar-id_07', 35: 'ToyConveyor-id_04', 36: 'ToyConveyor-id_05', 37: 'ToyConveyor-id_06', 38: 'valve-id_01',
     39: 'valve-id_03', 40: 'valve-id_05'}

    train_file_list = []
    train_file_list.extend(utils.get_filename_list(train_dir))
    train_dataset = ASDDataset(args, train_file_list, load_in_memory=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
    # set model
    args.num_classes = len(args.meta2label.keys())
    args.logger.info(f'Num classes: {args.num_classes}')
    net = STgramMFN(num_classes=args.num_classes, use_arcface=args.use_arcface,
                    m=float(args.m), s=float(args.s), sub=args.sub_center)
    if args.dp:
        net = nn.DataParallel(net, device_ids=args.device_ids)
    net = net.to(args.device)
    # optimizer & scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=float(args.lr))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.1*float(args.lr))
    # trainer
    trainer = Trainer(args=args,
                      net=net,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      transform=train_dataset.transform)
    # train model
    if not args.load_epoch:
        trainer.train(train_dataloader)
    # test model
    load_epoch = args.load_epoch if args.load_epoch else 'best'
    model_path = os.path.join(args.writer.log_dir, 'model', f'{load_epoch}_checkpoint.pth.tar')
    if args.dp:
        trainer.net.module.load_state_dict(torch.load(model_path)['model'])
    else:
        if args.device == torch.device('cpu'):
            trainer.net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])
            # trainer.net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['clf_state_dict'])
        else:
            trainer.net.load_state_dict(torch.load(model_path)['model'])
            # trainer.net.load_state_dict(torch.load(model_path)['clf_state_dict'])

    net.eval()
    id_str = utils.get_machine_id_list(train_dir)
    assert len(id_str) == 1
    id_str = id_str[0]
    machine_type = 'valve'
    meta = machine_type + '-' + id_str
    label = args.meta2label[meta]
    test_files, y_true = utils.create_test_file_list(train_dir, id_str, dir_name='test')
    anomaly_score_list = []
    y_pred = [0. for _ in test_files]

    for file_idx, data in enumerate(tqdm(train_dataset, desc="infer")):
        x_wav, x_mel, label = data
        x_wav, x_mel = x_wav.unsqueeze(0).float().to(args.device), x_mel.unsqueeze(0).float().to(args.device)
        label = torch.tensor([label]).long().to(args.device)
        with torch.no_grad():
            predict_ids, feature = net(x_wav, x_mel, label)
        probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
        y_pred[file_idx] = probs[label]
        anomaly_score_list.append([os.path.basename(test_files[file_idx]), y_pred[file_idx]])

    # compute auc and pAuc
    max_fpr = 0.1
    y_true[0] = 0
    y_pred = [99 for i in y_pred]
    y_pred[0] = 95
    auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
    p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)

    # trainer.test(save=True, gmm_n=args.gmm_n)
    # trainer.evaluator(save=True, gmm_n=args.gmm_n)


def run():
    # init config parameters
    params = utils.load_yaml(file_path='./config.yaml')
    parser = argparse.ArgumentParser(description=params['description'])
    for key, value in params.items():
        parser.add_argument(f'--{key}', default=value, type=utils.set_type)
    args = parser.parse_args()
    # init logger and writer
    time_str = time.strftime('%Y-%m-%d-%H', time.localtime(time.time()))
    args.version = f'STgram-MFN(m={args.m},s={args.s})'
    args.version = f'{time_str}-{args.version}' if not args.load_epoch and args.time_version else args.version
    log_dir = f'runs/{args.version}'
    writer = SummaryWriter(log_dir=log_dir)
    logger = utils.get_logger(filename=os.path.join(log_dir, 'running.log'))
    # save version files
    if args.save_version_files: utils.save_load_version_files(log_dir, args.save_version_file_patterns, args.pass_dirs)
    # run
    args.writer, args.logger = writer, logger
    args.logger.info(args.version)
    infer(args)
    # save config file
    # utils.save_yaml_file(file_path=os.path.join(log_dir, 'config.yaml'), data=vars(args))


if __name__ == '__main__':
    run()