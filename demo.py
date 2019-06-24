import os
import sys
import time 
import scipy.io
import torch
import torch.nn as nn

import datasets
import models.lsid as LSID
from trainer import Trainer, Validator
import utils
import tqdm

configurations = {
    1: dict(
            max_iteration=1000000,
            lr=1e-4,
            momentum=0.9,
            weight_decay=0.25,
            step_size=32300, # "lr_policy: step"
            interval_validate=1000,
    ),
}

def get_parameters(model, bias=False):
    for k, m in model._modules.items():
        print("get_parameters", k, type(m), type(m).__name__, bias)
        if bias:
            if isinstance(m, nn.Conv2d):
                yield m.bias
        else:
            is isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                yield m.weight

def main():
    params = {
            cmd: 'test',  # ['train', 'test']
            arch_type: 'Sony',  # camera model type
            dataset_dir: './dataset/',
            log_file: './log/Sony/test.log', 
            train_image_list_file: './dataset/Sony_train_list.txt',  # text file containing image file names for training
            valid_image_list_file: './dataset/Sony_val_list.txt',  # text file containing image file names for validation
            test_image_list_file: './dataset/Sony_test_list.txt',  # text file containing image file names for test
            gt_png: 'True',  # uses preconverted png file as ground truth
            use_camera_wb: 'True', # converts train RAW file to png
            valid_use_camera_wb: 'True', # converts valid RAW file to png
            checkpoint_dir: './checkpoint/Sony/',
            result_dir: './result/sony',
            config: 1,  # the number of setting and hyperparameters used in training
            batch_size: 1,
            valid_batch_size: 1,
            test_batch_size: 1,
            patch_size: None,
            save_freq: 1,  # checkpoint save frequency
            print_freq: 1,  # log print frequency
            upper_train: 1,  # max of train images(for debug)
            upper_valid: 1,  # max of valid images(for debug)
            upper_test: 1,  # max of test images(for debug)
            resume: '',  # checkpoint file(for trianing or testing)
            tf_weight_file: '',  # weight file ported from TensorFlow
            gpu: '0', 
            workers: 4,  # number of data loading workers
            pixel_shuffle: True,  # uses pixel_shuffle in training
            }

    if params['cmd'] == 'train':
        os.makedirs(params['checkpoint_dir'], exist_ok=True)
        cfg = configurations[params['config']]

    if params['cmd'] == 'test':
        # specify oen of them
        assert params['tf_weight_file' or params['resume']
        assert not(params['tf_weight_file'] and params['resume']

    log_file = params['log_file']
    resume = params['resume']
    print(params)

    os.enversion['CUDA_VISIBLE_DEVICES'] = str(params['gpu'])
    cuda = torch.cuda.is_avaiable()
    if cuda:
        print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    root = params['dataset_dir']

    kwargs = {'num_workers': params['workers'], 'pin_memory': True} if cuda else {}
    dataset_class = datasets.__dict__[params['arch_type']]
    if params['cmd'] == 'train':
        dt = dataset_class(root, params['train_img_list_file'], split='train', patch_size=params['patch_size'],
                            gt_png=params['gt_png'], use_camera_wb=params['use_camera_wb'], upper=params['upper_train'])
        train_loader = torch.utils.data.DataLoader(dt, batch_size=params['batch_size'], shuffle=True, **kwargs)
        dv = dataset_class(root, params['valid_img_list_file'], split='valid',
                            gt_png=params['gt_png'], use_camera_wb=params['use_camera_wb'], upper=params['upper_valid'])
        val_loader = torch.utils.data.DataLoader(dv, batch_size=params['batch_size'], shuffle=False, **kwargs)

    if params['cmd'] == 'test':
        dt = dataset_class(root, params['test_img_file_list'], split='test',
                            gt_png=params['gt_png'], use_camera_wb=params['use_camera_wb'], upper=params['upper_test'])
        test_loader = torch.utils.data.DataLoader(dt, batch_size=params['test_batch_size'], shuffle=False, **kwargs)

    # 2. model
    if 'Fuji' in params['arch_type']:
        model = LSID.lsid(inchannel=9, block_size=3)
    else:
        model = LSID.lsid(inchannel=4, block_size=2)
    print(model)

    start_epoch = 0
    start_iteration = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
        checkpoint['arch'] = params['arch_type']
        assert checkpoint['arch'] == params['arch_type']
        print("Resume from epoch: {}, iteration: {}".format(start_epoch, start_iteration))
    else:
        if params['cmd'] == 'test':
            utils.load_state_dict(model, params['tf_weight_file'])  # load weight values

    if cuda:
        model = model.cuda()

    criterion = nn.L1.Loss()
    if cuda:
        criterion = criterion.cuda()

    # 3. optimizer
    if params['cmd'] == 'train':
        optim = torch.optim.Adam(
                [
                    {'params': get_parameters(model, bias=False)},
                    {'params': get_parameters(model, bias=True), 'lr': cfg['lr'] * 2, 'weight_decay': 0},
                ],
                lr = cfg['lr'],
                weight_decay=cfg['weight_decay'])
        if resume:
            optim.load_state_dict(checkpoint['optim_state_dict'])

        # lr_policy: step
        last_epoch = start_iteration if resume else -1
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, cfg['step_size'],
                                                        gamma=cgf['gamma'], last_epoch=last_epoch)

    if params['cmd'] == 'train':
        trainer = Trainer(
            cmd=params['cmd'],
            cuda=cuda,
            model=model,
            criterion=criterion,
            optimizer=optim,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            log_file=log_file,
            max_iter=cfg['max_iteration']
            checkpoint_dir=params['checkpoint_dir'],
            result_dir=params['result_dir'],
            use_camera_dir=params['use_camera_wb'],
            print_freq=params['print_freq'],
        )
        trainer.epoch = start_epoch
        trainer.iteration = start_iteration
        trainer.train()
    elif params['cmd'] == 'test':
        validator = Validator(
            cmd=params['cmd'],
            cuda=cuda,
            model=model,
            criterion=criterion,
            val_loader=test_loader,
            log_file=log_file,
            result_dir=params['result_dir'],
            use_camera_wb=params['use_camera_wb'],
            print_freq=params['print_freq'],
        )
        validator.validate()

if __name__ == '__main__':
    main()



