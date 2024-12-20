import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/wd_sr3_16_64.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="0")
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', '-w', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', '-l',action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    val_list = []
    psnr1_queue = torch.zeros(5)
    psnr2_queue = torch.zeros(5)
    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val1':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_list.append(Data.create_dataloader(val_set, dataset_opt, phase))
        elif phase == 'val2':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_list.append(Data.create_dataloader(val_set, dataset_opt, phase))
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    early_stop = True

    times = []

    if opt['phase'] == 'train':
        while current_step < n_iter or early_stop:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                start_time = time.time()
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()

                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)
                end_time = time.time()

                times.append(end_time - start_time)
                # print("time average{}, {}".format(current_step, np.mean(times)))

                # validation
                if current_step % opt['train']['val_freq'] == 0:

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')

                    # train out
                    train_out_path = '{}/{}/train'.format(opt['path']['results'], current_step)
                    train_psnr = 0.0
                    os.makedirs(train_out_path, exist_ok=True)
                    diffusion.test(continous=False)
                    visuals = diffusion.get_current_visuals()
                    for i in range(visuals['SR'].shape[0]):
                        fake_img = Metrics.tensor2mhd(visuals['INF'][i])
                        sr_img = Metrics.tensor2mhd(visuals['SR'][i])
                        hr_img = Metrics.tensor2mhd(visuals['HR'][i])
                        train_img = np.concatenate([fake_img, sr_img, hr_img], axis=1)
                        Metrics.save_mhd(train_img, '{}/{}_train_{}.mhd'.format(train_out_path, current_step, i))
                        train_psnr += Metrics.calculate_psnr(sr_img, hr_img)

                    train_psnr = train_psnr / visuals['SR'].shape[0]
                    logger.info('# Train # PSNR: {:.4e}'.format(train_psnr))

                    val_i = 0
                    val_psnr = []
                    for val_loader in val_list:
                        val_i += 1
                        avg_psnr = 0.0
                        idx = 0
                        result_path = '{}/{}/val{}'.format(opt['path']['results'], current_epoch, val_i)
                        os.makedirs(result_path, exist_ok=True)
                        for _,  val_data in enumerate(val_loader):
                            idx += 1
                            diffusion.feed_data(val_data)
                            diffusion.test(continous=False)
                            visuals = diffusion.get_current_visuals()
                            for i in range(visuals['SR'].shape[0]):
                                fake_img = Metrics.tensor2mhd(visuals['INF'][i])
                                sr_img = Metrics.tensor2mhd(visuals['SR'][i])
                                hr_img = Metrics.tensor2mhd(visuals['HR'][i])
                                val_img = np.concatenate([fake_img, sr_img, hr_img], axis=1)
                                Metrics.save_mhd(val_img, '{}/{}_{}_val_{}.mhd'.format(result_path, current_step, idx, i))
                                avg_psnr += Metrics.calculate_psnr(sr_img, hr_img)

                                if wandb_logger:
                                    wandb_logger.log_image(
                                        f'val{val_i}_{idx}',
                                        np.concatenate((fake_img, sr_img, hr_img), axis=1)
                                    )

                        avg_psnr = avg_psnr / visuals['SR'].shape[0]
                        val_psnr.append(avg_psnr)

                        # log
                        logger.info('# Validation{} # PSNR: {:.4e}'.format(val_i ,avg_psnr))
                        logger_val = logging.getLogger('val')  # validation logger
                        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                            current_epoch, current_step, avg_psnr))
                        # tensorboard logger
                        tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    psnr1_queue[1:] = psnr1_queue[:-1].clone()
                    psnr1_queue[0] = val_psnr[0]
                    psnr2_queue[1:] = psnr2_queue[:-1].clone()
                    psnr2_queue[0] = val_psnr[1]
                    move_avg_psnr1 = psnr1_queue.mean().item()
                    move_avg_psnr2 = psnr2_queue.mean().item()
                    move_avg_psnr = (move_avg_psnr1 + move_avg_psnr2) / 2

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'val/train_psnr': train_psnr,
                            'val/val1_psnr': val_psnr[0],
                            'val/val2_psnr': val_psnr[1],
                            'val_step': val_step,
                        })
                        val_step += 1

                    logger.info("val1_stop is {}, val2_stop is {}".format(abs(move_avg_psnr1 - val_psnr[0]) < 1, abs(move_avg_psnr2 - val_psnr[1]) < 1))
                    logger.info("train_stop is {}".format(abs(train_psnr - move_avg_psnr) < 3.0))
                    if current_step > 5000 and abs(move_avg_psnr1 - val_psnr[0]) < 1 and abs(move_avg_psnr2 - val_psnr[1]) < 1 and abs(train_psnr - move_avg_psnr) < 3.0:
                        logger.info('Early stopping.')
                        early_stop = False
                        break

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        logger.info('End of training.')
