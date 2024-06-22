import os
import time
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
from hparams import hparams as hps
from torch.utils.data import DataLoader
from utils.logger import Lip2WavLogger
from utils.dataset import VideoMelLoader, VMcollate
from model.model import Lip2Wav,  slice_segments
from model.criterion import HifiGanLoss
from datetime import datetime
from model.hifigan import MultiPeriodDiscriminator, MultiScaleDiscriminator

# >>>> DDP >>>>
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DistSampler

# <<<< DDP <<<<

current_time = datetime.now().strftime('%b%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

logging.basicConfig(level=logging.DEBUG)


def prepare_dataloaders_vid(args, fdir):
	trainset = VideoMelLoader(fdir, hps, "train")
	collate_fn = VMcollate(hps, hps.n_frames_per_step)
	if args.ddp:
		train_sampler = DistSampler(trainset)
		train_loader = DataLoader(trainset, num_workers=hps.n_workers, sampler=train_sampler,
								batch_size=hps.batch_size, pin_memory=hps.pin_mem,
								drop_last=True, collate_fn=collate_fn)
	else:
		train_loader = DataLoader(trainset, num_workers=hps.n_workers, shuffle=True,
								batch_size=hps.batch_size, pin_memory=hps.pin_mem,
								drop_last=True, collate_fn=collate_fn)
	return train_loader


def prepare_dataloaders_vid_test(fdir):
	trainset = VideoMelLoader(fdir, hps, "test")
	collate_fn = VMcollate(hps, hps.n_frames_per_step)
	test_loader = DataLoader(trainset, num_workers=hps.n_workers, shuffle=False,
							 batch_size=hps.batch_size, pin_memory=hps.pin_mem,
							 drop_last=True, collate_fn=collate_fn)
	return test_loader


def load_checkpoint(ckpt_pth, model, optimizer):
	ckpt_dict = torch.load(ckpt_pth, map_location='cpu')
	model.load_state_dict(ckpt_dict['model'])
	optimizer.load_state_dict(ckpt_dict['optimizer'])
	iteration = ckpt_dict['iteration']
	return model, optimizer, iteration


def save_checkpoint(model, optimizer, iteration, ckpt_pth):
	torch.save({'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'iteration': iteration}, ckpt_pth)


def load_checkpoint_com(ckpt_pth, model, model_d, opt_gen, opt_disc):
	ckpt_dict = torch.load(ckpt_pth, map_location='cpu')
	model.load_state_dict({k.replace('module.', ''):v for k,v in ckpt_dict['model'].items()})
	model_d.load_state_dict({k.replace('module.', ''):v for k,v in ckpt_dict['model_d'].items()})
	opt_gen.load_state_dict(ckpt_dict['opt_gen'])
	opt_disc.load_state_dict(ckpt_dict['opt_disc'])
	iter = ckpt_dict['iter']
	return model, model_d, opt_gen, opt_disc, iter


def save_checkpoint_com(model, model_d, opt_gen, opt_disc, iter, ckpt_pth):
	torch.save({'model': model.state_dict(),
				'model_d': model_d.state_dict(),
				'opt_gen': opt_gen.state_dict(),
				'opt_disc': opt_disc.state_dict(),
				'iter': iter}, ckpt_pth)


def setup_ddp(rank, ngpu):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '65481'
    dist.init_process_group('nccl', rank=rank, world_size=ngpu)
    torch.cuda.set_device(rank)
    

def cleanup_ddp():
    dist.destroy_process_group()


def train(rank, args):
	if args.ddp:
		setup_ddp(rank, args.ngpu)

	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	model = Lip2Wav().to(device)
	model_d = nn.ModuleDict({
		'mpd': MultiPeriodDiscriminator(),
		'msd': MultiScaleDiscriminator()
	}).to(device)

	# freeze parameters
	model.frontend.requires_grad_(False)

	criterion = HifiGanLoss().to(device)

	optimizer_gen = torch.optim.AdamW(
		filter(lambda p: p.requires_grad, model.parameters()),
		lr=hps.lr_h,
		betas=[hps.adam_b1_h, hps.adam_b2_h],
		eps=hps.eps_h,
		weight_decay=hps.weight_decay_h
	)

	optimizer_disc = torch.optim.AdamW(
		model_d.parameters(),
		lr=hps.lr_h,
		betas=[hps.adam_b1_h, hps.adam_b2_h],
		eps=hps.eps_h,
		weight_decay=hps.weight_decay_h
	)

	optimizers = [optimizer_gen, optimizer_disc]

	epoch = 1
	iteration = 1

	if args.ckpt_pth != '':
		model, model_d, optimizer_gen, optimizer_disc, iteration = load_checkpoint_com(
			args.ckpt_pth, model, model_d, optimizer_gen, optimizer_disc)
		iteration += 1
	else:
		# load checkpoint of sub module
		assert os.path.exists(args.frontend), "ERROR: frontend checkpoint path invalid"
		fr_ckpt_dict = torch.load(args.frontend, map_location='cpu')
		model.frontend.load_state_dict(fr_ckpt_dict['model'])

	scheduler_gen = torch.optim.lr_scheduler.StepLR(
		optimizer=optimizer_gen,
		step_size=hps.decay_step,
		gamma=hps.lr_decay,
		last_epoch=iteration - 2
	)

	scheduler_disc = torch.optim.lr_scheduler.StepLR(
		optimizer=optimizer_disc,
		step_size=hps.decay_step,
		gamma=hps.lr_decay,
		last_epoch=iteration - 2
	)

	schedulers = [scheduler_gen, scheduler_disc]

	train_loader = prepare_dataloaders_vid(args, args.data_dir)
	test_loader = prepare_dataloaders_vid_test(args.data_dir)

	if args.log_dir != '' and not (args.ddp and rank != 0):
		if not os.path.isdir(args.log_dir + current_time):
			os.makedirs(args.log_dir + current_time)
			os.chmod(args.log_dir + current_time, 0o775)
		logger = Lip2WavLogger(args.log_dir + current_time)

	if args.ckpt_dir != '' and not os.path.isdir(args.ckpt_dir + current_time) and not (args.ddp and rank != 0):
		os.makedirs(args.ckpt_dir + current_time)
		os.chmod(args.ckpt_dir + current_time, 0o775)

	if args.ddp:
		model = DDP(model, device_ids=[rank], find_unused_parameters=True)
		model_d = DDP(model_d, device_ids=[rank], find_unused_parameters=True)
		dist.barrier()

	model.train()
	model_d.train()

	while True:
		if epoch > hps.max_iter:
			break
		logging.info('start epoch {}'.format(epoch))
		for batch in train_loader:
			if batch is None:
				continue
			if args.ddp:
				x, y = model.module.frontend.parse_batch_wav(batch)
			else:
				x, y = model.frontend.parse_batch_wav(batch)
			
			mel_pred, lin_out, wav_pred, ids = model(x)
			wav_gt = slice_segments(y[1].unsqueeze(1), ids * hps.hop_size, segment_size=hps.wav_segment_size).squeeze(1)

			output = {'mel': mel_pred, 'wav': wav_pred, 'lin': lin_out}
			target = {'mel': y[0], 'wav': wav_gt}

			for opt_idx in [0, 1]:
				start = time.perf_counter()
				loss_output, total_loss = criterion(
					output, target, model_d if not args.ddp else model_d.module, opt_idx)
				optimizers[opt_idx].zero_grad()
				total_loss.backward()
				grad_norm = torch.nn.utils.clip_grad_norm_(
					model.parameters(), hps.grad_clip_thresh)
				# skip if grad is NaN
				if (grad_norm != grad_norm):
					logging.info('Iter: {}, INF GRAD'.format(iteration))
					continue

				optimizers[opt_idx].step()
				schedulers[opt_idx].step()

				dur = time.perf_counter() - start
				
				if opt_idx == 0:
					logging.info('rank:{rank}, GEN: Iter:{iter}, loss:{loss}, mel_loss:{mel}, gen_s:{gen_s}, gen_f:{gen_f}, fm_s:{fm_s}, fm_f:{fm_f}, grad_norm:{grad_norm}, lr:{lr}, time:{dur}'.format(
								  rank=rank,
								  iter=iteration, loss=total_loss, mel=loss_output['mel'],
								  gen_s=loss_output['gen_s'], gen_f=loss_output['gen_f'],
								  fm_s=loss_output['fm_s'], fm_f=loss_output['fm_f'],
								  # aux=loss_output['aux_mel'],
								  grad_norm=grad_norm, lr=optimizers[opt_idx].param_groups[0]['lr'], dur=dur))
				else:
					logging.info('rank:{rank}, DISC: Iter:{iter}, loss:{loss}, disc_f:{disc_f}, disc_s:{disc_s}, grad_norm:{grad_norm}, lr:{lr}, time:{dur}'.format(
								  rank=rank,
								  iter=iteration, loss=total_loss,
								  disc_s=loss_output['disc_s'], disc_f=loss_output['disc_f'],
								  grad_norm=grad_norm, lr=optimizers[opt_idx].param_groups[0]['lr'], dur=dur))

				if (iteration % hps.iters_per_ckpt == 0) and opt_idx == 1 and args.ckpt_dir != '' and not (args.ddp and rank != 0):
					ckpt_pth = os.path.join(args.ckpt_dir+ current_time, 'ckpt_{}'.format(iteration))
					save_checkpoint_com(model, model_d, optimizer_gen, optimizer_disc, iteration, ckpt_pth)

				if (iteration % hps.iters_per_log == 0) and args.log_dir != '' and not (args.ddp and rank != 0):
					lr = optimizers[opt_idx].param_groups[0]['lr']
					logger.log_training(output=output, target=target, loss_output=loss_output, grad_norm=grad_norm,
										lr=lr, iter=iteration, opt_idx=opt_idx)
				
				if (iteration % hps.iters_per_sample == 0) and opt_idx == 1 and args.log_dir != '' and not (args.ddp and rank != 0):
					model.eval()
					with torch.no_grad():
						for i, batch in enumerate(test_loader):
							if i == 0:
								if args.ddp:
									x_test, y_test = model.module.frontend.parse_batch_wav(batch)
									wav_out = model.module.inference(x_test, 'train')
								else:
									x_test, y_test = model.frontend.parse_batch_wav(batch)
									wav_out = model.inference(x_test, 'train')
								output_ = { 'wav':wav_out }
								target_ = { 'mel': y_test[0], 'wav': y_test[1] }
								logger.sample_training(output_, target_, iteration)
							else:
								break
					model.train()
				
				if args.ddp:
					dist.barrier()

			iteration += 1

		epoch += 1

	if args.log_dir != '':      
		logger.close()
  
	if args.ddp:
		cleanup_ddp()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# path
	parser.add_argument('-d', '--data_dir', type=str, default='data',
						help='directory to load data')
	parser.add_argument('-l', '--log_dir', type=str, default='log/',
						help='directory to save tensorboard logs')
	parser.add_argument('-cd', '--ckpt_dir', type=str, default='ckpt/',
						help='directory to save checkpoints')
	parser.add_argument('-cp', '--ckpt_pth', type=str, default='',
						help='path to load checkpoints')
	parser.add_argument('-fr', '--frontend', type=str, default='',
						help='path to load frontend checkpoints')
	parser.add_argument('--ddp', action='store_true',
						help='use distributed data parallel')
	parser.add_argument('--ngpu', type=int, default=1,
						help='use distributed data parallel')

	args = parser.parse_args()

	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.benchmark = False  # faster due to dynamic input shape


	if args.ddp:
		mp.spawn(
			train,
			args=(args,),
			nprocs=args.ngpu,
			join=True
		)
	else:
		train(0, args)
