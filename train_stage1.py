import os
import time
import torch
import argparse
import numpy as np
from utils.util import mode
from hparams import hparams as hps
from torch.utils.data import DataLoader
from utils.logger import Tacotron2Logger
from utils.dataset import VideoMelLoader, VMcollate
from model.model import Lip2Mel
from model.criterion import NarLoss
from datetime import datetime

current_time = datetime.now().strftime('%b%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def prepare_dataloaders_vid(fdir):
	trainset = VideoMelLoader(fdir, hps, "train")
	collate_fn = VMcollate(hps, hps.n_frames_per_step)
	train_loader = DataLoader(trainset, num_workers = hps.n_workers, shuffle = True,
							  batch_size = hps.batch_size, pin_memory = hps.pin_mem,
							  drop_last = True, collate_fn = collate_fn)
	return train_loader

def prepare_dataloaders_vid_test(fdir):
	trainset = VideoMelLoader(fdir, hps, "test")
	collate_fn = VMcollate(hps, hps.n_frames_per_step)
	test_loader = DataLoader(trainset, num_workers = hps.n_workers, shuffle = False,
							  batch_size = hps.batch_size, pin_memory = hps.pin_mem,
							  drop_last = True, collate_fn = collate_fn)
	return test_loader

def load_checkpoint(ckpt_pth, model, optimizer):
	ckpt_dict = torch.load(ckpt_pth)
	model.load_state_dict(ckpt_dict['model'])
	optimizer.load_state_dict(ckpt_dict['optimizer'])
	iteration = ckpt_dict['iteration']
	return model, optimizer, iteration


def save_checkpoint(model, optimizer, iteration, ckpt_pth):
	torch.save({'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'iteration': iteration}, ckpt_pth)


def train(args):
	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	# build model
	model = Lip2Mel()

	mode(model, True)
	model = model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr = hps.lr,
								betas = hps.betas, eps = hps.eps,
								weight_decay = hps.weight_decay)
	criterion = NarLoss().to(device)
	
	# load checkpoint
	iteration = 1
	if args.ckpt_pth != '':
		model, optimizer, iteration = load_checkpoint(args.ckpt_pth, model, optimizer)
		iteration += 1 # next iteration is iteration+1
	
	# get scheduler
	if hps.sch:
		lr_lambda = lambda step: hps.sch_step**0.5*min((step+1)*hps.sch_step**-1.5, (step+1)**-0.5)
		if args.ckpt_pth != '':
			scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch = iteration)
		else:
			scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
	
	# make dataset
	# train_loader = prepare_dataloaders(args.data_dir)
	train_loader = prepare_dataloaders_vid(args.data_dir)

	# get logger ready
	if args.log_dir != '':
		if not os.path.isdir(args.log_dir+ current_time):
			os.makedirs(args.log_dir+ current_time)
			os.chmod(args.log_dir+ current_time, 0o775)
		logger = Tacotron2Logger(args.log_dir+ current_time)

	# get ckpt_dir ready
	if args.ckpt_dir != '' and not os.path.isdir(args.ckpt_dir+ current_time):
		os.makedirs(args.ckpt_dir+ current_time)
		os.chmod(args.ckpt_dir+ current_time, 0o775)
	
	model.train()
	
	# add debug
	# torch.autograd.set_detect_anomaly(True)
	# ================ MAIN TRAINNIG LOOP! ===================
	while iteration <= hps.max_iter:
		for batch in train_loader:
			if iteration > hps.max_iter:
				break
			start = time.perf_counter()
			if batch is None:
				continue
			x, y = model.parse_batch_wav(batch)
			
			_, y_pred = model(x)

			# loss
			l1_loss, ssim_loss = criterion(y_pred, y)

			loss = l1_loss + ssim_loss
			items = [l1_loss.item(), ssim_loss.item()]
			# zero grad
			model.zero_grad()
			
			# backward, grad_norm, and update

			# with torch.autograd.detect_anomaly():
			loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hps.grad_clip_thresh)
			# skip if grad is NaN
			if (grad_norm != grad_norm):
				print('Iter: {}, INF GRAD'.format(iteration))
				continue

			optimizer.step()
			if hps.sch:
				scheduler.step()
			
			# info
			dur = time.perf_counter()-start
			print('Iter: {} Loss: {:.5f} Grad Norm: {:.5f} {:.5f}s/it'.format(
				iteration, sum(items), grad_norm, dur))

			# log vid
			if args.log_dir != '' and (iteration % hps.iters_per_log == 0):
				learning_rate = optimizer.param_groups[0]['lr']
				logger.log_training_vid(y_pred, y, items, grad_norm, learning_rate, iteration)

			# save ckpt
			if args.ckpt_dir != '' and (iteration % hps.iters_per_ckpt == 0):
				ckpt_pth = os.path.join(args.ckpt_dir+ current_time, 'ckpt_{}'.format(iteration))
				save_checkpoint(model, optimizer, iteration, ckpt_pth)

			iteration += 1
	if args.log_dir != '':
		logger.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# path	
	parser.add_argument('-d', '--data_dir', type = str, default = 'data',
						help = 'directory to load data')
	parser.add_argument('-l', '--log_dir', type = str, default = 'log/',
						help = 'directory to save tensorboard logs')
	parser.add_argument('-cd', '--ckpt_dir', type = str, default = 'ckpt/',
						help = 'directory to save checkpoints')
	parser.add_argument('-cp', '--ckpt_pth', type = str, default = '',
						help = 'path to load checkpoints')

	args = parser.parse_args()


	train(args)
