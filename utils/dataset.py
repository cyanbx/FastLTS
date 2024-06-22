import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import cv2
from glob import glob
from pathlib import Path
from os.path import dirname, join, basename, isfile

import librosa

random.seed(0)

def files_to_list(fdir):
	f_list = []
	with open(os.path.join(fdir, 'metadata.csv'), encoding = 'utf-8') as f:
		for line in f:
			parts = line.strip().split('|')
			wav_path = os.path.join(fdir, 'wavs', '%s.wav' % parts[0])
			f_list.append([wav_path, parts[1]])
	return f_list

def find_max_frame(video_images):
	max_f = -1
	for img in video_images:
		if int(os.path.basename(img).split(".")[0]) > max_f:
			max_f = int(os.path.basename(img).split(".")[0])
	return max_f

class VideoMelLoader(torch.utils.data.Dataset):
	def __init__(self, fdir, hparams, split='tcdtimit_train'):
		self._hparams = hparams
		self.filelist = self.get_image_list(split, fdir)

	def get_image_list(self, split, data_root):
		filelist = []
		video_images = []

		with open(os.path.join(data_root, '{}.txt'.format(split))) as vidlist:
			for vid_id in vidlist:
				vid_id = vid_id.strip()
				video_images.extend(list(glob(os.path.join(data_root, 'preprocessed', vid_id, '*/*.jpg'))))
		
		max_f = find_max_frame(video_images)
		for img in video_images:
			num = int(os.path.basename(img).split(".")[0])
			if num > self._hparams.T // 2 and num < max_f - self._hparams.T // 2:
				filelist.extend([img])
		
		return filelist

	def get_window(self, center_frame):
		center_id = self.get_frame_id(center_frame)
		vidname = dirname(center_frame)
		if self._hparams.T % 2:
			window_ids = range(center_id - self._hparams.T // 2, center_id + self._hparams.T // 2 + 1)
		else:
			window_ids = range(center_id - self._hparams.T // 2, center_id + self._hparams.T // 2)

		window_fnames = list()
		for frame_id in window_ids:
			frame = join(vidname, '{}.jpg'.format(frame_id))

			if not isfile(frame):
				return None
			window_fnames.append(frame)
		return window_fnames

	def crop_audio_window(self, spec, wav, center_frame):
		# estimate total number of frames from spec (num_features, T)
		# num_frames = (T x hop_size * fps) / sample_rate
		start_frame_id = self.get_frame_id(center_frame) - self._hparams.T // 2
		total_num_frames = int((spec.shape[0] * self._hparams.hop_size * self._hparams.fps) / self._hparams.sample_rate)

		start_idx = int(spec.shape[0] * start_frame_id / float(total_num_frames))
		end_idx = start_idx + self._hparams.mel_step_size
		
		start_wav_index = start_idx * self._hparams.hop_size
		end_wav_index = start_wav_index + self._hparams.wav_step_size

		return spec[start_idx: end_idx, :], wav[start_wav_index: end_wav_index]

	def get_frame_id(self, frame):
		return int(basename(frame).split('.')[0])

	def __getitem__(self, index):
		img_name = self.filelist[index]
		window_fnames = self.get_window(img_name)

		mel = np.load(os.path.join(os.path.dirname(img_name), 'mels.npz'))['spec'].T
		wav = librosa.core.load(os.path.join(os.path.dirname(img_name), 'audio.wav'), sr=self._hparams.sample_rate)[0]
		
		mel, wav = self.crop_audio_window(mel, wav, img_name)

		window = []
		if type(window_fnames) == list:
			for fname in window_fnames:
				img = cv2.imread(fname)
				path = Path(fname)
				img = cv2.resize(img, (self._hparams.img_size, self._hparams.img_size))

				window.append(img)
			x = np.asarray(window) / 255.
		else:
			x = None
		embed_target = np.zeros([256], dtype=np.float32)

		item_name = "_".join(img_name.split('/')[-3:])
		
		return x, mel.T, wav, embed_target, len(mel), item_name

	def __len__(self):
		return len(self.filelist)

class VMcollate():
	def __init__(self, hparams, n_frames_per_step):
		self.n_frames_per_step = n_frames_per_step
		self._hparams = hparams

	def __call__(self, batch):
		# Right zero-pad all one-hot text sequences to max input length
		vid_refined, mel_refined, wav_refined, target_lengths, embed_refined, item_names = [], [], [], [], [], []

		for i, (vid, mel, wav, emb_tar, aud_len, item_name) in enumerate(batch):
			if not vid is None:
				vid = torch.Tensor(vid)
				mel = torch.Tensor(mel)
				wav = torch.Tensor(wav)
    
				vid_refined.append(vid)
				mel_refined.append(mel)
				wav_refined.append(wav)
   
				embed_refined.append(emb_tar)
				target_lengths.append(aud_len)

				item_names.append(item_name)

		input_lengths, ids_sorted_decreasing = torch.sort(
			torch.LongTensor(np.array([len(v) for v in vid_refined])),
			dim=0, descending=True)

		bsz = len(input_lengths)
		if (bsz == 0):
			return None
		
		max_input_len = input_lengths[0]
		# vid_padded = torch.LongTensor(len(batch), max_input_len) #todo
		vid_padded = torch.Tensor(bsz, 3, max_input_len, self._hparams.img_size, self._hparams.img_size) #todo
		vid_padded.zero_()

		item_sorted = []

		for i in range(len(ids_sorted_decreasing)):
			vid_sorted = vid_refined[ids_sorted_decreasing[i]]
			vid_sorted = vid_sorted.permute(3, 0, 1, 2)
			vid_padded[i, :, :vid_sorted.size(1), :, :] = vid_sorted

		# Right zero-pad mel-spec
		# num_mels = vid_refined[1].size(0) #todo check
		num_mels = self._hparams.num_mels
		max_target_len = max([m.size(1) for m in mel_refined])

		if max_target_len % self.n_frames_per_step != 0:
			max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
			assert max_target_len % self.n_frames_per_step == 0

		mel_padded = torch.FloatTensor(bsz, num_mels, max_target_len)
		mel_padded.zero_()
		target_lengths = torch.LongTensor(bsz)
  
		max_wav_len = max([w.size(0) for w in wav_refined])
		wav_padded = torch.FloatTensor(bsz, max_wav_len)
		wav_padded.zero_()
	
		for i in range(len(ids_sorted_decreasing)):
			mel = mel_refined[ids_sorted_decreasing[i]]
			mel_padded[i, :, :mel.size(1)] = mel
			target_lengths[i] = mel.size(1)
			
			wav = wav_refined[ids_sorted_decreasing[i]]
			wav_padded[i, :wav.size(0)] = wav

			item_sorted.append(item_names[ids_sorted_decreasing[i]])

		embed_targets = torch.LongTensor(np.array(embed_refined))
		split_infos = torch.LongTensor(np.array([max_input_len, max_target_len]))

		return vid_padded, input_lengths, mel_padded, wav_padded, target_lengths, split_infos, embed_targets, item_names

