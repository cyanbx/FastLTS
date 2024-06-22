from torch import nn
from torch.nn import functional as F

from utils.ssim import ssim
from utils.mel_spectrogram import mel_spectrogram
from model.hifigan import feature_loss, generator_loss, discriminator_loss


class NarLoss(nn.Module):
	def __init__(self):
		super(NarLoss, self).__init__()
	
	def l1_loss(self, decoder_output, target):
		# decoder_output : B x T x n_mel
		# target : B x T x n_mel
		l1_loss = F.l1_loss(decoder_output, target, reduction='none')
		weights = self.weights_nonzero_speech(target)
		l1_loss = (l1_loss * weights).sum() / weights.sum()
		return l1_loss
	
	def mse_loss(self, decoder_output, target):
		assert decoder_output.shape == target.shape
		mse_loss = F.mse_loss(decoder_output, target, reduction='none')
		weights = self.weights_nonzero_speech(target)
		mse_loss = (mse_loss * weights).sum() / weights.sum()
		return mse_loss
	
	def ssim_loss(self, decoder_output, target, bias=6.0):
		# decoder_output : B x T x n_mel
		# target : B x T x n_mel
		assert decoder_output.shape == target.shape
		weights = self.weights_nonzero_speech(target)
		decoder_output = decoder_output[:, None] + bias
		target = target[:, None] + bias
		ssim_loss = 1 - ssim(decoder_output, target, size_average=False)
		ssim_loss = (ssim_loss * weights).sum() / weights.sum()
		return ssim_loss
	
	def weights_nonzero_speech(self, target):
		# target : B x T x mel
		# Assign weight 1.0 to all labels except for padding (id=0).
		dim = target.size(-1)
		return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)

	def forward(self, decoder_output, targets):
		mel_target = targets[0]
		mel_target.requires_grad = False

		mel_l1_loss = self.l1_loss(decoder_output, mel_target) * 10
		mel_ssim_loss = self.ssim_loss(decoder_output, mel_target) * 10

		return mel_l1_loss, mel_ssim_loss
	

class HifiGanLoss(nn.Module):
	def __init__(self):
		super(HifiGanLoss, self).__init__()
		self.hifi_hps = {
			'fft_size': 800,
			'audio_num_mel_bins': 80,
			'audio_sample_rate': 16000,
			'hop_size': 200,
			'win_size': 800,
			'fmin': 55,
			'fmax': 7600
		}
		self.lambda_mel = 45.
		self.aux_criterion = NarLoss()
		self.wav_pred = None

	def forward(self, output, target, model_disc, optim_idx):
		wav_gt = target['wav']
		
		if optim_idx == 0:
			wav_pred = output['wav']

			wav_gt_mel = mel_spectrogram(wav_gt, self.hifi_hps)
			wav_pred_mel = mel_spectrogram(wav_pred.squeeze(1), self.hifi_hps)
			loss_mel = F.l1_loss(wav_pred_mel, wav_gt_mel) * self.lambda_mel
			
			y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = model_disc['mpd'](wav_gt.unsqueeze(1), wav_pred)
			y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = model_disc['msd'](wav_gt.unsqueeze(1), wav_pred)

			loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
			loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
			loss_gen_f, _ = generator_loss(y_df_hat_g)
			loss_gen_s, _ = generator_loss(y_ds_hat_g)
			
			loss_output = {
				'gen_s': loss_gen_s, 'gen_f': loss_gen_f,
				'fm_s': loss_fm_s, 'fm_f': loss_fm_f, 'mel': loss_mel,
			}
			total_loss = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel # + aux_loss
			
			self.wav_pred = wav_pred.detach()
		else:
			wav_pred_ = self.wav_pred
			y_df_hat_r, y_df_hat_g, _, _ = model_disc['mpd'](wav_gt.unsqueeze(1), wav_pred_.detach())
			y_ds_hat_r, y_ds_hat_g, _, _ = model_disc['msd'](wav_gt.unsqueeze(1), wav_pred_.detach())
			
			loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
			loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
			
			loss_output = {'disc_s': loss_disc_f, 'disc_f': loss_disc_s}
			
			total_loss = loss_disc_s + loss_disc_f
		
		return loss_output, total_loss