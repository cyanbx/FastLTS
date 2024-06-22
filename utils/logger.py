from utils.util import to_arr
from hparams import hparams as hps
from tensorboardX import SummaryWriter
from utils.audio import inv_mel_spectrogram
from utils.plot import plot_spectrogram_to_numpy
from utils.mel_spectrogram import mel_spectrogram

class Lip2WavLogger(SummaryWriter):
    def __init__(self, logdir):
        super(Lip2WavLogger, self).__init__(logdir, flush_secs=5)
        self.hifi_hps = {
			'fft_size': 800,
			'audio_num_mel_bins': 80,
			'audio_sample_rate': 16000,
			'hop_size': 200,
			'win_size': 800,
			'fmin': 55,
			'fmax': 7600
		}

    def log_training(self, output, target, loss_output, grad_norm, lr, iter, opt_idx):
        if opt_idx == 0:
            self.add_scalar('mel_loss', loss_output['mel'], iter)
            # self.add_scalar('aux_mel_loss', loss_output['aux_mel'], iter)
            self.add_scalar('gen_s', loss_output['gen_s'], iter)
            self.add_scalar('gen_f', loss_output['gen_f'], iter)
            self.add_scalar('fm_f', loss_output['fm_f'], iter)
            self.add_scalar('fm_s', loss_output['fm_s'], iter)
            self.add_scalar('gen_loss', sum(loss_output.values()), iter)
            self.add_scalar('gen_grad_norm', grad_norm, iter)
            self.add_scalar('gen_lr', lr, iter)

            mel_out = inv_mel_spectrogram(output['mel'][0].contiguous().cpu().detach().numpy(), hps)
            wav_from_mel = inv_mel_spectrogram(target['mel'][0].contiguous().cpu().detach().numpy(), hps)
            self.add_audio('wav_pred', output['wav'][0], iter, sample_rate=hps.sample_rate)
            self.add_audio('wav_gt', target['wav'][0], iter, sample_rate=hps.sample_rate)
            self.add_audio('mel_output', mel_out, iter, sample_rate=hps.sample_rate)
            self.add_audio('mel_gt', wav_from_mel, iter, sample_rate=hps.sample_rate)

            wav_pred = output['wav'][0]
            mel_from_wav_pred = mel_spectrogram(wav_pred, self.hifi_hps).squeeze(0)
            fig = plot_spectrogram_to_numpy(mel_from_wav_pred.contiguous().cpu().detach().numpy(), figsize=(2, 3))
            self.add_image('training_spec', fig, iter)

            wav_gt = target['wav'][0]
            mel_from_wav_gt = mel_spectrogram(wav_gt.unsqueeze(0), self.hifi_hps).squeeze(0)
            fig_gt = plot_spectrogram_to_numpy(mel_from_wav_gt.contiguous().cpu().detach().numpy(), figsize=(2, 3))
            self.add_image('gt_spec', fig_gt, iter)

            fig_md = plot_spectrogram_to_numpy(output['mel'][0].contiguous().cpu().detach().numpy())
            self.add_image('training_mel_output', fig_md, iter)

            fig_li = plot_spectrogram_to_numpy(output['lin'][0].contiguous().cpu().detach().numpy(), figsize=(2, 3))
            self.add_image('training_lin_out', fig_li, iter)
            
        else:
            self.add_scalar('disc_s', loss_output['disc_s'], iter)
            self.add_scalar('disc_f', loss_output['disc_f'], iter)
            self.add_scalar('disc_loss', sum(loss_output.values()), iter)
            self.add_scalar('disc_grad_norm', grad_norm, iter)
            self.add_scalar('disc_lr', lr, iter)
    
    def sample_training(self, output, target, iter):
        wav_gt = target['wav'][0]
        mel_from_wav_gt = mel_spectrogram(wav_gt.unsqueeze(0), self.hifi_hps).squeeze(0)
        fig_gt = plot_spectrogram_to_numpy(mel_from_wav_gt.contiguous().cpu().detach().numpy())
        self.add_image('test.gt_spec', fig_gt, iter)

        wav_pred = output['wav'][0]
        mel_from_wav_pred = mel_spectrogram(wav_pred, self.hifi_hps).squeeze(0)
        fig = plot_spectrogram_to_numpy(mel_from_wav_pred.contiguous().cpu().detach().numpy())
        self.add_image('test.spec', fig, iter)

        wav_from_mel = inv_mel_spectrogram(target['mel'][0].contiguous().cpu().detach().numpy(), hps)
        self.add_audio('test.wav_pred', output['wav'][0], iter, sample_rate=hps.sample_rate)
        self.add_audio('test.wav_gt', target['wav'][0], iter, sample_rate=hps.sample_rate)
        self.add_audio('test.mel_gt', wav_from_mel, iter, sample_rate=hps.sample_rate)


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir, flush_secs=5)

    def log_training(self, reduced_loss, grad_norm, learning_rate, iteration):
        self.add_scalar("training.loss", reduced_loss, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)

    def log_training_vid(self, output, target, reduced_loss, grad_norm, learning_rate, iteration):
        l1_loss, ssim_loss = reduced_loss
        self.add_scalar("training.l1_loss", l1_loss, iteration)
        self.add_scalar("training.ssim_loss", ssim_loss, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)

        mel_outputs = to_arr(output[0])
        mel_target = to_arr(target[0][0])

        self.add_image(
            "mel_outputs",
            plot_spectrogram_to_numpy(mel_outputs),
            iteration)

        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_target),
            iteration)

        # save audio
        wav = inv_mel_spectrogram(mel_outputs, hps)
        wav_target = inv_mel_spectrogram(mel_target, hps)
        self.add_audio('pred', wav, iteration, hps.sample_rate)
        self.add_audio('target', wav_target, iteration, hps.sample_rate)

    def sample_training(self, output, target, iteration):
        mel_outputs = to_arr(output[0])
        mel_target = to_arr(target[0][0])

        self.add_image(
            "mel_outputs_test",
            plot_spectrogram_to_numpy(mel_outputs),
            iteration)

        self.add_image(
            "mel_target_test",
            plot_spectrogram_to_numpy(mel_target),
            iteration)

        # save audio
        wav = inv_mel_spectrogram(mel_outputs, hps)
        wav_target = inv_mel_spectrogram(mel_target, hps)
        self.add_audio('pred_test', wav, iteration, hps.sample_rate)
        self.add_audio('target_test', wav_target, iteration, hps.sample_rate)
