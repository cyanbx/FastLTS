import math
import torch
from math import sqrt
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from hparams import hparams as hps

from model.common_layers import Linear
from model.tts_modules import FastspeechDecoder
from model.transformers import PositionalEncoding, make_transformer_encoder
from model.hifigan import HifiGanGenerator
from utils.util import to_var, get_mask_from_lengths


class Vid2Tokens(nn.Module):
	def __init__(self,
				 in_chans=3,
				 out_chans=32,
				 kernel_size=(5, 5, 5),
				 stride=(1, 3, 3),
				 padding=(2, 2, 2),
				 norm_type='ln'
				):
		super().__init__()
		self.norm_type = norm_type
		self.conv = nn.Conv3d(in_chans,
			out_chans,
			kernel_size=kernel_size,
			stride=stride,
			padding=padding
		)
		self.norm = nn.BatchNorm2d(out_chans) if norm_type == 'bn' else nn.LayerNorm((out_chans, 32, 32))
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

	def forward(self, x):
		x = self.conv(x)
		x = rearrange(x, 'b c t h w -> (b t) c h w')
		x = self.norm(x)
		x = self.maxpool(x)
		return x


class LeFF(nn.Module):
	def __init__(self,
				 in_features,
				 hidden_features=None,
				 out_features=None,
				 act_layer=nn.GELU,
				 drop=0.,
				 kernel_size=3,
				 with_bn=True
				):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features

		# a.k.a. linear augmentation
		self.conv1 = nn.Conv2d(
			in_features,
			hidden_features,
			kernel_size=1,
			stride=1,
			padding=0
		)
		# depthwise
		self.conv2 = nn.Conv2d(
			hidden_features,
			hidden_features,
			kernel_size=kernel_size,
			padding=(kernel_size - 1)//2,
			stride=1,
			# every channel has its own filter
			groups=hidden_features
		)
		# back to out features
		self.conv3 = nn.Conv2d(
			hidden_features,
			out_features,
			kernel_size=1,
			stride=1,
			padding=0
		)
		self.act = act_layer()

		self.dropout = nn.Dropout(drop)

		self.with_bn = with_bn

		if self.with_bn:
			self.bn1 = nn.BatchNorm2d(hidden_features)
			self.bn2 = nn.BatchNorm2d(hidden_features)
			self.bn3 = nn.BatchNorm2d(out_features)

	def forward(self, x):
		b, n, k = x.size()
		x = x.reshape(b, int(math.sqrt(n)), int(math.sqrt(n)), k).permute(0, 3, 1, 2)
		if self.with_bn:
			x = self.conv1(x)
			x = self.bn1(x)
			x = self.act(x)
			x = self.conv2(x)
			x = self.bn2(x)
			x = self.act(x)
			x = self.conv3(x)
			x = self.bn3(x)
			x = self.dropout(x)
		else:
			x = self.conv1(x)
			x = self.act(x)
			x = self.conv2(x)
			x = self.act(x)
			x = self.conv3(x)
			x = self.dropout(x)

		tokens = x.flatten(2).permute(0, 2, 1)
		return tokens

class LeFFPerformerLayer(nn.Module):
	def __init__(self,
				 input_dim,
				 head_dim,
				 head_cnt,
				 kernel_ratio,
				 kernel_size=3,
				 dp1=0.1,
				 dp2=0.2,
				 act_layer=nn.GELU,
				 norm_layer=nn.LayerNorm,
				 with_bn=False
				):
		
		super().__init__()
		self.emb = head_cnt * head_dim
		self.kqv = nn.Linear(input_dim, 3 * self.emb)
		self.dp = nn.Dropout(dp1)
		self.proj = nn.Linear(self.emb, self.emb)
		self.head_cnt = head_cnt
		self.norm1 = norm_layer(input_dim)
		self.norm2 = norm_layer(self.emb)
		self.epsilon = 1e-8

		self.leff = LeFF(
			self.emb,
			hidden_features=int(4.0 * self.emb),
			act_layer=act_layer,
			drop=dp2,
			kernel_size=kernel_size,
			with_bn=with_bn
		)

		self.m = int(self.emb * kernel_ratio)
		self.w = torch.randn(self.m, self.emb)
		self.w = nn.Parameter(nn.init.orthogonal_(self.w) * sqrt(self.m), requires_grad=False)

	def prm_exp(self, x):
		xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
		wtx = torch.einsum('bti,mi->btm', x.float(), self.w)

		return torch.exp(wtx - xd) / sqrt(self.m)

	def single_attn(self, x):
		k, q, v = torch.split(self.kqv(x), self.emb, dim=-1)
		kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
		D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
		kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
		y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
		# skip connection
		y = v + self.dp(self.proj(y))  # same as token_transformer in T2T layer, use v as skip connection
		return y

	def forward(self, x):
		x = self.single_attn(self.norm1(x))
		x = x + self.leff(self.norm2(x))
		return x
	
		
class Ceit_Transformer(nn.Module):
	def __init__(self,
				 input_dim,
				 dim,
				 head_cnt,
				 leff_kernel_size=3,
				 kernel_ratio=0.5,
				 depth=4
				):
		super().__init__()
		self.blocks = nn.ModuleList([
			LeFFPerformerLayer(
				input_dim = input_dim if i == 0 else dim,
				head_dim = dim // head_cnt,
				head_cnt=head_cnt,
				kernel_ratio=kernel_ratio,
				kernel_size=leff_kernel_size
			)
			for i in range(depth)
		])

	def forward(self, x):
		for block in self.blocks:
			x = block(x)
		return x

class CeitEncoder(nn.Module):
	def __init__(self,
				 token_dim,
				 spatial_encoder_dim,
				 head_cnt=6,
				 leff_kernel_size=3,
				 depth=6
				):	
		
		super().__init__()
		self.to_patch = Vid2Tokens(
			in_chans=3,
			out_chans=token_dim
		)

		self.spatial_transformer = Ceit_Transformer(
			input_dim=token_dim,
			dim=spatial_encoder_dim,
			head_cnt=head_cnt,
			leff_kernel_size=leff_kernel_size,
			depth=depth
		)

		self.space_pos_embedding = nn.Parameter(torch.randn(1, hps.T, 16*16, token_dim))
		self.space_fusion = nn.Linear(16*16*spatial_encoder_dim, hps.encoder_embedding_dim)
		self.temporal_transformer = make_transformer_encoder(N_layer=hps.encoder_depth, d_model=hps.encoder_embedding_dim)
		self.temporal_pos_embedding = PositionalEncoding(d_model=hps.encoder_embedding_dim, dropout=0.1)

	def forward(self, x):
		b, c, t, h, w = x.shape
		x = self.to_patch(x) # [(B T) C H W]
		x = rearrange(x, '(b t) c h w -> b t (h w) c', t=t)
		x += self.space_pos_embedding[:, :t, :, :]
		x = rearrange(x, 'b t n c -> (b t) n c')
		x = self.spatial_transformer(x)
		x = rearrange(x, '(b t) n c -> b t (n c)',t=t)
		x = self.space_fusion(x)
		x = self.temporal_transformer(self.temporal_pos_embedding(x), None)
		
		return x

	def inference(self, x):
		with torch.no_grad():
			b, c, t, h, w = x.shape
			x = self.to_patch(x) # [(B T) C H W]
			x = rearrange(x, '(b t) c h w -> b t (h w) c', t=t)
			x += self.space_pos_embedding[:, :t, :, :]
			x = rearrange(x, 'b t n c -> (b t) n c')
			x = self.spatial_transformer(x)
			x = rearrange(x, '(b t) n c -> b t (n c)',t=t)
			x = self.space_fusion(x)
			x = self.temporal_transformer(self.temporal_pos_embedding(x), None)

		return x


class DecoderNAR(nn.Module):
	def __init__(self):
		super().__init__()
		self.decoder = FastspeechDecoder(hidden_size=hps.encoder_embedding_dim,
										 num_layers=hps.fs_depth,
										 kernel_size=hps.fs_kernel_size,
										 num_heads=hps.fs_head)

		self.mel_out = Linear(hps.encoder_embedding_dim, hps.num_mels, bias=True)

	def dup_hidden(self, encoder_out, rate=1.0, mel_len=240):
		rate_int = int(rate)
		rate_tail = rate - rate_int

		ds = torch.full([encoder_out.shape[1]], rate_int, dtype=int)
		rate_tail_mtx = torch.full(ds.shape, rate_tail)
		tail = torch.bernoulli(rate_tail_mtx)
		ds = ds + tail

		diff = torch.sum(ds) - mel_len
		for i in range(ds.shape[0]):
			if (diff == 0):
				break
			elif (diff > 0 and ds[i] == rate_int + 1):
				ds[i] = ds[i] - 1
				diff = diff - 1
			elif (diff < 0 and ds[i] == rate_int):
				ds[i] = ds[i] + 1
				diff = diff + 1

		regulated = torch.cat([self.dup_single(x, ds)
							  for x in encoder_out], dim=0)

		return regulated

	def dup_single(self, x, ds):
		copied = torch.cat([x_.repeat(int(d_), 1)
						   for x_, d_ in zip(x, ds)], dim=0)
		return copied.unsqueeze(0)

	def forward(self, encoder_out, mels):
		rate = mels.shape[2] / encoder_out.shape[1]
		decoder_inp = self.dup_hidden(encoder_out, rate, mels.shape[2])
		decoder_out, mel_out = self.run_decoder(decoder_inp, None)
		return decoder_out, mel_out

	def inference(self, encoder_out, fps=30, hop=80):
		rate = hop / fps
		mel_length = encoder_out.shape[1] * rate
		decoder_inp = self.dup_hidden(encoder_out, rate, mel_length)
		decoder_out, mel_out = self.run_decoder(decoder_inp, None)
		return decoder_out, mel_out

	def run_decoder(self, decoder_inp, tgt_nonpadding=None):
		decoder_out = self.decoder(decoder_inp)
		mel_out = self.mel_out(decoder_out)
		return decoder_out, mel_out


def is_end_of_frames(output, eps=0.2):
	return (output.data <= eps).all()


class Lip2Mel(nn.Module):
	def __init__(self):
		super(Lip2Mel, self).__init__()
		self.num_mels = hps.num_mels
		self.mask_padding = hps.mask_padding
		self.n_frames_per_step = hps.n_frames_per_step
		
		self.encoder = CeitEncoder(token_dim=hps.token_dim,
			spatial_encoder_dim=hps.spatial_encoder_dim, head_cnt=hps.encoder_head, 
			leff_kernel_size=hps.leff_kernel_size, depth=hps.encoder_depth
		)
		self.decoder_nar = DecoderNAR()
	
	def parse_batch_wav(self, batch):
		vid_padded, input_lengths, mel_padded, wav_padded, target_lengths, split_infos, embed_targets, item_names = batch
		vid_padded = to_var(vid_padded).float()
		input_lengths = to_var(input_lengths).float()
		mel_padded = to_var(mel_padded).float()
		wav_padded = to_var(wav_padded).float()
		target_lengths = to_var(target_lengths).float()

		max_len_vid = split_infos[0].data.item()
		max_len_target = split_infos[1].data.item()

		mel_padded = to_var(mel_padded).float()

		return(
			(vid_padded, input_lengths, mel_padded, max_len_vid, target_lengths, item_names),
			(mel_padded, wav_padded))

	def parse_output(self, outputs, output_lengths=None):
		if self.mask_padding and output_lengths is not None:
			mask = ~get_mask_from_lengths(output_lengths, True)  # (B, T)
			mask = mask.expand(self.num_mels, mask.size(0),
							   mask.size(1))  # (80, B, T)
			mask = mask.permute(1, 0, 2)  # (B, 80, T)

			outputs[0].data.masked_fill_(mask, 0.0)  # (B, 80, T)
			outputs[1].data.masked_fill_(mask, 0.0)  # (B, 80, T)
			slice = torch.arange(0, mask.size(2), self.n_frames_per_step)
			outputs[2].data.masked_fill_(mask[:, 0, slice], 1e3)

		return outputs

	def forward(self, inputs):
		vid_inputs, vid_lengths, mels, max_len, output_lengths, item_names = inputs
		vid_lengths, output_lengths = vid_lengths.data, output_lengths.data

		embedded_inputs = vid_inputs.type(torch.FloatTensor).to(next(self.parameters()).device)
		encoder_outputs = self.encoder(embedded_inputs)

		decoder_output, mel_output = self.decoder_nar(encoder_outputs, mels)

		mel_output = mel_output.permute(0, 2, 1)
		return decoder_output, mel_output

	def inference(self, inputs, mode='train'):
		if mode == 'train':
			vid_inputs, vid_lengths, mels, max_len, output_lengths = inputs
		else:
			vid_inputs = inputs.float()
			# vid_inputs = to_var(torch.from_numpy(vid_inputs)).float()
			# vid_inputs = vid_inputs.permute(
			# 	3, 0, 1, 2).unsqueeze(0).contiguous()

		embedded_inputs = vid_inputs.type(torch.FloatTensor).to(next(self.parameters()).device)
		encoder_outputs = self.encoder.inference(embedded_inputs)

		decoder_out, mel_output = self.decoder_nar.inference(encoder_outputs)
		mel_output = mel_output.permute(0, 2, 1)
		return decoder_out, mel_output


def slice_segments(x, ids_str, segment_size=4):
	ret = torch.zeros_like(x[:, :, :segment_size])
	for i in range(x.size(0)):
		idx_str = ids_str[i]
		idx_end = idx_str + segment_size
		ret[i] = x[i, :, idx_str:idx_end]
	return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
	b, d, t = x.size()
	if x_lengths is None:
		x_lengths = t
	ids_str_max = x_lengths - segment_size + 1
	ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
	ret = slice_segments(x, ids_str, segment_size)
	return ret, ids_str


class Lip2WavDecoder(nn.Module):
	def __init__(self):
		super(Lip2WavDecoder, self).__init__()
		self.linear = Linear(hps.encoder_embedding_dim, 80, bias=True)
		self.hifi_hps = {
			'resblock_kernel_sizes':[ 3,7,11 ],
			'resblock_dilation_sizes': [ [ 1,3,5 ], [ 1,3,5 ], [ 1,3,5 ] ],
			'upsample_rates': [ 5,5,4,2 ],
			'upsample_kernel_sizes': [ 9,9,8,4 ],
			'upsample_initial_channel': 128,
			'resblock': "1"
		}
		self.vocoder = HifiGanGenerator(self.hifi_hps)

	def forward(self, x):
		x = x.permute(0, 2, 1)
		x_sliced, ids = rand_slice_segments(x, segment_size=hps.mel_segment_size)
		x_sliced = x_sliced.permute(0, 2, 1)
		
		lin_out = self.linear(x_sliced)
		lin_out = lin_out.permute(0, 2, 1)
		wav_out = self.vocoder(lin_out)

		return lin_out, wav_out, ids
	
	def inference(self, x):
		lin_out = self.linear(x)
		lin_out = lin_out.permute(0, 2, 1)
		wav_out = self.vocoder(lin_out)

		return lin_out, wav_out


class Lip2Wav(nn.Module):
	def __init__(self):
		super(Lip2Wav, self).__init__()
		self.frontend = Lip2Mel()
		self.vocoder = Lip2WavDecoder()
		
	def forward(self, x):
		decoder_out, mel_out = self.frontend(x)
		lin_out, wav_out, ids = self.vocoder(decoder_out)
		
		return mel_out, lin_out, wav_out, ids

	def inference(self, x, mode='test'):
		# torch.cuda.synchronize()
		
		decoder_out, _ = self.frontend.inference(x, mode)
		_, wav_out = self.vocoder.inference(decoder_out)

		return wav_out

	