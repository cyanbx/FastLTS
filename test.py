import os
import torch
import argparse
import utils.audio as audio

from tqdm import tqdm
from model.model import Lip2Wav
from hparams import hparams as hps
from torch.utils.data import DataLoader
from utils.dataset import VideoMelLoader, VMcollate


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def prepare_dataloaders_vid_test(fdir, bs):
	testset = VideoMelLoader(fdir, hps, "test")
	collate_fn = VMcollate(hps, hps.n_frames_per_step)
	test_loader = DataLoader(testset, num_workers = hps.n_workers, shuffle = False,
							  batch_size = bs, pin_memory = hps.pin_mem,
							  drop_last = True, collate_fn = collate_fn)
	return test_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
	# path	
    parser.add_argument('-d', '--data_dir', type = str, default = 'data',
						help = 'directory to load data')
    parser.add_argument('-cp', '--ckpt_pth', type = str, default = '',
						help = 'path to load checkpoints')
    parser.add_argument('-o', '--output_dir', type = str, default = '',
						help = 'path to load output directory')
    parser.add_argument('-bs', '--batch_size', type = int, default = 1,
						help = 'inference batch size')
    

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
	
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    test_loader = prepare_dataloaders_vid_test(args.data_dir, args.batch_size)

    net = Lip2Wav().to(device)

    ckpt_dict = torch.load(args.ckpt_pth)
    net.load_state_dict(ckpt_dict['model'])
    
    for bid, batch in tqdm(enumerate(test_loader)):
        try:
            x, y = net.frontend.parse_batch_wav(batch)
            item_names = x[-1]
            wav_pred = net.inference(x[0])
            for i in range(len(item_names)):
                audio.save_wav(wav_pred[i].squeeze(0).detach().cpu().numpy(), os.path.join(args.output_dir, '{}.wav'.format(item_names[i])), sr=16000)
        except:
             print('WARNING: CORRUPTED DATA: {}'.format(bid))
