# FastLTS: Non-Autoregressive End-to-End Unconstrained Lip-to-Speech Synthesis

#### Yongqi Wang, Zhou Zhao | Zhejiang University

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2207.03800) [![Demo](https://img.shields.io/badge/Demo-Video-red.svg?logo=Youtube)](https://www.youtube.com/watch?v=vD8wk3gYrH4)  

This is the PyTorch implementation of FastLTS (ACM MM'22), a non-autoregressive end-to-end model for unconstrained lip-to-speech synthesis.

## Note

The correctness of this open-source version is still under validation. Feel free to create an issue if you find any problems.

## Checkpoints

| Speaker | Checkpoint |
| --- | --- |
| Chemistry Lectures | [Google Drive](https://drive.google.com/drive/folders/1q3cWnDoPxsC5TWf_ehDzyWNs4piqmcrh?usp=drive_link) |
| Chess Analysis | [Google Drive](https://drive.google.com/drive/folders/1q4qDMAwmKGVtw740Udi9wv5sbieyBLJu?usp=drive_link) |
| Hardware Security | [Google Drive](https://drive.google.com/drive/folders/1q88j1v7jUqdupfjs3OcGo45Q2n8DShKM?usp=drive_link) |

## Dependencies

* python >= 3.6
* pytorch >= 1.7.0
* numpy
* scipy
* pillow
* inflect
* librosa
* Unidecode
* matplotlib
* tensorboardX
* ffmpeg `sudo apt-get install ffmpeg`

## Data pre-processing

We adopt the same data format as Lip2Wav. Please download the datasets and following its preprocessing method in the [Lip2Wav repository](https://github.com/Rudrabha/Lip2Wav/tree/master).

## Training

### First stage

Suppose we use the `chess` split in the Lip2Wav dataset. Use the following command for the first stage training.

```shell
python train_stage1.py -d <DATA_DIR>/chess -l <LOG_DIR> -cd <CKPT_DIR>
```

An additional `-cp` argument can be used to restore training from a checkpoint.

### Second stage

When observing the convergence of the first-stage model, load its checkpoint for the second-stage training with the command:

```shell
python train_stage2.py -d <DATA_DIR>/chess -l <LOG_DIR> -cd <CKPT_DIR> -fr <PATH_TO_STAGE1_CKPT>
```

An additional `-cp` argument can be used to restore training from a checkpoint, which should not be used with `-fr` together. Besides, we add distributed data parallel in training of stage2, with can be turned on with `--ddp` and `--ngpu <GPU_NUM>`.


## Inference

```shell
python test.py -d <DATA_DIR>/chess -cp <PATH_TO_STAGE2_CKPT> -o <OUTPUT_DIR> -bs <BATCH_SIZE>
```

## Acknowledgements
Part of the code is borrowed from the following repos. We would like to thank the authors of these repos for their contribution.

* Lip2Wav-pytorch: https://github.com/joannahong/Lip2Wav-pytorch
* NATSpeech: https://github.com/NATSpeech/NATSpeech
* Hifi-GAN: https://github.com/jik876/hifi-gan

## Citations

If you find this code useful in your research, please cite our work:

```bib
@inproceedings{wang2022fastlts,
  title={Fastlts: Non-autoregressive end-to-end unconstrained lip-to-speech synthesis},
  author={Wang, Yongqi and Zhao, Zhou},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={5678--5687},
  year={2022}
}
```