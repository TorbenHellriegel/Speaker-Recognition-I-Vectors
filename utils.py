import glob
import os
import random
import subprocess

import h5py
import numpy as np
import resampy
import yaml
from scipy.io import wavfile
from scipy.signal import fftconvolve

EPS = 1e-20
sampling_rate = 16000
data_folder_path = '../../../../../../../../../data/7hellrie'


def safe_makedir(dirname):
    """This function takes a directory name as an argument"""
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def explore_file(filepath):
    """
    This function is used to explore any hdf5 file.
    """
    h5 = h5py.File(filepath, mode="r")
    print(h5.keys())
    for key, value in h5.items():
        print("Key:", key)
        print("Value Type:", value)
        value = np.array(value)
        print("Value Shape:", value.shape)
        print("Value:", value)
        print("="*25)


def convert_wav(inpath, outpath, no_channels, sampling_rate, bit_precision,
                showWarning=False):
    """
    Convert the waves to a pre-defined sampling rate, number of channels and
    bit-precision using SoX tool. So, it should be installed!!
    """
    parent, _ = os.path.split(outpath)
    safe_makedir(parent)
    command = "sox {} -r {} -c {} -b {} {}".format( inpath,
                                                    sampling_rate,
                                                    no_channels,
                                                    bit_precision,
                                                    outpath)
    if showWarning:
        subprocess.call(command, shell=True) 
    else:
        with open(os.devnull, 'w') as FNULL:
            subprocess.call(command, shell=True, stdout=FNULL,
                                                    stderr=subprocess.STDOUT)


def convert_wav_aug(inpath, outpath):
    """
    Applies predetermined augmentations to the wav
    """
    parent, _ = os.path.split(outpath)
    safe_makedir(parent)
    
    rate, sample = wavfile.read(inpath, np.dtype)
    dtype = sample.dtype
    sample = resampy.resample(sample, rate, sampling_rate)
    
    aug0 = 'none'
    (aug1, aug2) = random.sample(set(['music', 'speech', 'noise', 'rir']), 2)
    augmented_sample0 = augment_data(sample, aug0)
    augmented_sample1 = augment_data(sample, aug1)
    augmented_sample2 = augment_data(sample, aug2)
    augmented_sample0 = augmented_sample0.astype(dtype)
    augmented_sample1 = augmented_sample1.astype(dtype)
    augmented_sample2 = augmented_sample2.astype(dtype)
    
    outwav0 = (outpath.split('.')[1]+aug0+'.wav')[1:]
    outwav1 = (outpath.split('.')[1]+aug1+'.wav')[1:]
    outwav2 = (outpath.split('.')[1]+aug2+'.wav')[1:]
    wavfile.write(outwav0, sampling_rate, augmented_sample0)
    wavfile.write(outwav1, sampling_rate, augmented_sample1)
    wavfile.write(outwav2, sampling_rate, augmented_sample2)
    
    return outwav0, outwav1, outwav2


def parse_yaml(filepath="conf.yaml"):
    """
    This method parses the YAML configuration file and returns the parsed info
    as python dictionary.
    Args:
        filepath (string): relative path of the YAML configuration file
    """
    with open(filepath, 'r') as fin:
        try:
            conf_dictionary = yaml.safe_load(fin)
            return conf_dictionary
        except Exception as exc:
            print("ERROR while parsing YAML conf.")
            print(exc)

            
def augment_data(sample, augmentation):
    sample = cut_to_sec(sample, 3)

    if(augmentation == 'music'):
        aug_sample = augment_musan_music(sample)
    elif(augmentation == 'speech'):
        aug_sample = augment_musan_speech(sample)
    elif(augmentation == 'noise'):
        aug_sample = augment_musan_noise(sample)
    elif(augmentation == 'rir'):
        aug_sample = augment_rir(sample)
    else:
        aug_sample = sample

    return aug_sample

def cut_to_sec(sample, length):
    if(len(sample) < sampling_rate*length):
        new_sample = np.pad(sample, (0, sampling_rate*length-len(sample)), 'constant', constant_values=(0, 0))
    else:
        start_point = random.randint(0, len(sample) - sampling_rate*length)
        new_sample = sample[start_point:start_point + sampling_rate*length]
    return new_sample

def add_with_certain_snr(sample, noise, min_snr_db=5, max_snr_db=20):
    sample = sample.astype('int64')
    noise = noise.astype('int64')

    sample_rms = np.sqrt(np.mean(sample**2))
    noise_rms = np.sqrt(np.mean(noise**2))
    wanted_snr = random.randint(min_snr_db, max_snr_db)
    wanted_noise_rms = np.sqrt(sample_rms**2 / 10**(wanted_snr/10))

    new_noise = noise * wanted_noise_rms/(noise_rms+EPS)
    noisy_sample = sample + new_noise
    return noisy_sample

def augment_musan_music(sample):
    musan_music_path = data_folder_path + '/musan/music/*/*.wav'
    #print('load sample: augmenting with musan music')

    song_path = random.choice(glob.glob(musan_music_path))
    rate, song = wavfile.read(song_path, np.dtype)
    song = resampy.resample(song, rate, sampling_rate)

    song = cut_to_sec(song, 3)
    aug_sample = add_with_certain_snr(sample, song, min_snr_db=5, max_snr_db=15)
    return aug_sample

def augment_musan_speech(sample):
    musan_speech_path = data_folder_path + '/musan/speech/*/*.wav'
    #print('load sample: augmenting with musan speech')

    speaker_path = random.choice(glob.glob(musan_speech_path))
    rate, speakers = wavfile.read(speaker_path, np.dtype)
    speakers = resampy.resample(speakers, rate, sampling_rate)
    speakers = cut_to_sec(speakers, 3)

    for i in range(random.randint(2, 6)):
        speaker_path = random.choice(glob.glob(musan_speech_path))
        rate, speaker = wavfile.read(speaker_path, np.dtype)
        speaker = resampy.resample(speaker, rate, sampling_rate)
        speaker = cut_to_sec(speaker, 3)
        speakers = speakers + speaker
        
    aug_sample = add_with_certain_snr(sample, speakers, min_snr_db=13, max_snr_db=20)
    return aug_sample

def augment_musan_noise(sample): #TODO leave noise at 1 sec each or overlap?
    musan_noise_path = data_folder_path + '/musan/noise/*/*.wav'
    #print('load sample: augmenting with musan noise')
    
    for i in range(3):
        noise_path = random.choice(glob.glob(musan_noise_path))
        rate, noise = wavfile.read(noise_path, np.dtype)
        noise = resampy.resample(noise, rate, sampling_rate)
        noise = cut_to_sec(noise, 1)
        sample[i:i+sampling_rate] = add_with_certain_snr(sample[i:i+sampling_rate], noise, min_snr_db=0, max_snr_db=15)

    return sample

def augment_rir(sample):
    rir_noise_path = data_folder_path + '/RIRS_NOISES/simulated_rirs/*/*/*.wav'
    #print('load sample: augmenting with rir')

    rir_path = random.choice(glob.glob(rir_noise_path))
    _, rir = wavfile.read(rir_path, np.dtype)
    aug_sample = fftconvolve(sample, rir)
    aug_sample = aug_sample / abs(aug_sample).max()

    sample_max = abs(sample).max()
    aug_max = abs(aug_sample).max()
    aug_sample = aug_sample * (sample_max/aug_max)

    aug_sample = sample + aug_sample[:len(sample)]
    return aug_sample
