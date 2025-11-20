import argparse
import os
from os import listdir
from os.path import isfile, join
import librosa as rosa
import librosa.display as disp
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)
# Needed in case logging has been previously setup on the system with callback handlers
logger.setLevel(logging.INFO)


SUPPORTED_SAMPLE_RATE = 44100
FRAME_LENGTH = 1024
CQT_BINS = 84

def extract_audio_features(songsList, featureDir):
  for song in songsList:
    rawAudioData, sr = rosa.load(path=song, sr=SUPPORTED_SAMPLE_RATE, mono=True, offset=0.0, duration=60.0, dtype='float32')
    logger.info('Computing CQT for song = {} with sample rate={}'.format(song, sr))
    cqtFromAudio = np.abs((rosa.cqt(y=rawAudioData, sr=SUPPORTED_SAMPLE_RATE,
                                hop_length=FRAME_LENGTH, n_bins=CQT_BINS)))
    logger.info('Computing MFCC for song = {} with sample rate={}'.format(song, sr))
    mfccFromAudio = rosa.feature.mfcc(y=rawAudioData, sr=SUPPORTED_SAMPLE_RATE, n_mfcc=20, lifter=40, n_fft= FRAME_LENGTH, hop_length=FRAME_LENGTH)
    mfccEnergy = np.linalg.norm(mfccFromAudio, axis=1, keepdims=True)
    normalizedMfcc = mfccFromAudio / (mfccEnergy + 1e-8) # avoid divide by zero
    logger.info('Generating images from audio features...')
    fig = plt.figure(num=1, figsize=(10,8))
    plt.subplot(2,1,1)
    rosa.display.specshow(normalizedMfcc, x_axis='time', y_axis='time')
    plt.title('MFCC')
    plt.tight_layout()
    plt.subplot(2,1,2)
    rosa.display.specshow(rosa.amplitude_to_db(cqtFromAudio, ref=np.max), sr=SUPPORTED_SAMPLE_RATE,
                                        x_axis='time', y_axis='cqt_hz', hop_length=FRAME_LENGTH)
    plt.title('CQT')
    pltFile = featureDir / Path(song.stem + '.png')
    pltFile.unlink() if pltFile.exists() else None
    fig.savefig(str(pltFile))
    plt.close(fig)


if __name__ == '__main__':
    aParser = argparse.ArgumentParser(description="Feature extraction from audion songs")
    aParser.add_argument("songs_dir", type=str, help="Folder/directory path with audio files (mp3 format)")
    aParser.add_argument("out_dir", type=str, nargs='?', default=os.getcwd(), help="Subfolder for extracted features (png format)")

    aArgs = aParser.parse_args()

    songsDirPath = Path(aArgs.songs_dir)
    songsListing = listdir(aArgs.songs_dir)
    logger.info(f"Processing songs from {songsDirPath}\nComplete List: {songsListing}")

    songsIn = [songsDirPath / Path(f) for f in songsListing if f.endswith("mp3") ]
    #for file in songsIn:
    #  if str(file).endswith("mp3"):
    #    logger.info(f'Audio file found! {file}')

    featureDirPath = Path(aArgs.out_dir)
    if aArgs.out_dir != os.getcwd():
        featureDirPath = songsDirPath / Path(aArgs.out_dir)
        featureDirPath.mkdir(parents=True, exist_ok=True)

    logger.info(f'Generating audio features in {featureDirPath}')
    extract_audio_features(songsIn, featureDirPath)
