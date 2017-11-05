# -*- coding: utf-8 -*-

# Import libraries
import scipy.io.wavfile
import scipy
import pydub
import glob
import sys
import os
import numpy as np
import json

import spectrogram_extractor as SpectrogramExtractor

# Constants
MFCC = 'mfcc'
SPECTROGRAM = 'spectrogram'
SEARCH_LEVEL = 3
TEMP_FOLDER = '/temp/'
EXTRACTIONS = {
    MFCC,
    SPECTROGRAM
}


class FeatureExtractor(object):

    def __init__(self, extractions={MFCC, SPECTROGRAM}):
        self.extractions = extractions

    def set_extractions(self, extractions=None):
        self.extractions = extractions

    @staticmethod
    def load_mp3(mp3_file_path, temp_folder_path=''):
        # read mp3 file
        mp3 = pydub.AudioSegment.from_mp3(mp3_file_path)

        # convert to wav
        if not os.path.exists(temp_folder_path + '/' + TEMP_FOLDER):
            os.makedirs(temp_folder_path + '/' + TEMP_FOLDER)
        mp3.export(temp_folder_path + TEMP_FOLDER + 'file.wav', format='wav')

        # read wav file
        rate, audio_data_stereo = scipy.io.wavfile.read(
            temp_folder_path + TEMP_FOLDER + 'file.wav')

        audio_data_mono = np.sum(audio_data_stereo.astype(float), axis=1) / 2

        return rate, audio_data_mono

    @staticmethod
    def extract_features(audio_data, rate, extractions):
        features = dict()
        if SPECTROGRAM in extractions:
            features[SPECTROGRAM] = SpectrogramExtractor.get_spectrogram(
                audio_data, rate)
        if MFCC in extractions:
            # features[MFCC] =
            pass

        return features

    @staticmethod
    def write_to_file(features, output_folder_path, input_file_path, genres_dict):
        for feature_type, feature in features.items():
            infile_name = input_file_path.split('/')[-1].split('.')[0]
            outdir = output_folder_path + '/' + feature_type + '/' + \
                genres_dict.get(infile_name, ['other'])[0]
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outfile_name = outdir + '/' + infile_name
            np.save(outfile_name, feature)

    def process_folder(self, input_folder_path=None, output_folder_path=None, genres_dict=None):
        if input_folder_path is not None:
            features_dict = dict()
            audio_file_paths = list()

            for search_level in range(SEARCH_LEVEL):
                audio_file_paths.extend(
                    glob.glob(input_folder_path + ('/*' * search_level) + '/*.mp3'))

            # Process the mp3 files
            for audio_file_path in audio_file_paths:
                try:
                    # Load the mp3 file
                    rate, audio_data = FeatureExtractor.load_mp3(
                        audio_file_path, output_folder_path)

                    if rate != 44100:
                        audio_data = scipy.signal.resample(
                            audio_data, (int((len(audio_data) / rate) * 44100)))
                        rate = 44100

                    # Get features
                    features = FeatureExtractor.extract_features(
                        audio_data, rate, self.extractions)

                    # Write to file
                    FeatureExtractor.write_to_file(
                        features, output_folder_path, audio_file_path, genres_dict)
                except:
                    print(audio_file_path)


def main():
    fe = FeatureExtractor(extractions=EXTRACTIONS)
    input_folder_path = sys.argv[1]
    output_folder_path = sys.argv[2]
    genres_dict = json.load(open(sys.argv[3], 'r'))
    fe.process_folder(input_folder_path, output_folder_path, genres_dict)

if __name__ == '__main__':
    main()
