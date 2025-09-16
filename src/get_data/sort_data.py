# The purpose of this programm is to organize the raw data of LibriStutter and LibriSpeech into a format that can be used to train a DL model.

import os
import shutil
from my_utils import coalesce_data, filter_data

if __name__ == "__main__":
    # First sort the LibriStutter and LibriSpeech data into their own large respective folders

    COALESCE = False
    FILTER = True

    path_to_librispeech = './ScienceFair2025/data/LibriSpeech'
    path_to_libristutter = './ScienceFair2025/data/LibriStutter'

    if COALESCE:

        coalesce_data(['./ScienceFair2025/raw_data/LibriSpeech/train-clean-100'], 
                    path_to_librispeech)
        coalesce_data(['./ScienceFair2025/raw_data/LibriStutter/LibriStutter_part_1/LibriStutter_audio', 
                    './ScienceFair2025/raw_data/LibriStutter/LibriStutter_part_2/LibriStutter_audio', 
                    './ScienceFair2025/raw_data/LibriStutter/LibriStutter_part_3/LibriStutter_audio'],
                    path_to_libristutter)
        
    if FILTER:

        filter_data(path_to_librispeech, path_to_libristutter)

    