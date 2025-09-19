#!/usr/bin/env python
import os
import logging
from typing import List, Tuple, Dict
from datasets import Dataset, Features, Audio, Value
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import local.data_prep.stm as stm
import multiprocessing
from functools import partial
import argparse
from pathlib import Path

# Constants
AUDIO_SAMPLING_RATE = 16000
TARGET_LANGUAGE = 'eng'
# Already processed files
DATA_FILES = [
    'ara.train-all_sp', 'ara.train-cts_sp', 'ara.iwslt22_test', 'ara.fleurs_test',
    'cmn.train-all_sp', 'cmn.train-cts_sp', 'cmn.bbn_cts_bolt_test', 'cmn.fleurs_test',
    'kor.train-all_sp', 'kor.train-cts_sp',  'kor.fleurs_test', 'kor.uhura_test',
    'rus.train-all_sp', 'rus.train-cts_sp', 'rus.uhura_test', 'rus.fleurs_test',
    'spa.train-all_sp', 'spa.train-cts_sp', 'spa.fisher_test', 'spa.callhome_test', 'spa.fleurs_test',
]
# DATA_FILES = [ 'ara.train-all_sp', 'spa.train-all_sp' ]
# DATA_FILES = [ 'ara.train-all_sp', 'spa.train-all_sp' ]

# Read stm file for transcript and translation respectively
# Convert the utterance file path to the utterance id
# Create a dictionary with the utterance id as the key and the transcript and translation as the value
# Create a HuggingFace dataset from the dictionary

# stm file name pattern: <task>.<src_lang>-<tgt_lang>.<dataset>.stm
# e.g. st.ara-eng.dev.stm, sr.ara-ara.train.stm, st.ara-eng.test.stm
# sr indicates speech recognition
# st indicates speech translation


def parse_stm_filename(stm_file_path: str) -> Tuple[str, str, str, str]:
    """ Parse stm file name and return task, source language and target language respectively.

    Example:
        >>> stm_file_path = "./st.ara-eng.dev.stm"
        >>> parse_stm_filename(stm_file_path)
        ("st", "ara", "eng", "dev")
    """
    stm_file_name = os.path.splitext(os.path.basename(stm_file_path))[0]
    task, src_lang_tgt_lang, dataset = stm_file_name.split('.')
    src_lang, tgt_lang = src_lang_tgt_lang.split('-')
    return task, src_lang, tgt_lang, dataset

# In a directory with different stm files, find the stm files correponding to the same dataset
# e.g. st.ara-eng.dev.stm, sr.ara-ara.dev.stm


def find_stm_files(directory: str) -> List[Tuple[str, str]]:
    """ Return a list of sr and st files from the given directory with each tuple containing the sr and st file from the same dataset.

    Example:
        >>> directory = "."
        >>> find_stm_files(directory)
        [("sr.ara-eng.dev.stm", "st.ara-ara.dev.stm"), ("sr.ara-eng.dev1.stm", "st.ara-ara.dev1.stm")]
    """
    # Find all stm files in the directory
    stm_files = [os.path.join(root, file) for root, _, files in os.walk(
        directory) for file in files if file.endswith(".stm")]

    # Create a list of tuples with each tuple containing the sr and st file from the same dataset
    stm_files_list = []
    for stm_file in stm_files:
        task, src_lang, tgt_lang, dataset = parse_stm_filename(stm_file)
        if task == "st":
            sr_file = f"{os.path.join(os.path.dirname(stm_file))}/sr.{src_lang}-{src_lang}.{dataset}.stm"
            if sr_file in stm_files:  # st file should be present in the directory
                stm_files_list.append((sr_file, stm_file))

    return stm_files_list


def read_stm_file(stm_file_path: str,) -> Dict[str, Tuple[str, str]]:
    """ Read stm file and return a dictionary with utterance id as the key and the wav file and transcript as the value.
    """
    stm_dict = dict()
    with open(stm_file_path, mode="r") as stm_file:
        lines = stm_file.readlines()
        for line in lines:
            utterance_id = stm.parse_StmUtterance(
                line).utterance_id(stereo=True)
            transcript = stm.parse_StmUtterance(line).transcript
            wav_file_name = stm.parse_StmUtterance(line).filename
            stm_dict[utterance_id] = (wav_file_name, transcript)

    return stm_dict


def get_transcript_translation_dict(stm_file: Tuple[str, str]) -> Dict[str, Tuple[str, str, str, str, str]]:
    """ Return a dictionary with utterance id as the key and values are src_lang, tgt_lang, wav_file, transcript and translation.
    """
    transcript_translation_dict = dict()
    sr_file, st_file = stm_file
    # print(f"Processing following sr file: {sr_file}")
    sr_dict = read_stm_file(sr_file)
    st_dict = read_stm_file(st_file)
    # parse_stm_filename
    task, src_lang, tgt_lang, dataset = parse_stm_filename(st_file)

    for utterance_id, (wav_file, transcript) in sr_dict.items():
        transcript_translation_dict[utterance_id] = (
            src_lang, tgt_lang, wav_file, transcript, st_dict[utterance_id][1])

    return transcript_translation_dict

# 914769nou_B1-ma_4972-B_00177070_00179462-A_00000000_00002880


def preprocess_for_hf_dataset(transcript_translation_dict: Dict[str, Tuple[str, str, str, str, str]], raw_data_location: str):
    """ Load audio from the given dictionary and create a HuggingFace dataset.

    Example:
        >>> transcript_translation_dict = {
        ...     "914769nou_B1-ma_4972-B_00177070_00179462-A_00000000_00002880": ('cmn', 'eng', 
        '/home/hltcoe/cxiao/scale23/whisper/recipe/st/dump/cmn/merged/dev1/format.61/data/914769nou_B1-ma_4972-B_00177070_00179462.wav', 
        '就是在那个什么 唔就就是哎呀我说那个啊 AROUND QUIET MALL 什么那那那一片儿 AROUND QUIET 那个 MALL 那边儿 是吗 是吗 对它那儿有专营一个卖但是它不是经常卖 是吗 就说它停一段儿它会有一 有有便宜哈我到时候看看哎差不多快半个小时嘞我怕到时你们那 对 你可以看 哎你不用急它半个小时人家会告诉你说的', 
        "just in that you know um that is ah I meant that AROUND QUIET MALL that that that area AROUND QUITE that MALL there are you are you yes there's a franchised store but it's not selling all the time are you it would stop for a while it would have a would be cheaper ha I would have a look by the time ah it's almost half an hour I am afraid later you yes you can have a look ah you don't have to rush they would tell you if it's half and hour"),

    """
    audio_files, utterance_ids, transcripts, translations, src_langs, tgt_langs = [
    ], [], [], [], [], []
    for utterance_id, (src_lang, tgt_lang, wav_file, transcript, translation) in tqdm(transcript_translation_dict.items()):
        audio_files.append(os.path.join(raw_data_location, wav_file))
        utterance_ids.append(utterance_id)
        transcripts.append(transcript)
        translations.append(translation)
        src_langs.append(src_lang)
        tgt_langs.append(tgt_lang)

    return {"audio": audio_files, "uttid": utterance_ids, "transcript": transcripts, "translation": translations, "src_lang": src_langs, "tgt_lang": tgt_langs}


def create_hf_dataset(dataset, transcript_file):
    """
    Create a HuggingFace dataset from the given dataset.

    Args:
        dataset (Dict[str, List]): Dictionary containing audio files, utterance ids, transcripts, translations, source languages and target languages.
        transcript_file (str): Path to the transcript file.

    Returns:
        Dataset: HuggingFace dataset.

    Example:
        >>> dataset = {
        ...     "audio": ["audio_file1", "audio_file2"],
        ...     "uttid": ["uttid1", "uttid2"],
        ...     "transcript": ["transcript1", "transcript2"],
        ...     "translation": ["translation1", "translation2"],
        ...     "src_lang": ["src_lang1", "src_lang2"],
        ...     "tgt_lang": ["tgt_lang1", "tgt_lang2"],
        ... }
        >>> transcript_file = "./ara.dev.text"
        >>> create_dataset(dataset, transcript_file)
        # Returns a HuggingFace dataset
    """
    text_file_name, _ = os.path.splitext(os.path.basename(transcript_file))
    logging.info("-----------------------------")
    logging.info(text_file_name)
    logging.info("-----------------------------")

    features = Features({
        "audio": Audio(
            sampling_rate=AUDIO_SAMPLING_RATE,
        ),
        "uttid": Value(dtype="string"),
        "transcript": Value(dtype="string"),
        "translation": Value(dtype="string"),
        "src_lang": Value(dtype="string"),
        "tgt_lang": Value(dtype="string"),
    })

    return Dataset.from_dict(dataset, features=features)


def process_stm_file_pair(stm_file_pair, raw_data_location, output_path, data_files=DATA_FILES):
    # create a huggingface dataset from the stm file
    sr_file, st_file = stm_file_pair

    task, src_lang, tgt_lang, dataset_name = parse_stm_filename(st_file)
    dataset_path = f"{output_path}/{src_lang}.{dataset_name}/"

    if f"{src_lang}.{dataset_name}" not in data_files and (not os.path.exists(dataset_path) or os.path.exists(dataset_path) and not os.listdir(dataset_path)):
        # combine the sr and st files into a dictionary
        transcript_translation_dict = get_transcript_translation_dict(
            stm_file_pair)

        dataset = preprocess_for_hf_dataset(
            transcript_translation_dict, raw_data_location)

        hf_dataset = create_hf_dataset(dataset, st_file)

        # save the dataset to disk
        task, src_lang, tgt_lang, dataset_name = parse_stm_filename(st_file)

        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        hf_dataset.save_to_disk(dataset_path)

    else:
        print(
            f"Skipping {src_lang}.{dataset_name} as it already exists or it's in the skipping list.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-location", type=str,
                        default="/exp/cxiao/scale23/merged_data_base/",
                        help="Path to the raw data location.")
    parser.add_argument("--output-path", type=str,
                        default="/exp/cxiao/scale23/_merged_hf_data/",
                        help="Path to the output location.")
    parser.add_argument("--src-lang", type=str, default=None,
                        help="Source language.")
    args = parser.parse_args()

    raw_data_location = args.raw_data_location
    output_path = args.output_path

    print(f"raw_data_location: {raw_data_location}")
    print(f"output_path: {output_path}")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    stm_files = find_stm_files(raw_data_location)
    if args.src_lang:
        stm_files = [
            stm_file for stm_file in stm_files if args.src_lang in Path(stm_file[0]).stem]
        
    print(f"stm_files: {stm_files}")

    # Create a new function with raw_data_location and output_path pre-filled
    process_stm_func = partial(
        process_stm_file_pair, raw_data_location=raw_data_location, output_path=output_path)

    with multiprocessing.Pool() as pool:
        max_ = len(stm_files)
        with tqdm(total=max_, desc=f"Creating HuggingFace datasets from wav and text files in {raw_data_location}") as pbar:
            for i, _ in tqdm(enumerate(pool.imap_unordered(process_stm_func, stm_files))):
                pbar.update()
                
    # # The for-loop single-process version
    # for stm_file in tqdm(stm_files):
    #     process_stm_file_pair(stm_file, raw_data_location, output_path)


if __name__ == "__main__":
    main()