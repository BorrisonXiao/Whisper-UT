#!/usr/bin/env python
import os
import logging
from typing import List, Tuple, Dict
from datasets import Dataset, Features, Audio, Value
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import local.data_prep.stm as stm
import argparse
from pathlib import Path

# Constants
AUDIO_SAMPLING_RATE = 16000
TARGET_LANGUAGE = 'eng'
# IGNORE_LIST = [
#     'ara.train-all_sp', 'ara.train-cts_sp', 'ara.iwslt22_test', 'ara.fleurs_test',
#     'cmn.train-all_sp', 'cmn.train-cts_sp', 'cmn.bbn_cts_bolt_test', 'cmn.fleurs_test',
#     'kor.train-all_sp', 'kor.train-cts_sp',  'kor.fleurs_test', 'kor.uhura_test',
#     'rus.train-all_sp', 'rus.train-cts_sp', 'rus.uhura_test', 'rus.fleurs_test',
#     'spa.train-all_sp', 'spa.train-cts_sp', 'spa.dev2', 'spa.dev1', 'spa.callhome_test', 'spa.fleurs_test',
# ]
IGNORE_LIST = [
    'ara.train-all_sp', 'ara.train-cts_sp',
    'cmn.train-all_sp', 'cmn.train-cts_sp',
    'kor.train-all_sp', 'kor.train-cts_sp',
    'rus.train-all_sp', 'rus.train-cts_sp',
    'spa.train-all_sp', 'spa.train-cts_sp',
    'spa.fisher_test', 'spa.dev2', 'spa.dev1', 'spa.train-cts', 'spa.train-ood', 'spa.fleurs_test', 'spa.mtedx_test',
]

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
        if task == "sr":
            st_file = f"st.{src_lang}-{tgt_lang}.{dataset}.stm"
            if stm_file in stm_files:  # st file should be present in the directory
                stm_files_list.append(
                    (stm_file, os.path.join(os.path.dirname(stm_file), st_file)))

    return stm_files_list


def read_stm_file(stm_file_path: str, label="transcript") -> Dict[str, str]:
    """ Read stm file and return a dictionary with utterance id as the key and the transcript/translation as the value.
    """
    stm_dict = dict()
    with open(stm_file_path, mode="r") as stm_file:
        lines = stm_file.readlines()
        for line in lines:
            utterance_id = stm.parse_StmUtterance(
                line).utterance_id(stereo=True)
            if label == "transcript":
                transcript = stm.parse_StmUtterance(line).transcript()
                stm_dict[utterance_id] = transcript
            elif label == "translation":
                translation = stm.parse_StmUtterance(line).transcript()
                stm_dict[utterance_id] = translation

    return stm_dict


def find_text_files_from_wav_file(wav_file_path: str, directory: str) -> Tuple[str, str]:
    """
    Find transcript and translation files from the given wav file.
    Translation files have a hyphen between the source and target languages.

    Args:
        wav_file_path (str): Path to the wav file.

    Returns:
        Tuple[str, str]: Path to the transcript and translation files.


    Example:
        >>> wav_file_path = "./ara.dev.wav.scp"
        >>> directory = "./"
        >>> find_text_files_from_wav_file(wav_file_path, directory)
        ("./ara.dev.text", "./ara-eng.dev.text")
    """
    wav_file_name = os.path.splitext(
        os.path.splitext(os.path.basename(wav_file_path))[0])[0]
    logging.info(f"wav_file_name: {wav_file_name}")

    transcript_file, translation_file = None, None
    file_dict = {os.path.splitext(os.path.basename(file))[0]: os.path.join(
        root, file) for root, _, files in os.walk(directory) for file in files if file.endswith(".text")}

    transcript_file = file_dict.get(wav_file_name)
    translation_file = file_dict.get(
        f"{wav_file_name.split('.')[0]}-{TARGET_LANGUAGE}.{wav_file_name.split('.')[1]}")

    return transcript_file, translation_file


def find_wav_files(directory: str) -> List[str]:
    """
    Find all wav files in the given directory.

    Args:
        directory (str): Path to the directory.

    Returns:
        List[str]: List of wav files in the directory.

    Example:
        >>> directory = "."
        >>> find_wav_files(directory)
        ["./ara.dev.wav.scp", "./ara.train.wav.scp"]
    """
    # wav_files = [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.endswith(".wav.scp")]
    # return wav_files

    wav_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if IGNORE_LIST:
                if file.endswith(".wav.scp") and not any(file.startswith(ignore) for ignore in IGNORE_LIST):
                    wav_files.append(os.path.join(root, file))
            else:
                if file.endswith(".wav.scp"):
                    wav_files.append(os.path.join(root, file))

    return wav_files


def process_lines(wav_file, transcript_file, translation_file):
    """
    Process lines from wav, transcript and translation files.

    Args:
        wav_file (str): Path to the wav file.
        transcript_file (str): Path to the transcript file.
        translation_file (str): Path to the translation file.

    Returns:
        List[Tuple[str, str, str]]: List of tuples containing lines from wav, transcript and translation files.

    Example:
        >>> wav_file = "./ara.dev.wav.scp"
        >>> transcript_file = "./ara.dev.text"
        >>> translation_file = "./ara-eng.dev.text"
        >>> process_lines(wav_file, transcript_file, translation_file)
        # Returns list of tuples containing lines from wav, transcript and translation files
    """
    reordered_lines = []
    # with open(wav_file, mode="r") as w_f, open(transcript_file, mode="r") as t_f, open(translation_file, mode="r") as tr_f:
    #     for wav_line, transcript_line, translation_line in zip(w_f, t_f, tr_f):
    #         wav_line, transcript_line, translation_line = wav_line.strip(
    #         ), transcript_line.strip(), translation_line.strip()
    #         utterance_id, wav_file_path = wav_line.split()
    #         transcript_key, transcript = transcript_line.split(
    #         )[0], " ".join(transcript_line.split()[1:])
    #         translation_key, translation = translation_line.split(
    #         )[0], " ".join(translation_line.split()[1:])

    #         while utterance_id != transcript_key != translation_key:
    #             with open(transcript_file, mode="r") as t_f, open(translation_file, mode="r") as tr_f:
    #                 for t_line, tr_line in zip(t_f, tr_f):
    #                     t_line, tr_line = t_line.strip(), tr_line.strip()
    #                     transcript_key, transcript = t_line.split(
    #                     )[0], " ".join(t_line.split()[1:])
    #                     translation_key, translation = tr_line.split(
    #                     )[0], " ".join(tr_line.split()[1:])
    #                     if utterance_id == transcript_key == translation_key:
    #                         transcript_line, translation_line = t_line, tr_line
    #                         break

    #         reordered_lines.append(
    #             (wav_line, transcript_line, translation_line))
    #         logging.debug(reordered_lines[:3])
    res = {}
    with open(wav_file, mode="r") as w_f, open(transcript_file, mode="r") as t_f, open(translation_file, mode="r") as tr_f:
        for line in w_f:
            line = line.strip()
            utterance_id, wav_file_path = line.split()
            res[utterance_id] = {"wav_line": line}
        for line in t_f:
            line = line.strip()
            transcript_key, transcript = line.split()[0], " ".join(line.split()[1:])
            res[transcript_key]["transcript_line"] = line
        for line in tr_f:
            line = line.strip()
            translation_key, translation = line.split()[0], " ".join(line.split()[1:])
            res[translation_key]["translation_line"] = line

    for key, value in res.items():
        reordered_lines.append((value["wav_line"], value["transcript_line"], value["translation_line"]))

    return reordered_lines


def load_audio(reordered_lines, wav_file):
    """
    Load audio from reordered lines.

    Args:
        reordered_lines (List[Tuple[str, str, str]]): List of tuples containing lines from wav, transcript and translation files.
        wav_file (str): Path to the wav file.

    Returns:
        Dict[str, List]: Dictionary containing audio files, utterance ids, transcripts, translations, source languages and target languages.

    Example:
        >>> reordered_lines = [("wav_line1", "transcript_line1", "translation_line1"), ("wav_line2", "transcript_line2", "translation_line2")]
        >>> wav_file = "./ara.dev.wav.scp"
        >>> load_audio(reordered_lines, wav_file)
        # Returns dictionary containing audio files, utterance ids, transcripts, translations, source languages and target languages
    """
    audio_files, utterance_ids, transcripts, translations = [], [], [], []
    src_lang, tgt_lang = [], []

    for wav_line, transcript_line, translation_line in tqdm(reordered_lines, desc=f"Loading audio: {wav_file}"):
        wav_line, transcript_line, translation_line = wav_line.strip(
        ), transcript_line.strip(), translation_line.strip()
        utterance_id, wav_file_path = wav_line.split()
        transcript_key, transcript = transcript_line.split(
        )[0], " ".join(transcript_line.split()[1:])
        translation_key, translation = translation_line.split(
        )[0], " ".join(translation_line.split()[1:])

        audio_files.append(wav_file_path)
        utterance_ids.append(utterance_id)
        transcripts.append(transcript)
        translations.append(translation)
        src_lang.append(os.path.splitext(os.path.splitext(
            os.path.basename(wav_file))[0])[0].split('.')[0])
        tgt_lang.append(TARGET_LANGUAGE)

    return {"audio": audio_files, "uttid": utterance_ids, "transcript": transcripts, "translation": translations, "src_lang": src_lang, "tgt_lang": tgt_lang}


def create_dataset(dataset, transcript_file):
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


def process_wav_text_file(wav_text_file: Tuple[str, str, str]) -> Tuple[str, Dataset]:
    """
    Process a wav and text file and create a HuggingFace dataset.

    Args:
        wav_text_file (Tuple[str, str, str]): Tuple containing paths to the wav, transcript and translation files.

    Returns:
        Tuple[str, Dataset]: Tuple containing the name of the transcript file and the HuggingFace dataset.

    Example:
        >>> wav_text_file = ("./ara.dev.wav.scp", "./ara.dev.text", "./ara-eng.dev.text")
        >>> process_wav_text_file(wav_text_file)
        # Returns the name of the transcript file and the HuggingFace dataset
    """
    wav_file, transcript_file, translation_file = wav_text_file
    logging.info(f"----------------------------------\n\
                  Processing following wav file: {wav_file}\n\
                  ----------------------------------")
    reordered_lines = process_lines(
        wav_file, transcript_file, translation_file)
    dataset = load_audio(reordered_lines, wav_file)
    return os.path.splitext(os.path.basename(transcript_file))[0], create_dataset(dataset, transcript_file)


def create_dataset_from_wav_text_files(directory: str) -> Dict[str, Dataset]:
    """
    Create a HuggingFace dataset from wav and text files in the given directory.

    Args:
        directory (str): Path to the directory.

    Returns:
        Dict[str, Dataset]: Dictionary of HuggingFace datasets.

    Example:
        >>> directory = "."
        >>> create_dataset_from_wav_text_files(directory)
        # Returns a dictionary of HuggingFace datasets
    """
    all_wav_files = find_wav_files(directory)

    print(f"Number of wav files: {len(all_wav_files)}")

    import pprint
    pprint.pprint(f"Processing following wav files: {all_wav_files}")

    wav_text_files = [(wav_file, *find_text_files_from_wav_file(wav_file, directory))
                      for wav_file in all_wav_files]
    logging.info(
        f"Found {len(all_wav_files)} wav files and corresponding text files.")

    hf_datasets = {}

    with ProcessPoolExecutor() as executor:
        for text_file_name, dataset in executor.map(process_wav_text_file, wav_text_files):
            hf_datasets[text_file_name] = dataset
    # for (text_file_name, dataset) in process_wav_text_file(wav_text_files[0]):
    #     hf_datasets[text_file_name] = dataset

    return hf_datasets


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

    hf_datasets = create_dataset_from_wav_text_files(raw_data_location)

    for key, dataset in hf_datasets.items():
        dataset_path = f"{output_path}/{key}/"

        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        dataset.save_to_disk(dataset_path)


if __name__ == "__main__":
    main()
