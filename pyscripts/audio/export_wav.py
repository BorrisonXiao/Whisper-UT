#!/usr/bin/env python
# Copyright 2023 Johns Hopkins University (Cihan Xiao)
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re
from tqdm import tqdm
import shutil
from local.data_prep.stm import parse_StmUtterance


def export_wav(dumpdir, src_lang, outdir, refdir, split_dev=False):
    """
    Helper script for exporting the wav_raw.scp files (from relative to absolute path).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    lang = dumpdir.name
    rawdir = dumpdir / "raw"
    _dumpdir = dumpdir if dumpdir.parts[-1] != src_lang else dumpdir.parent
    dump_root = _dumpdir.absolute()
    # Iterate over testsets first
    testsetdirs = list(rawdir.glob("*_test")) + list(rawdir.glob("dev*"))
    for testsetdir in testsetdirs:
        setname = testsetdir.name
        srcfile = testsetdir / "wav_raw.scp"
        if not srcfile.exists():
            print("Warning: {} does not exist.".format(srcfile))
            continue
        with open(srcfile, "r") as f:
            lines = f.readlines()
        with open(outdir / f"{lang}.{setname}.wav.scp", "w") as f:
            for line in lines:
                utt, wav = line.strip().split(maxsplit=1)

                # Remove the leading "dump[.*]/" in the path
                # wav = re.sub(r"^dump/", "", wav)
                wav = re.sub(r"^dump[0-9a-zA-Z\_]*/", "", wav)
                print(f"{utt} {dump_root / wav}", file=f)
        # Copy over the text file
        shutil.copy(testsetdir / "text", outdir / f"{lang}.{setname}.text")
        shutil.copy(testsetdir / "text.tc.eng", outdir /
                    f"{lang}-eng.{setname}.text")
    # Iterate over org/(dev|train) sets
    for dsetdir in tqdm(rawdir.glob("org/*"), desc=f"Processing {lang}"):
        setname = dsetdir.name
        srcfile = dsetdir / "wav_raw.scp"
        textfile = dsetdir / "text" if (dsetdir / "text").exists() else rawdir / setname / "text"
        transfile = dsetdir / "text.tc.eng" if (dsetdir / "text.tc.eng").exists() else rawdir / setname / "text.tc.eng"
        if not srcfile.exists():
            print("Warning: {} does not exist.".format(srcfile))
            continue
        with open(srcfile, "r") as f:
            lines = f.readlines()
        with open(textfile, "r") as f:
            lines_text = f.readlines()
        with open(transfile, "r") as f:
            lines_trans = f.readlines()
        if split_dev and "dev" in setname:
            _setname = setname
            # Open the reference dev1 and dev2 files to get the uttids
            with open(refdir / lang / f"sr.{lang}-{lang}.{_setname}.stm", "r") as f:
                lines_ref = f.readlines()
            uttids = [parse_StmUtterance(line).utterance_id(stereo=True) for line in lines_ref]
            for i, out_fname in enumerate([f"{lang}.{_setname}.wav.scp", f"{lang}.{_setname}.text", f"{lang}-eng.{_setname}.text"]):
                with open(outdir / out_fname, "w") as f:
                    if i == 0:
                        _lines = lines
                    elif i == 1:
                        _lines = lines_text
                    else:
                        _lines = lines_trans
                    for line in _lines:
                        utt, content = line.strip().split(maxsplit=1)

                        if utt in uttids:
                            if out_fname.endswith(".wav.scp"):
                                # Remove the leading "dump/" in the path
                                content = re.sub(r"^dump[0-9a-zA-Z\_]*/", "", content)
                                print(f"{utt} {dump_root / content}", file=f)
                            else:
                                print(line.strip(), file=f)
        else:
            with open(outdir / f"{lang}.{setname}.wav.scp", "w") as f:
                for line in lines:
                    utt, wav = line.strip().split(maxsplit=1)

                    # Remove the leading "dump/" in the path
                    wav = re.sub(r"^dump[0-9a-zA-Z\_]*/", "", wav)
                    print(f"{utt} {dump_root / wav}", file=f)
            # Copy over the text file
            shutil.copy(textfile, outdir / f"{lang}.{setname}.text")
            shutil.copy(transfile, outdir /
                        f"{lang}-eng.{setname}.text")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dumpdir", help="dump directory",
                        type=Path, default="/exp/cxiao/scale23/dump_scale23")
    parser.add_argument("--src-lang", help="source language",
                        default=["ara", "cmn", "spa", "rus", "kor"], type=str)
    parser.add_argument("--outdir", help="output directory",
                        type=Path, default="dump/export")
    parser.add_argument("--refdir", help="reference directory",
                        type=Path, default="/exp/scale23/data/3-way")
    parser.add_argument("--split-dev", help="split dev set into dev1 and dev2",
                        action="store_true")
    args = parser.parse_args()

    export_wav(dumpdir=args.dumpdir, src_lang=args.src_lang, outdir=args.outdir, refdir=args.refdir,
               split_dev=args.split_dev)


if __name__ == "__main__":
    main()