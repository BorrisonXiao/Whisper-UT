#!/usr/bin/env python
# This code is intended for use as the official scoring script for SCALE23. It
# is mostly just a wrapper around sclite, except for it parses text sot that it
# can score using the "Mixed/Mandarin Error Rate" as opposed to just the normal
# WER. This script assumes that all reference files are stored as STMs, and all
# hypothesis files produced are also in the STMs.
#
# The script will produce the standard scoring files that sclite produces:
#    .sys, .raw, .prf, .pra,
#

 

import argparse
from pathlib import Path
import re
from subprocess import check_call, DEVNULL


def _reformat_mixed_mandarin_text(text):
    '''
        Given some text, find consecutive chunks of Mandarin characters and
        convert these portions into individual characters. For the
        non-Mandarin chunks, we keep everything as is. 
    '''
    curr_pos = 0
    transcript = []
    num_syllables_used = [] 
    # A syllable is arbitrarily defined as 2 non-mandarin chars or 1 mandarin
    for group in re.finditer(r'([\u2E80-\u2FD5\u3190-\u319f\u3400-\u4DBF\u4E00-\u9FCC\uF900-\uFAAD]+\s*)+', text):
        # First with start with non-Mandarin
        group_start, group_end = group.start(), group.end()
        if group_start > curr_pos:
            non_mandarin = text[curr_pos:group_start].replace('{', '').replace('}', '')
            non_mandarin_words = non_mandarin.split()
            # One syllable per 2 chars. If there is just 1 char in the word it
            # gets one syllable.
            num_syllables_used.extend([max(1, len(w)//2) for w in non_mandarin_words])
            transcript.extend(non_mandarin_words)

        # Now we do Mandarin
        mandarin = text[group_start:group_end]
        mandarin = re.sub(
            r'([\u2E80-\u2FD5\u3190-\u319f\u3400-\u4DBF\u4E00-\u9FCC\uF900-\uFAAD])',
            r'\1 ', mandarin
        )
        mandarin = re.sub(r' +', ' ', mandarin).split()
        # One syllable per mandarin char.
        num_syllables_used.extend([1]*len(mandarin))
        transcript.extend(mandarin)
        curr_pos = group_end

    # Once we've reached the end of the last mandarin group, the rest is
    # non-mandarin
    if curr_pos < len(text):
        non_mandarin = text[curr_pos:].replace('{', '').replace('}', '')
        non_mandarin_words = non_mandarin.split()
        num_syllables_used.extend([max(1, len(w)//2) for w in non_mandarin_words])
        transcript.extend(non_mandarin_words)

    return transcript, num_syllables_used


def _parse_stm_line(l):
    '''
        Parse a line of the stm file. The format is 

        audiopath channel speaker start end label text

        In the event that there is not text (it is empty for the segment), then
        we catch that case. 
    '''
    try:
        path, channel, speaker, start, end, lbl, text = l.strip().split(None, 6)
    except ValueError:
        path, channel, speaker, start, end, _ = l.strip().split(None, 5)
        text = ''
    recoid = path
    text, _ = _reformat_mixed_mandarin_text(text)
    text = ' '.join(text)
    return {
        'recoid': recoid,
        'channel': channel,
        'speaker': speaker,
        'start': float(start),
        'end': float(end),
        'text': text,
        'label': lbl,
    }


def _segment_to_uniform_ctm(seg):
    '''
        Convert an stm segment into a ctm. We make a special case when it comes
        to mandarin characters.
    '''
    text, channel, recoid = seg['text'], seg['channel'], seg['recoid']
    transcript, num_syllables_used = _reformat_mixed_mandarin_text(text)
    
    duration_per_syllable = (seg['end'] - seg['start']) / sum(num_syllables_used) 
    start = seg['start']
    for w, num_syllables in zip(transcript, num_syllables_used):
        dur = num_syllables * duration_per_syllable 
        yield [recoid, channel, start, dur, w]
        start += dur


def main(args):
    # Make the output scoring directory
    odir = Path(args.odir)
    odir.mkdir(mode=511, parents=True, exist_ok=True)
    recos = {}
    # Read the hypothesis stm file and convert it to ctm for sclite scoring.
    # The ctm is fake in the sense that the time markings are all fake and come
    # from the predefined, reference segmentation.
    with open(args.stm_hyp, 'r', encoding='utf-8') as fp:
        for l in fp:
            seg = _parse_stm_line(l)
            for ctm in _segment_to_uniform_ctm(seg):
                recoid = seg['recoid']
                if recoid not in recos:
                    recos[recoid] = []
                recos[recoid].append(ctm)
   
    stm_recos = {}
    # Read in the reference stm file.
    with open(args.stm_ref, 'r', encoding='utf-8') as fp1, open(odir / 'ref.stm', 'w', encoding='utf-8') as fp2:
        for l in fp1:
            seg = _parse_stm_line(l)
            recoid = seg['recoid']
            if recoid not in stm_recos:
                stm_recos[recoid] = []
            stm_recos[recoid].append(seg)
        
        # Sort the reference stm file and apply the mixed mandarin text
        # normalization scheme to enable scoring with mixed error rate.
        # The mixed error rate is the same as WER where each mandarin character
        # is treated as a separate word.
        for recoid in sorted(stm_recos.keys()):
            for seg in sorted(stm_recos[recoid], key=lambda x: (x['channel'], x['start'])):
                r, c, s = seg['recoid'], seg['channel'], seg['speaker']
                b, e, l, t = seg['start'], seg['end'], seg['label'], seg['text']
                print(f'{r} {c} {s} {b:0.2f} {e:0.2f} {l} {t}', file=fp2)

    with open(odir / 'hyp.ctm', 'w', encoding='utf-8') as fp:
        for recoid in sorted(recos.keys()):
            for ctm in sorted(recos[recoid], key=lambda x: (x[1], x[2])):
                print(f'{ctm[0]} {ctm[1]} {ctm[2]:0.2f} {ctm[3]:0.2f} {ctm[4]} 1.0', file=fp) 
    
    cmd = [args.sclite] + f"-r {odir / 'ref.stm'} stm -h {odir / 'hyp.ctm'} ctm -e utf-8 -D -c NOASCII DH -O {args.odir} -o all".split()
    status = check_call(cmd, shell=False, stderr=DEVNULL, stdout=DEVNULL)
     
             
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        
        Converts an stm to a ctm by assigning fake time labels to each word in
        a segment by distributing the words uniformly over time:

        ./utils/stm_wer.py ~/kaldi/tools/sckt/bin/sclite ref.stm hyp.stm scoring_dir 
        """
    )
    parser.add_argument("sclite", help='path to sclite binary')
    parser.add_argument("stm_ref", help='the reference stm file')
    parser.add_argument("stm_hyp", help='the hypothesis stm file')
    parser.add_argument("odir", help='the output directory to be created where'
        ' all scoring outputs are dumped.'
    )
    args = parser.parse_args()
    main(args)
