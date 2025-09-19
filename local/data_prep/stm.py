# -*- coding: utf-8 -*-
import random
import logging
import os
import re
import locale
import shutil
import subprocess
import tempfile

from functools import cmp_to_key
from collections import defaultdict


logger = logging.getLogger(__name__)


PAREN_RE = re.compile(r'\((\S+)\)')


class StmUtterance(object):
    def __init__(self, filename, channel, speaker, start_time, stop_time, transcript, label='<O>'):
        self.filename = filename
        self.channel = channel
        self._speaker = speaker
        self.start_time = start_time
        self.stop_time = stop_time
        self.transcript = transcript
        self.label = label

    def key(self):
        return (self.filename, self.channel, self.speaker, self.start_time, self.stop_time, self.transcript, self.label)

    def __eq__(self, o):
        if isinstance(o, StmUtterance):
            return self.key() == o.key()
        return NotImplemented

    def __hash__(self):
        return hash(self.key())

    def __str__(self):
        return '%s %s %s %.2f %.2f %s %s' % (self.filename, self.channel, self.speaker, self.start_time, self.stop_time, self.label, self.transcript)

    def __repr__(self):
        return str(self)

    def recording_id(self, stereo=False):
        return get_recording_id(self.filename, self.channel, stereo)

    def _split_rec_id(self, stereo=False):
        rec_id = self.recording_id(stereo=False)
        suffixes = ['.L', '.R', '-A', '-B']
        for s in suffixes:
            if rec_id.endswith(s):
                rec_id = rec_id[:-len(s)]
                break
        return rec_id

    @property
    def speaker(self):
        #if self._short_speaker:
        # For now always return speaker as is
        if True:
            return self._speaker
        return self._speaker.rjust(36, '0')

    @property
    def segment_str(self):
        return '{0:d}-{1:d}'.format(int(self.start_time * 1000), int(self.stop_time * 1000))

    @property
    def dur(self):
        return self.stop_time - self.start_time

    @property
    def words(self):
        return self.transcript.strip().split()

    @property
    def audioext(self):
        return os.path.splitext(self.filename)[1].lower()

    @property
    def audio_format(self):
        return {'.ul': 'MULAW',
                '.lu': 'RMULAW',
                '.mulaw': 'RULAW',
                '.wav': 'WAVE',
                '.mp3': 'MP3',
                '.flac': 'FLAC',
               }.get(self.audioext, None)

    def words_per_second(self, precision = 1):
        return round(1.0 * len(self.words) / self.dur, precision)

    def utterance_id(self, stereo=False, ignore_segments=False):
        rec_id = self.recording_id(stereo=stereo)
        # skip spkr prefix if ist's already present in recording_id
        if rec_id.startswith('%s-' % self.speaker):
            utt_id = '%s_%08.0f_%08.0f' % (rec_id, self.start_time*100, self.stop_time*100)
        else:
            utt_id = '%s-%s_%08.0f_%08.0f' % (self.speaker, rec_id, self.start_time*100, self.stop_time*100)
            if ignore_segments:
                utt_id = self.speaker
        if ignore_segments:
            return rec_id
        return utt_id

def pad_stm(utts, pad_duration, min_sil_duration_after_padding=0.05):
    """
    Add padding to start/end of utterances, assuming input s from a single
    recording_id and sorted by start_time
    """
    if pad_duration == 0.0:
        s = Stm([])
        s._utts = utts
        return s

    recording_groups = defaultdict(list)
    for u in utts:
        recording_groups[(u.filename, u.channel)].append(u)

    # pad each recording
    stm_padded = []
    for stm in recording_groups.values():
        stm_sorted = sorted(stm, key=lambda s: s.start_time)

        # pad first utterance, with min duration of 0
        start_times_padded = [max(stm_sorted[0].start_time - pad_duration, 0.0)]
        end_times_padded = []

        for i in range(1, len(stm_sorted)):
            # measure silence between utterances
            sil = stm_sorted[i].start_time - stm_sorted[i-1].stop_time
            if sil < (pad_duration * 2.0):
                # split silence and perserve at least 0.05 silence
                p = sil * 0.5 - min_sil_duration_after_padding
                if p <= 0:
                    logger.debug('skipping padding in {} for silence between %f and %f'.format(stm_sorted[i].filename, stm_sorted[i-1].stop_time, stm_sorted[i].start_time))
                    end_times_padded.append(stm_sorted[i-1].stop_time)
                    start_times_padded.append(stm_sorted[i].start_time)
                else:
                    end_times_padded.append(stm_sorted[i-1].stop_time + p)
                    start_times_padded.append(stm_sorted[i].start_time - p)
            else:
                end_times_padded.append(stm_sorted[i-1].stop_time + pad_duration)
                start_times_padded.append(stm_sorted[i].start_time - pad_duration)
        # skip last utterance
        end_times_padded.append(stm_sorted[-1].stop_time)

        for i, s in enumerate(stm_sorted):
            stm_padded.append(StmUtterance(
                s.filename,
                s.channel,
                s.speaker,
                start_times_padded[i],
                end_times_padded[i],
                s.transcript,
                label=s.label))
    final_stm = Stm([])
    final_stm._utts = stm_padded
    return final_stm


def get_recording_id(audio_filename, channel, stereo, audiokey_prefix=None):
    if audio_filename.endswith('.R') or audio_filename.endswith('.L'):
        recording_id = os.path.basename(audio_filename)
    else:
        recording_id, _ = os.path.splitext(os.path.basename(audio_filename))
        if stereo:
            recording_id = '%s-%s' % (recording_id, channel)

    if audiokey_prefix is not None:
        recording_id = '%s-%s' % (audiokey_prefix, recording_id)

    return recording_id


def parse_StmUtterance(line, min_dur=0.0, start_at_zero=False):
    fields = line.strip().split()
    if len(fields) < 6:
        raise Exception('too few fields in stm line "{}"'.format(line))
    label = '<O>'
    trans_field = 5
    if fields[5].startswith('<') and fields[5].endswith('>'):
        label = fields[5]
        trans_field = 6
    filename = fields[0]
    channel = _parse_stm_line_channel(fields[1], start_at_zero=start_at_zero)
    speaker = fields[2]
    start_time = float(fields[3])
    stop_time = float(fields[4])

    if start_time > stop_time or start_time == stop_time:
        raise Exception('bad times in "{}"'.format(line))

    if abs(start_time - stop_time) < min_dur:
        raise Exception('utterance too short "{}"'.format(line))

    transcript = ' '.join(fields[trans_field:])
    return StmUtterance(
            filename,
            channel,
            speaker,
            start_time,
            stop_time,
            transcript,
            label=label)


def unshorten(input_filename, output_filename):
    if shutil.which("sph2pipe") is None:
        raise Exception('sph2pipe not in PATH')

    if shutil.which("sox") is None:
        raise Exception('sox not in PATH')

    # convert both channels to wav
    mono_filenames = []
    try:
        for c in [1, 2]:
            cmd = 'sph2pipe -f wav -p -c %d %s' % (c, input_filename)
            convert_process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE)
            stdout, stderr = convert_process.communicate()

            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp:
                temp.write(stdout)
                mono_filenames.append(temp.name)

        # combine mono channels into stereo wav
        cmd = 'sox -M %s %s %s' % (mono_filenames[0], mono_filenames[1], output_filename)
        merge_process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
        stdout, stderr = merge_process.communicate()
    finally:
        for f in mono_filenames:
            if os.path.isfile(f):
                os.remove(f)


def _parse_stm_line_channel(c, start_at_zero=False):
    _c = c.lower()

    if _c == 'a' or (not start_at_zero and _c == '1') or (start_at_zero and _c == '0'):
        return 'A'
    elif _c == 'b' or (not start_at_zero and _c == '2') or (start_at_zero and _c == '1'):
        return 'B'
    raise Exception('Failed to parse channel string \'{}\''.format(c))


class Stm(object):
    """Representation of STM file

    Different from the spec, the first column containing the filename should
    contain the full path to the audio file and any file extension.
    """
    def __init__(self, utterances, comments=None):
        """
        Create new STM from a stream of utterances

        Args:
            utterances: an iterable of StmUtterance objects
        """
        self._utts = []
        self._comments = []
        if comments is not None:
            for c in comments:
                self._comments.append(c)
        self._seconds = 0.0
        seen_ids = set()
        for u in utterances:
            if not isinstance(u, StmUtterance):
                self._comments.append(u)
            elif u not in seen_ids:
                seen_ids.add(u)
                self._utts.append(u)
                self._seconds += u.stop_time - u.start_time

        del seen_ids

    def __eq__(self, other):
        if isinstance(other, Stm):
            return self._utts == other._utts and self._comments == other._comments
        return False

    def __str__(self):
        return '\n'.join([str(u) for u in self._utts] + [str(c) for c in self._comments])

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self._utts)

    def __iter__(self):
        yield from self._utts

    def remove(self, utt):
        """
        Remove utterance from stm object and decreases the total number of seconds
        """
        try:
            self._utts.remove(utt)
            self._seconds -= utt.dur
        except:
            logger.warning("Utterance does not exists")

    def remove_utts_by_keys(self, utt_keys):
        new_utt_list = []
        new_duration = 0.0
        for utt in self._utts:
            if utt.key() not in utt_keys:
                new_duration += utt.dur
                new_utt_list.append(utt)
        self._seconds = new_duration
        self._utts = new_utt_list


    def simple_audio_groups(self):
        audio_utts = defaultdict(list)
        for u in self:
            audio_utts[u.filename].append(u)
        audio_groups = []
        for audio, utts in sorted(audio_utts.items(), key=lambda x: x[0]):
            s = Stm([])
            s._comments = self._comments
            s._utts = utts
            audio_groups.append(s)
        return audio_groups


    def log_wps_metrics(self, metrics_out_file = None):
        words_per_second_buckets = defaultdict(int)
        precision = 1
        for stm_utterance in self:
            words_per_second_buckets[stm_utterance.words_per_second(precision)] += 1

        out_handle = open(metrics_out_file,"w") if metrics_out_file else None

        for bucket in sorted(words_per_second_buckets.keys()):
            line = "%s\t%s" % (bucket, words_per_second_buckets[bucket])
            if out_handle:
                out_handle.write("%s\n" % line)
            else:
                logger.warn(line)

        if out_handle:
            out_handle.close()


    @property
    def seconds(self):
        if self._seconds == 0.0:
            secs = 0.0
            for u in self:
                secs += u.dur
            self._seconds = secs
        return self._seconds

    @property
    def hours(self):
        return self.seconds / 3600.0

    @property
    def comments(self):
        return self._comments

    @staticmethod
    def _group_by_audio_id(utts, determine_audio_id=lambda u: u.filename):
        groups = []
        audio_id = None
        for u in utts:
            curr_audio_id = determine_audio_id(u)
            if audio_id is None or audio_id != curr_audio_id:
                groups.append([])
            groups[-1].append(u)
            audio_id = curr_audio_id
        return groups

    @staticmethod
    def parse(fh, min_dur=0.0, start_at_zero=False):
        return Stm(Stm._parse(fh, min_dur=min_dur, start_at_zero=start_at_zero))

    @staticmethod
    def _parse(fh, min_dur=0.0, start_at_zero=False):
        for _line in fh:
            line = _line.strip()
            if len(line) < 1:
                continue
            if line.startswith(';;'):
                yield line
                continue
            try:
                yield parse_StmUtterance(line, min_dur=min_dur, start_at_zero=start_at_zero)
            except Exception as e:
                logger.warning('failed to parse stm line: {} - {}'.format(line.strip(), str(e)))

    def write(self, fh):
        for c in self._comments:
            fh.write(str(c) + '\n')
        for u in self._utts:
            fh.write(str(u) + '\n')

    def pad(self, pad_duration, min_sil_duration_after_padding=0.05):
        """
        Add padding to start/end of utterances, assuming input s from a single
        recording_id and sorted by start_time
        """
        padded = pad_stm(self._utts, pad_duration, min_sil_duration_after_padding=min_sil_duration_after_padding)
        padded._comments = self._comments
        return padded


    def convert_audio(self, audio_dir):
        """
        convert stereo sphere files with shorten compression to stereo
        wave files and update stm to reflect path to wave
        """

        if not os.path.isdir(audio_dir):
            os.makedirs(audio_dir)

        converted_utts = []
        converted_audios = set()
        for u in self:
            wav_filename = os.path.join(audio_dir,
                                        os.path.splitext(os.path.basename(u.filename))[0] + '.wav')
            if wav_filename not in converted_audios:
                # check to see if audio already exists
                if not os.path.isfile(wav_filename):
                    logger.info('Converting %s to %s without shorten compression' %
                                (u.filename, wav_filename))
                    unshorten(u.filename, wav_filename)
                converted_audios.add(wav_filename)
            u.filename = wav_filename

            converted_utts.append(u)

        s = Stm([])
        s._utts = converted_utts
        s._comments = self._comments

        return s

    def remove_overlap_utts(self):
        """
        remove utterances that have same start/stop time
        """

        counts = defaultdict(int)
        for u in self:
            utt_id = (u.filename, u.start_time, u.stop_time, u.channel)
            counts[utt_id] += 1

        # second pass to remove overlapping utterances
        filtered_utts = []
        for u in self:
            utt_id = (u.filename, u.start_time, u.stop_time, u.channel)
            if counts[utt_id] > 1:
                logger.debug('Removing overlapping utterances %s %s %f %f' %
                             (u.filename, u.channel, u.start_time, u.stop_time))
            else:
                filtered_utts.append(u)

        s = Stm([])
        s._utts = filtered_utts
        s._comments = self._comments

        return s


    def espnet_lists(self, stereo=False, ignore_segments=False, fs=8000):
        """
        Creates wav_scp, segments (optional), text, and spk2utt list/dicts
        """
        wav_scp = set()
        segments = set()
        text = []
        reco2channel = set()
        spk2utt = defaultdict(set)

        for s in self._utts:

            audioext = os.path.splitext(s.filename)[1].lower()
            if audioext == '.bz2':
                convert_cmd = 'ivec/bz2_to_wav.py %s %d |' % (s.filename, fs)
            elif audioext == '.mulaw':
                convert_cmd = 'sox -r %.0f -c 1 -t ul %s -b 16 -t .wav - |' % (fs, s.filename)
            elif audioext == '.mu':
                convert_cmd = 'sox -r %.0f -c 1 -t lu %s -b 16 -t .wav - |' % (fs, s.filename)
            elif audioext == '.alaw' or audioext == 'al' or audioext == '.a8':
                convert_cmd = 'sox -r %.0f -c 1 -t al %s -b 16 -t .wav - |' % (fs, s.filename)
            elif audioext == '.wav':
                if stereo:
                    if s.channel == 'A':
                        channel = 1
                    else:
                        channel = 2
                    convert_cmd = 'sox %s -r %.0f -b 16 -t .wav - remix %s |' % (s.filename, fs, channel)
                else:
                    convert_cmd = 'sox %s -r %.0f -b 16 -t .wav - |' % (s.filename, fs)
            elif audioext == '.flac':
                if stereo:
                    if s.channel == 'A':
                        channel = 1
                    else:
                        channel = 2
                    convert_cmd = 'sox -t flac %s -t .wav -r %d -b 16 - remix %s |' % (s.filename, fs, channel)
                else:
                    convert_cmd = 'sox -t flac %s -t .wav -r %d -b 16 - |' % (s.filename, fs)
            elif audioext == '.mp3':
                if stereo:
                    if s.channel == 'A':
                        channel = 1
                    else:
                        channel = 2
                    convert_cmd = 'ffmpeg -i %s -ar %.0f -ac %d -f wav - |' % (s.filename, fs, channel)
                else:
                    convert_cmd = 'ffmpeg -i %s -ar %.0f -f wav - |' % (s.filename, fs)
            elif audioext == '.sph':
                if s.channel == 'A':
                    channel = 1
                else:
                    channel = 2
                convert_cmd = 'sph2pipe -f wav -p -c %d %s |' % (channel, s.filename)
            elif audioext == '.wavpcm':
                convert_cmd = 'sox -c 1 -t wavpcm -e signed-integer %s -t wavpcm - |' % (s.filename)
            elif audioext == ".m4a":
                if stereo:
                    if s.channel == 'A':
                        channel = 1
                    else:
                        channel = 2
                    convert_cmd = 'ffmpeg -i %s -ar %.0f -ac %d -f wav - |' % (s.filename, fs, channel)
                else:
                    convert_cmd = 'ffmpeg -i %s -ar %.0f -f wav - |' % (s.filename, fs)
            else:
                logger.warning('unknown audioext: %s, skipping' % s.filename)
                continue

            rec_id = s.recording_id(stereo=stereo)
            utt_id = s.utterance_id(stereo=stereo, ignore_segments=ignore_segments)
            spk2utt[s.speaker].add(utt_id)
            if not ignore_segments:
                segments.add('%s %s %.2f %.2f' % (utt_id, rec_id, s.start_time, s.stop_time))

            wav_scp.add('%s %s' % (rec_id, convert_cmd))

            words = PAREN_RE.sub('\1', s.transcript)
            if stereo:
                reco2channel.add('%s %s %s' % (rec_id, os.path.splitext(os.path.basename(s.filename))[0], s.channel))

            text.append('%s %s' % (utt_id, words))

        return text, wav_scp, spk2utt, reco2channel, segments


    def flatten(self):
        grouped_lines = defaultdict(list)
        for s in self._utts:
            grouped_lines[s.filename + '-' + s.channel].append(s)

        stm_utts = []
        for g in grouped_lines.values():
            words = []
            sorted_g = sorted(g, key=lambda t: t.start_time)
            for utt in sorted_g:
                words.append(utt.transcript)

            stm_utts.append(StmUtterance(
                g[0].filename,
                g[0].channel,
                g[0].speaker,
                sorted_g[0].start_time,
                sorted_g[-1].stop_time,
                ' '.join(words),
                label=g[0].label))
        final_stm = Stm([])
        final_stm._utts = stm_utts
        final_stm._comments = self._comments
        return final_stm


    @staticmethod
    def _write_sorted_list(fh, lst):
        for line in sorted(lst, key=cmp_to_key(locale.strcoll)):
            fh.write(line + '\n')


    def write_espnet_dir(self, data_dir, stereo=False, ignore_segments=False, fs=8000):
        """
        Creates an espnet datadir for this stm
        """
        # text (uttid <transcription>)
        # wav.scp (uttid path) (use recid if segments)
        # utt2spk (uttid speakerid) (use uttid if no speaker)
        # segments (uttid recid start end)

        text, wav_scp, spk2utt, reco2channel, segments = self.espnet_lists(stereo=stereo, ignore_segments=ignore_segments, fs=fs)

        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        locale.setlocale(locale.LC_ALL, "C")

        # sort
        spk2utt_keys = sorted(list(spk2utt.keys()), key=cmp_to_key(locale.strcoll))

        utt2spk = {}
        for spk, utts in spk2utt.items():
            for utt in utts:
                utt2spk[utt] = spk

        utt2spk_keys = sorted(list(utt2spk.keys()), key=cmp_to_key(locale.strcoll))

        logger.info('read %d utterances, writing to %s' % (len(utt2spk), data_dir))
        with open(os.path.join(data_dir, 'wav.scp'), 'w', encoding='utf-8') as fh:
            Stm._write_sorted_list(fh, wav_scp)

        with open(os.path.join(data_dir, 'spk2utt'), 'w', encoding='utf-8') as fh:
            for spk in spk2utt_keys:
                fh.write(spk + ' ')
                for utt in sorted(spk2utt[spk], key=cmp_to_key(locale.strcoll)):
                    fh.write(utt + ' ')
                fh.write('\n')

        with open(os.path.join(data_dir, 'utt2spk'), 'w', encoding='utf-8') as fh:
            for utt in utt2spk_keys:
                fh.write('%s %s\n' % (utt, utt2spk[utt]))

        if not ignore_segments:
            with open(os.path.join(data_dir, 'segments'), 'w', encoding='utf-8') as fh:
                Stm._write_sorted_list(fh, segments)

        if stereo:
            with open(os.path.join(data_dir, 'reco2file_and_channel'), 'w', encoding='utf-8') as fh:
                Stm._write_sorted_list(fh, reco2channel)

        with open(os.path.join(data_dir, 'text'), 'w', encoding='utf-8') as fh:
            Stm._write_sorted_list(fh, text)

        with open(os.path.join(data_dir, 'stm'), 'w', encoding='utf-8') as fh:
            self.write(fh)


    def filter(self, min_utt_dur=0.01, max_utt_dur=300.):
        s = Stm([])
        utts = [u for u in self._utts if u.dur >= min_utt_dur and u.dur <= max_utt_dur]
        s._utts = utts
        s._comments = self._comments
        return s
