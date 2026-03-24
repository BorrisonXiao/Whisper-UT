import argparse


from data_prep.stm import Stm

def _get_parser():
    """
    Helper function to define the argument parser
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("input_stm", help="STM to convert")
    parser.add_argument("output_dir", help="Directory to store new files")
    parser.add_argument("--stereo", action="store_true", help="Use two channels")
    parser.add_argument("--ignore-segments", action="store_true", help="Do not use times for utt_id")
    parser.add_argument("--fs", default=8000, help="Frame Sample Rate")
    parser.add_argument("--min-duration", default=0.0, help="Minimum duration of the utterance")
    parser.add_argument("--start-at-zero", action="store_true", help="Channel start at 0 instead of 1")

    return parser


def stm_to_espnet(
        input_stm,
        output_dir,
        use_stereo,
        ignore_segments,
        min_dur=0.0,
        fs=8000,
        start_at_zero=False):
    """
        Helper function to convert STM to ESPnet format

        Parameters:
            input_stm - STM to convert
            output_dir - Directory to store ESPnet formatted data
            use_stereo - Use 2 channels
            ignore_segments - Do not write segments file
            min_dur - minimum duration for the utterances
            fs - sample rate
            start_at_zero - STM channels start at zero instead of 1
    """
    with open(input_stm, 'r', encoding='utf-8') as fh:
        stm = Stm.parse(fh, min_dur=min_dur, start_at_zero=start_at_zero)
    stm.write_espnet_dir(output_dir, stereo=use_stereo, ignore_segments=ignore_segments, fs=fs)


def main(args):
    input_stm = args.input_stm
    output_dir = args.output_dir
    use_stereo = args.stereo
    ignore_segments = args.ignore_segments
    min_dur = float(args.min_duration)
    fs = int(args.fs)
    start_at_zero = args.start_at_zero

    stm_to_espnet(input_stm, output_dir, use_stereo, ignore_segments, min_dur=min_dur, fs=fs, start_at_zero=start_at_zero)


if __name__ == "__main__":
    args = _get_parser().parse_args()

    main(args)

