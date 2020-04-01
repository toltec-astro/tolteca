#! /usr/bin/env python


# Author:
#   Zhiyuan Ma

"""This recipe takes a tune file, and use the frequencies of the tones
to generate a wav file.

"""
from tolteca.io.toltec import KidsModelParams
from tollan.utils.log import get_logger, init_log
import numpy as np
import math
from pyaudio import PyAudio, paUInt8


def generate_sine_wave(frequency, duration, volume=0.2, sample_rate=44100):
    ''' Generate a tone at the given frequency.

        Limited to unsigned 8-bit samples at a given sample_rate.
        The sample rate should be at least double the frequency.
    '''
    if sample_rate < (frequency * 2):
        print('Warning: sample_rate must be at least double the frequency '
              f'to accurately represent it:\n    sample_rate {sample_rate}'
              f' ≯ {frequency*2} (frequency {frequency}*2)')

    num_samples = int(sample_rate * duration)
    rest_frames = num_samples % sample_rate

    pa = PyAudio()
    stream = pa.open(
        format=paUInt8,
        channels=1,  # mono
        rate=sample_rate,
        output=True,
    )

    # make samples
    def s(i):
        return volume * math.sin(
            2 * math.pi * frequency * i / sample_rate)
    samples = (int(
        s(i) * 0x7F + 0x80) for i in range(num_samples))

    # write several samples at a time
    for buf in zip(*([samples] * sample_rate)):
        stream.write(bytes(buf))

    # fill remainder of frameset with silence
    stream.write(b'\x80' * rest_frames)

    stream.stop_stream()
    stream.close()
    pa.terminate()


def main(tunefile, wavfile=None):
    logger = get_logger()
    tune = KidsModelParams(tunefile)
    logger.debug(f'tune model: {tune.model}')
    f_a4 = 440
    # a #a b c #c d #d e f #f g #g
    # 0 1  2 3 4  5 6  7 8 9 10 11
    natural_am_scale = (2 ** np.linspace(0, 1, 12)) * f_a4
    # c d e g a
    penta_c_maj_scale = natural_am_scale[[3, 5, 7, 10, 0]]
    scale = penta_c_maj_scale
    logger.debug(f'scale: {scale}')

    notes = (np.tile(
        tune.model.fr.value, (len(scale), 1)).T % scale).T
    notes = np.argmin(notes, axis=0)
    logger.debug(f'notes: {notes}')
    bps = 80 / 60.
    for n in notes:
        generate_sine_wave(
            frequency=scale[n],  # Hz, waves per second C6
            duration=1. / bps,  # seconds to play sound
            volume=0.25,  # 0..1 how loud it is
            sample_rate=44100
        )


if __name__ == "__main__":
    init_log(level='DEBUG')

    import argparse
    parser = argparse.ArgumentParser(
            description='Generate a tune from tune file.')
    parser.add_argument("tunefile", help='The tune file to use.')
    parser.add_argument(
            "--output", '-o', help='output file. If not set, play.')
    args = parser.parse_args()
    main(args.tunefile, wavfile=args.output)
