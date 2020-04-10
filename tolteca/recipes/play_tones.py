#! /usr/bin/env python


# Author:
#   Zhiyuan Ma
# History:
#   2020/04/10 Zhiyuan Ma:
#       - First staged.

"""This recipe takes a tune file, and use the frequencies of the tones
to generate a wav file.

Additional required dependencies:

    $ pip install midi2audio MIDIUtil

`Fluidsynth` is required to run midi2audio. On macOS, it can be
installed via homebrew:

    $ brew install fluidsynth

At least one sound font needs to be present to be able to render
the MIDI, which can be found on
https://github.com/FluidSynth/fluidsynth/wiki/SoundFont

To install the sound font, follow the doc of `midi2audio` at

https://github.com/bzamecnik/midi2audio

To play the sounds in-line, one also need to install:

    pip install playsound
    # on macOS, also do:
    pip install pyobjc

"""
from tolteca.io.toltec import KidsModelParams
from tollan.utils.log import get_logger, init_log
import numpy as np
from midiutil import MIDIFile
from midi2audio import FluidSynth
import tempfile


def main(tunefile, wavfile=None):

    logger = get_logger()
    tune = KidsModelParams(tunefile)
    logger.debug(f'tune model: {tune.model}')

    f_a4 = 440
    # a #a b c #c d #d e f #f g #g
    # 0 1  2 3 4  5 6  7 8 9 10 11
    #                      A   B   C   D    E  F   G   A
    natural_am_midi = np.arange(57, 57 + 13)
    natural_am_scale = (2 ** np.linspace(0, 1, 13)) * f_a4

    # c d e g a
    # am_to_penta_c_maj
    trans = [3, 5, 7, 10, 12]

    # d e g a b
    # am_to_yo
    # trans = [5, 7, 10, 12, 2]

    midi_scale = natural_am_midi[trans]
    scale = natural_am_scale[trans]

    logger.debug(f'midi scale: {midi_scale}')
    logger.debug(f'scale: {scale}')

    notes = (np.tile(
        tune.model.fr.value, (len(scale), 1)).T % scale).T
    notes = np.argmin(notes, axis=0)
    logger.debug(f'notes:\n{notes}')

    midi_notes = midi_scale[notes]
    # make midi
    track = 0
    channel = 0
    time = 0   # In beats
    duration = 0.5   # In beats
    tempo = 120  # In BPM
    volume = 100  # 0-127, as per the MIDI standard

    # One track, defaults to format 1 (tempo track
    # automatically created)
    MyMIDI = MIDIFile(1)
    MyMIDI.addTempo(track, time, tempo)

    for note in midi_notes:
        MyMIDI.addNote(track, channel, note, time, duration, volume)
        time = time + 0.25

    with tempfile.NamedTemporaryFile() as f:
        MyMIDI.writeFile(f)

        def to_wav(wavfile):
            FluidSynth().midi_to_audio(f.name, wavfile)

        if wavfile is None:
            from playsound import playsound
            with tempfile.NamedTemporaryFile() as g:
                to_wav(g.name)
                playsound(g.name)
        else:
            to_wav(wavfile)


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
