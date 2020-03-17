#!/usr/bin/env python3
"""
Clean each capnp recording of any unfinished writing as it will crash otherwise
"""
import argparse
from multiprocessing import Process
from pathlib import Path
from shutil import copyfile
import derp.util


def clean(topic, path, tmp_path):
    """ Cleaning is simply the act of writing only complete messages """
    copyfile(path, tmp_path)
    with open(tmp_path, 'rb') as topic_reader, open(path, 'wb') as topic_writer:
        for msg in derp.util.TOPICS[topic].read_multiple(topic_reader):
            msg.as_builder().write(topic_writer)
            topic_writer.flush()


def main():
    """ Create a process for each cleaning """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=Path, help="location of a recording")
    args = parser.parse_args()

    for topic in derp.util.TOPICS:
        path = args.path / (topic + '.bin')
        tmp_path = args.path / (topic + '.bin.bak')
        if path.exists():
            proc = Process(target=clean, name=topic, args=(topic, path, tmp_path))
            proc.start()
            proc.join()
        tmp_path.unlink()


if __name__ == "__main__":
    main()
