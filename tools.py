from typing import IO

from tqdm import tqdm


def hex2(n):
    x = '%x' % (n,)
    return ('0' * (len(x) % 2)) + x


class ReadableFileInput:
    def __init__(self, filepaths, mode='r', verbose=False):
        self.filepaths = iter(tqdm(filepaths)) if verbose else iter(filepaths)
        self.mode = mode
        self.opened_file = None

    def __enter__(self):
        self.opened_file = open(next(self.filepaths), self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.opened_file.close()

    def read(self, num_bytes):
        reading = self.opened_file.read(num_bytes)

        if reading:
            return reading

        self.opened_file.close()
        try:
            self.opened_file = open(next(self.filepaths), self.mode)
        except StopIteration as exc:
            raise EOFError from exc
        return self.read(num_bytes)
