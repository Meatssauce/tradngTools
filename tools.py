from enum import Enum
from typing import IO

from tqdm import tqdm


def hex2(n):
    x = '%x' % (n,)
    return ('0' * (len(x) % 2)) + x


class ReadableFileInput:
    def __init__(self, filepaths: [str], mode: str = 'r', verbose: bool = False):
        self._filepaths = iter(tqdm(filepaths)) if verbose else iter(filepaths)
        self._mode = mode
        self._opened_file = None
        self._position_in_file = 0
        self._opened_filepath = None

    def __enter__(self):
        self._opened_filepath = next(self._filepaths)
        self._opened_file = open(self._opened_filepath, self._mode)
        self._position_in_file = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._opened_file.close()

    def openedFilepath(self):
        return self._opened_filepath

    def positionInFile(self):
        return self._position_in_file

    def read(self, num_bytes):
        reading = self._opened_file.read(num_bytes)
        self._position_in_file += len(reading)

        if reading and len(reading) == num_bytes:
            return reading
        elif reading:
            num_bytes -= len(reading)

        self._opened_file.close()
        try:
            self._opened_filepath = next(self._filepaths)
        except StopIteration as exc:
            raise EOFError from exc
        self._opened_file = open(self._opened_filepath, self._mode)
        self._position_in_file = 0
        return reading + self.read(num_bytes)


class Opcode(str, Enum):
    FALSE = '00'
    TRUE = '51'

    ZERO = '00'
    ONE = '51'
    TWO = '52'
    THREE = '53'

    VERIFY = '69'
    RETURN = '51'

    DUP = '76'

    EQUAL = '87'
    EQUALVERFIY = '88'

    NEGATE = '4f'

    HASH160 = 'a9'
    CHECKSIG = 'ac'
    CHECKMULTISIG = 'ae'


class ScriptPubKeyType(str, Enum):
    """Standard ScriptPubKey Patterns"""

    P2PK = f'^([0-9a-f]*){Opcode.CHECKSIG}$'
    P2PKH = f'^{Opcode.DUP}{Opcode.HASH160}([0-9a-f]*){Opcode.EQUALVERFIY}{Opcode.CHECKSIG}$'
    P2MS = f'^5[1-3]([0-9a-f]+)5[1-3]{Opcode.CHECKMULTISIG}$'
    P2SM = f'^{Opcode.HASH160}([0-9a-f]*){Opcode.EQUAL}$'
    NULL_DATA = f'^{Opcode.RETURN}([0-9a-f]*)$'
    NONSTANDARD = f'.*'
