from hashlib import sha256
import re
from dataclasses import dataclass
from typing import IO

from opcodes import Opcode


def read_varint(file: IO) -> int:
    tx_count = file.read(1).hex()

    if tx_count.startswith('fd'):
        tx_count = file.read(2)[::-1].hex()
    elif tx_count.startswith('fe'):
        tx_count = file.read(4)[::-1].hex()
    elif tx_count.startswith('ff'):
        tx_count = file.read(8)[::-1].hex()

    return int(tx_count, base=16)


# def read_varint_bytes(stream: bytes) -> tuple[int, int]:
#     tx_count = file.read(1).hex()
#     total_bytes_read = 1
#
#     if tx_count.startswith('fd'):
#         tx_count = file.read(2)[::-1].hex()
#         total_bytes_read += 2
#     elif tx_count.startswith('fe'):
#         tx_count = file.read(4)[::-1].hex()
#         total_bytes_read += 4
#     elif tx_count.startswith('ff'):
#         tx_count = file.read(8)[::-1].hex()
#         total_bytes_read += 8
#
#     return int(tx_count, base=16), total_bytes_read


@dataclass(frozen=True)
class Input:
    tx_id: str  # that of a tx whose output we take as input
    vout: str  # index of input as an output in the tx
    scriptSig_size: int  # number of bytes
    scriptSig: str
    sequence: str  # ignored

    @classmethod
    def from_file(cls, file: IO):
        tx_id = file.read(32)[::-1].hex()
        vout = file.read(4)[::-1].hex()
        scriptSig_size = read_varint(file)
        scriptSig = file.read(scriptSig_size).hex()
        sequence = file.read(4)[::-1].hex()
        return cls(tx_id, vout, scriptSig_size, scriptSig, sequence)

    # @classmethod
    # def from_bytes(cls, stream: bytes) -> 'Input':
    #     tx_id = stream[:32][::-1].hex()
    #     stream = stream[32:]
    #
    #     vout = stream[:4][::-1].hex()
    #     stream = stream[4:]
    #
    #     scriptSig_size, bytes_read = read_varint_bytes(stream)
    #     stream = stream[bytes_read:]
    #
    #     scriptSig = stream[:scriptSig_size].hex()
    #     stream = stream[scriptSig_size:]
    #
    #     sequence = stream[:4][::-1].hex()
    #     stream = stream[4:]
    #
    #     return cls(tx_id, vout, scriptSig_size, scriptSig, sequence)

    def is_coinbase(self):
        return self.tx_id == '0' * 64 and self.vout == 'f' * 8


def hex2Base58(payload: str):
    """Converts a hex string into a base58 string

    :param payload: the hexadecimal string to be converted
    :return: corresponding base58 string
    """

    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    sb = ''
    payload = int(payload, base=16)
    while payload > 0:
        r = payload % 58
        sb += alphabet[r]
        payload = payload // 58
    return sb[::-1]


def key2Address(payload: str, version: int):
    """Convert public key hash to bitcoin address

    :param payload: public key to be converted (hexadecimal)
    :param version: version prefix in decimal see https://en.bitcoin.it/wiki/Base58Check_encoding#Version_bytes
    :return: corresponding bitcoin address (Base58Check)
    """

    prefix = f'{version:0{2}x}'
    checksum = sha256(bytes.fromhex(prefix + payload)).digest()[:4].hex()  # checksum only take first 4 bytes
    return hex2Base58(prefix + payload + checksum)


@dataclass(frozen=True)
class Output:
    value: int  # amount of BTC in satoshis.
    scriptPubKey_size: int  # number of bytes
    scriptPubKey: str

    @classmethod
    def from_file(cls, file: IO):
        value = int(file.read(8)[::-1].hex(), base=16)
        scriptPubKey_size = read_varint(file)
        scriptPubKey = file.read(scriptPubKey_size).hex()
        return cls(value, scriptPubKey_size, scriptPubKey)

    @property
    def recipients(self):
        """Bitcoin address(es) or public key(s) to which the output instance is sent"""

        # scriptPubKey patterns
        p2pkh = f'{Opcode.DUP}{Opcode.HASH160}([0-9a-f]{{40}}){Opcode.EQUALVERFIY}{Opcode.CHECKSIG}'
        p2pk = f'{Opcode.ONE}([0-9a-f]{{130}}){Opcode.THREE}{Opcode.CHECKMULTISIG}'
        p2ms = f'{Opcode.HASH160}([0-9a-f]{{130}}){{1,3}}{Opcode.EQUAL}'
        p2sh = f'{Opcode.RETURN}([0-9a-f]{{40}})'
        # null_data = f'([0-9a-f]+){Opcode.CHECKSIG}'

        if key_search := re.search(p2ms, self.scriptPubKey):
            return list(key_search.groups())

        for pattern in [p2pkh, p2pk, p2sh]:
            if not (key_search := re.search(pattern, self.scriptPubKey)):
                continue
            if pattern == p2pk:
                return [key_search.group(1)]
            else:
                return [key2Address(key_search.group(1), version='1' if pattern == 'pkh' else '3')]

        return []


@dataclass(frozen=True)
class Transaction:
    version: str
    input_count: int
    inputs: [Input]
    output_count: int
    outputs: [Output]
    locktime: str

    @classmethod
    def from_file(cls, file: IO):
        version = file.read(4)[::-1].hex()

        input_count = read_varint(file)
        inputs = [Input.from_file(file) for _ in range(input_count)]

        output_count = read_varint(file)
        outputs = [Output.from_file(file) for _ in range(output_count)]

        locktime = file.read(4)[::-1].hex()

        return cls(version, input_count, inputs, output_count, outputs, locktime)

    def is_coinbase(self):
        return len(self.inputs) == 1 and self.inputs[0].tx_id == '0' * 64 and self.inputs[0].vout == 'f' * 8


@dataclass(frozen=True)
class Block:
    magic_bytes: str
    size: int  # number of bytes

    # block_header = f.read(80).hex()

    version: str
    prev_block_hash: str
    merkle_root: str
    time_: str
    bits: str
    nonce: str

    tx_count: int
    transactions: [Transaction]

    @classmethod
    def from_file(cls, file: IO):
        magic_bytes = file.read(4)[::-1].hex()
        size = int(file.read(4)[::-1].hex(), base=16)

        # block_header = f.read(80).hex()

        version = file.read(4)[::-1].hex()
        prev_block_hash = file.read(32)[::-1].hex()
        merkle_root = file.read(32)[::-1].hex()
        time_ = file.read(4)[::-1].hex()
        bits = file.read(4)[::-1].hex()
        nonce = file.read(4)[::-1].hex()

        tx_count = read_varint(file)
        transactions = [Transaction.from_file(file) for _ in range(tx_count)]

        return cls(magic_bytes, size,
                   version, prev_block_hash, merkle_root, time_, bits, nonce,
                   tx_count, transactions)

