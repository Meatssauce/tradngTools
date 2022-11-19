from hashlib import sha256
import re
from dataclasses import dataclass
from typing import IO

from opcodes import Opcode
from tools import hex2


def read_varint(file: IO):
    tx_count = file.read(1).hex()

    if tx_count.startswith('fd'):
        tx_count = file.read(2)[::-1].hex()
    elif tx_count.startswith('fe'):
        tx_count = file.read(4)[::-1].hex()
    elif tx_count.startswith('ff'):
        tx_count = file.read(8)[::-1].hex()

    return int(tx_count, base=16)


def varint2Bytes(num: int):
    """Turn decimal int into hexadecimal varint and then bytes"""
    if num <= 252:  # 0xfc
        prefix = ''
        length = 2
    elif num <= 65535:  # int('f' * 4, base=16)
        prefix = 'fd'
        length = 4
    elif num <= 4294967295:  # int('f' * 8, base=16)
        prefix = 'fe'
        length = 8
    elif num <= 18446744073709551615:  # int('f' * 16, base=16)
        prefix = 'ff'
        length = 16
    else:
        raise ValueError('num too large for varint')

    return bytes.fromhex(f'{prefix}{num:0{length}x}')


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

    def to_bytes(self):
        data = bytes.fromhex(self.tx_id)[::-1] + bytes.fromhex(self.vout)[::-1] + varint2Bytes(self.scriptSig_size) + \
               bytes.fromhex(self.scriptSig) + bytes.fromhex(self.sequence)[::-1]
        return data

    def is_coinbase(self):
        return self.tx_id == '0' * 64 and self.vout == 'f' * 8


# 8def hex2Base58(payload: str):
#     """Converts a hex string into a base58 string
#
#     :param payload: the hexadecimal string to be converted
#     :return: corresponding base58 string
#     """
#
#     alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
#     sb = ''
#     payload = int(payload, base=16)
#     while payload > 0:
#         r = payload % 58
#         sb += alphabet[r]
#         payload = payload // 58
#     return sb[::-1]
#
#
# def key2Address(payload: str, version: int):
#     """Convert public key hash to bitcoin address
#
#     :param payload: public key to be converted (hexadecimal)
#     :param version: version prefix in decimal see https://en.bitcoin.it/wiki/Base58Check_encoding#Version_bytes
#     :return: corresponding bitcoin address (Base58Check)
#     """
#
#     prefix = f'{version:0{2}x}'[::-1]
#     checksum = sha256(bytes.fromhex(prefix + payload)).digest()[:4].hex()  # checksum only take first 4 bytes
#     return hex2Base58(prefix + payload + checksum)


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
        # null_data = f'([0-9a-f]*){Opcode.CHECKSIG}'

        if key_search := re.search(p2ms, self.scriptPubKey):
            return list(key_search.groups())

        for pattern in [p2pkh, p2pk, p2sh]:
            if not (key_search := re.search(pattern, self.scriptPubKey)):
                continue
            return [key_search.group(1)]

        return []

    @property
    def msg(self):
        if self.recipients:
            return None

        null_data = f'([0-9a-f]*){Opcode.CHECKSIG}'

        try:
            return re.search(null_data, self.scriptPubKey).group(1)
        except AttributeError as exc:
            raise ValueError('Failed to match scriptPubKey against any known pattern.') from exc

    def to_bytes(self):
        data = bytes.fromhex(hex2(self.value))[::-1] + varint2Bytes(self.scriptPubKey_size) + \
               bytes.fromhex(self.scriptPubKey)
        return data


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

    @property
    def id(self):
        return sha256(sha256(self.to_bytes()).digest()).digest()

    def to_bytes(self):
        data = bytes.fromhex(self.version)[::-1] + varint2Bytes(self.input_count) + \
               b''.join(varint2Bytes(i) + input_.to_bytes() for i, input_ in enumerate(self.inputs)) + \
               varint2Bytes(self.output_count) + \
               b''.join(varint2Bytes(i) + output.to_bytes() for i, output in enumerate(self.outputs)) + \
               bytes.fromhex(self.locktime)[::-1]
        return data

    def is_coinbase(self):
        return len(self.inputs) == 1 and self.inputs[0].id == '0' * 64 and self.inputs[0].vout == 'f' * 8


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

    def to_bytes(self):
        magic_bytes = bytes.fromhex(self.magic_bytes)[::-1]
        size = bytes.fromhex(hex2(self.size))[::-1]
        version = bytes.fromhex(self.version)[::-1]
        pre_block_hash = bytes.fromhex(self.prev_block_hash)[::-1]
        merkle_root = bytes.fromhex(self.merkle_root)[::-1]
        time_ = bytes.fromhex(self.time_)[::-1]
        bits = bytes.fromhex(self.bits)[::-1]
        nonce = bytes.fromhex(self.nonce)[::-1]
        tx_count = varint2Bytes(self.tx_count)
        transactions = b''.join(tx.to_bytes() for tx in self.transactions)
        data = bytes.fromhex(self.magic_bytes)[::-1] + bytes.fromhex(hex2(self.size))[::-1] + \
               bytes.fromhex(self.version)[::-1] + bytes.fromhex(self.prev_block_hash)[::-1] + \
               bytes.fromhex(self.merkle_root)[::-1] + bytes.fromhex(self.time_)[::-1] + \
               bytes.fromhex(self.bits)[::-1] + bytes.fromhex(self.nonce)[::-1] + \
               varint2Bytes(self.tx_count) + b''.join(tx.to_bytes() for tx in self.transactions)
        return data
