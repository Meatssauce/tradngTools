import fileinput
from hashlib import sha256
import re
from dataclasses import dataclass
from typing import IO

from tools import hex2, Opcode, ScriptPubKeyType


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


def decompress_pk(compressed: str):
    # Split compressed key in to prefix and x-coordinate
    prefix = compressed[:2]
    x = int(compressed[2:], base=16)

    # Secp256k1 curve parameters
    # p = int('fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f', base=16)
    p = 115792089237316195423570985008687907853269984665640564039457584007908834671663

    # Work out y values using the curve equation y^2 = x^3 + 7
    y_sq = (x ** 3 + 7) % p  # everything is modulo p

    # Secp256k1 is chosen in a special way so that the square root of y is y^((p+1)/4)
    y = pow(y_sq, (p + 1) // 4, p)  # use modular exponentation

    # Use prefix to select the correct value for y
    # * 02 prefix = y is even
    # * 03 prefix = y is odd
    if (prefix == '02' and y % 2 != 0) or (prefix == '03' and y % 2 == 0):
        y = p - y

    # Construct the uncompressed public key
    x = f'{x:0{64}x}'  # convert to hex and make sure it's 32 bytes (64 characters)
    y = f'{y:0{64}x}'
    uncompressed = '04' + x + y

    # Result
    return uncompressed


def split_public_keys(keys: str):
    """Split a string of public keys in hexadecimal

    :param keys: the string of public keys in hexadecimal to split
    :return: list of split keys
    """

    if not keys:
        return []

    if keys.startswith('04'):
        key_byte_count = 64
        pk = keys[:2 + key_byte_count * 2]
    elif keys.startswith('03') or keys.startswith('02'):
        key_byte_count = 32
        pk = decompress_pk(keys[:2 + key_byte_count * 2])
    else:
        raise ValueError('Illegal keys format')
    return [pk] + split_public_keys(keys[2 + key_byte_count * 2:])


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


# def hex2Base58(payload: str):
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
class Input:
    # Intrinsic properties
    tx_id: str  # that of a tx whose output we take as input
    vout: int  # index of input as an output in the tx
    scriptSig_size: int  # number of bytes
    scriptSig: str
    sequence: str  # ignored

    # Derived properties
    coinbase: bool

    @classmethod
    def from_file(cls, file: IO, coinbase: bool = False):
        tx_id = file.read(32)[::-1].hex()
        vout = int.from_bytes(file.read(4), byteorder='little')
        scriptSig_size = read_varint(file)
        scriptSig = file.read(scriptSig_size).hex()
        sequence = file.read(4)[::-1].hex()
        return cls(tx_id, vout, scriptSig_size, scriptSig, sequence, coinbase)

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
        data = bytes.fromhex(self.tx_id)[::-1] + self.vout.to_bytes(4, byteorder='little') + \
               varint2Bytes(self.scriptSig_size) + bytes.fromhex(self.scriptSig) + bytes.fromhex(self.sequence)[::-1]
        return data


@dataclass(frozen=True)
class Output:
    # Intrinsic properties
    value: int  # amount of BTC in satoshis.
    scriptPubKey_size: int  # number of bytes
    scriptPubKey: str

    # Derived properties
    coinbase: bool

    @property
    def scriptPubKey_type(self):
        for pattern in ScriptPubKeyType:
            if pattern == ScriptPubKeyType.NONSTANDARD:
                continue
            if re.search(f'{pattern}', self.scriptPubKey):
                return pattern
        return ScriptPubKeyType.NONSTANDARD

    @property
    def recipients(self):
        """Bitcoin address(es) or public key(s) to which the output instance is sent"""

        if key_search := re.search(f'{ScriptPubKeyType.P2MS}', self.scriptPubKey):
            keys = key_search.group(1)
            keys = split_public_keys(keys)
            return keys

        for pattern in [ScriptPubKeyType.P2PK, ScriptPubKeyType.P2PKH, ScriptPubKeyType.P2SM]:
            if not (key_search := re.search(f'{pattern}', self.scriptPubKey)):
                continue
            return [key_search.group(1)]

        return []

    @property
    def msg(self):
        if self.recipients:
            return None

        null_data = f'^{Opcode.RETURN}([0-9a-f]*)$'

        try:
            return re.search(null_data, self.scriptPubKey).group(1)
        except AttributeError as exc:
            raise ValueError('Failed to match scriptPubKey against any known pattern.') from exc

    @classmethod
    def from_file(cls, file: IO, coinbase: bool = False):
        value = int.from_bytes(file.read(8), byteorder='little')
        scriptPubKey_size = read_varint(file)
        scriptPubKey = file.read(scriptPubKey_size).hex()
        return cls(value, scriptPubKey_size, scriptPubKey, coinbase)

    def to_bytes(self):
        data = self.value.to_bytes(8, byteorder='little') + varint2Bytes(self.scriptPubKey_size) + \
               bytes.fromhex(self.scriptPubKey)
        return data


@dataclass(frozen=True)
class Transaction:
    # Intrinsic properties
    version: str
    input_count: int
    inputs: list[Input]
    output_count: int
    outputs: list[Output]
    locktime: str

    # Derived properties
    coinbase: bool

    @property
    def value(self):
        return sum(output.value for output in self.outputs)

    @property
    def id(self):
        return sha256(sha256(self.to_bytes()).digest()).digest()[::-1].hex()

    @classmethod
    def from_file(cls, file: IO, coinbase: bool = False):
        version = file.read(4)[::-1].hex()

        input_count = read_varint(file)
        inputs = [Input.from_file(file, coinbase) for _ in range(input_count)]

        output_count = read_varint(file)
        outputs = [Output.from_file(file, coinbase) for _ in range(output_count)]

        locktime = file.read(4)[::-1].hex()

        return cls(version, input_count, inputs, output_count, outputs, locktime, coinbase)

    def to_bytes(self):
        data = bytes.fromhex(self.version)[::-1] + varint2Bytes(self.input_count) + \
               b''.join(input_.to_bytes() for input_ in self.inputs) + \
               varint2Bytes(self.output_count) + \
               b''.join(output.to_bytes() for output in self.outputs) + \
               bytes.fromhex(self.locktime)[::-1]
        return data


@dataclass(frozen=True)
class Block:
    # Intrinsic properties
    magic_bytes: str
    size: int  # number of bytes

    # block_header = f.read(80).hex()

    version: str
    prev_block_hash: str
    merkle_root: str
    time_: int
    bits: int
    nonce: int

    tx_count: int
    transactions: list[Transaction]

    # Derived properties
    height: int

    @classmethod
    def from_file(cls, file: IO | fileinput.FileInput, height: int = 0):
        magic_bytes = file.read(4)[::-1].hex()
        size = int.from_bytes(file.read(4), byteorder='little')
        # block_header = f.read(80).hex()

        # header
        version = file.read(4)[::-1].hex()
        prev_block_hash = file.read(32)[::-1].hex()
        merkle_root = file.read(32)[::-1].hex()
        time_ = int.from_bytes(file.read(4), byteorder='little')
        bits = int.from_bytes(file.read(4), byteorder='little')
        nonce = int.from_bytes(file.read(4), byteorder='little')

        tx_count = read_varint(file)
        transactions = [Transaction.from_file(file, coinbase=i == 0) for i in range(tx_count)]

        return cls(magic_bytes, size,
                   version, prev_block_hash, merkle_root, time_, bits, nonce,
                   tx_count, transactions, height)

    def to_bytes(self):
        magic_bytes = bytes.fromhex(self.magic_bytes)[::-1]
        size = bytes.fromhex(f'{self.size:0{8}x}')[::-1]

        # header
        version = bytes.fromhex(self.version)[::-1]
        pre_block_hash = bytes.fromhex(self.prev_block_hash)[::-1]
        merkle_root = bytes.fromhex(self.merkle_root)[::-1]
        time_ = bytes.fromhex(f'{self.time_:0{8}x}')[::-1]
        bits = bytes.fromhex(f'{self.bits:0{8}x}')[::-1]
        nonce = bytes.fromhex(f'{self.nonce:0{8}x}')[::-1]

        tx_count = varint2Bytes(self.tx_count)
        transactions = b''.join(tx.to_bytes() for tx in self.transactions)

        data = (magic_bytes + size + version + pre_block_hash + merkle_root + time_ + bits + nonce +
                tx_count + transactions)

        return data
