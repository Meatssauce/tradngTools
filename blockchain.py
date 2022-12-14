import fileinput
import hashlib
import io
from abc import abstractmethod
from hashlib import sha256
import re
from dataclasses import dataclass
from typing import IO

import bitcoin

from tools import Opcode, PKScriptType


def read_varint(file: IO):
    """Read decimal varint from file (bytes io)"""
    tx_count = file.read(1)

    if tx_count.startswith(b'\xfd'):
        tx_count = file.read(2)
    elif tx_count.startswith(b'\xfe'):
        tx_count = file.read(4)
    elif tx_count.startswith(b'\xff'):
        tx_count = file.read(8)

    return int.from_bytes(tx_count, byteorder='little')


def varint_to_bytes(num: int):
    """Turn decimal int into hexadecimal varint and then bytes"""
    if num <= 252:  # 0xfc
        prefix = b''
        length = 1
    elif num <= 65535:  # int('f' * 4, base=16)
        prefix = b'\xfd'
        length = 2
    elif num <= 4294967295:  # int('f' * 8, base=16)
        prefix = b'\xfe'
        length = 4
    elif num <= 18446744073709551615:  # int('f' * 16, base=16)
        prefix = b'\xff'
        length = 8
    else:
        raise ValueError('num too large for varint')

    return prefix + num.to_bytes(length, byteorder='little')


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


def hex_to_base58(payload: str):
    """Converts a hex string into a base58 string

    :param payload: the hexadecimal string to convert
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


def base58_to_hex(payload: str):
    """Converts a base58 string into a hex string

    :param payload: the base58 string to convert
    :return: corresponding hexadecimal string
    """
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    symbol_to_quantity = {sym: i for i, sym in enumerate(alphabet)}
    payload = payload[::-1]
    num = 0

    for unit, symbol in enumerate(payload):
        num += 58 ** unit * symbol_to_quantity[symbol]

    return f'{num:0{40 + 2 + 8}x}'


def pk2pkh(payload: str):
    """Converts an uncompressed public key to public key hash (ripemd160)"""
    return hashlib.new('ripemd160', sha256(bytes.fromhex(payload)).digest()).digest().hex()


def keyhash2address(payload: str, version: int):
    """Convert public key hash to bitcoin address

    :param payload: public key to convert (hexadecimal)
    :param version: version prefix in decimal see https://en.bitcoin.it/wiki/Base58Check_encoding#Version_bytes
    :return: corresponding bitcoin address (Base58Check)
    """
    prefix = f'{version:0{2}x}'
    checksum = sha256(sha256(bytes.fromhex(prefix + payload)).digest()).digest()[:4].hex()  # only take first 4 bytes
    manual_leading_symbol = '1' if version == 0 else ''
    return manual_leading_symbol + hex_to_base58(prefix + payload + checksum)


def address2keyhash(payload: str):
    """Convert bitcoin address to public key hash (or script hash)

    :param payload: the bitcoin address to convert
    :return: corresponding public key hash (or script hash)
    """
    return base58_to_hex(payload[1:] if payload[0] == 1 else payload)[2:-8]


@dataclass(frozen=True)
class Input:
    # Intrinsic properties
    tx_id: str  # that of a tx whose output we take as input
    vout: int  # index of input as an output in the tx
    sig_script_size: int  # number of bytes
    sig_script: str
    sequence: str  # ignored

    # Derived properties
    coinbase: bool

    @property
    def original_output(self):
        return PastTransactions()[self.tx_id].outputs[self.vout]

    @classmethod
    def from_file(cls, file: IO, coinbase: bool = False):
        tx_id = file.read(32)[::-1].hex()
        vout = int.from_bytes(file.read(4), byteorder='little')
        sig_script_size = read_varint(file)
        sig_script = file.read(sig_script_size).hex()
        sequence = file.read(4)[::-1].hex()
        return cls(tx_id, vout, sig_script_size, sig_script, sequence, coinbase)

    def to_bytes(self):
        data = bytes.fromhex(self.tx_id)[::-1] + \
               self.vout.to_bytes(4, byteorder='little') + \
               varint_to_bytes(self.sig_script_size) + \
               bytes.fromhex(self.sig_script) + \
               bytes.fromhex(self.sequence)[::-1]
        return data


@dataclass(frozen=True)
class Output:
    # Intrinsic properties
    value: int  # amount of BTC in satoshis.
    pk_script_size: int  # number of bytes
    pk_script: str

    # Derived properties
    coinbase: bool

    @property
    def pk_script_type(self):
        for pattern in PKScriptType:
            if pattern == PKScriptType.NONSTANDARD:
                continue
            if re.search(f'{pattern}', self.pk_script):
                return pattern
        return PKScriptType.NONSTANDARD

    @property
    def recipients(self):
        """Bitcoin address(es) or public key(s) to which the output instance is sent"""
        if self.pk_script_type == PKScriptType.P2PKH:
            key = re.search(f'{PKScriptType.P2PKH}', self.pk_script).group(1)
            if len(key) == 42:
                key = key[2:]
            return [keyhash2address(key, version=0)]

        if self.pk_script_type == PKScriptType.P2SH:
            key = re.search(f'{PKScriptType.P2SH}', self.pk_script).group(1)
            if len(key) == 42:
                key = key[2:]
            return [keyhash2address(key, version=5)]

        if self.pk_script_type == PKScriptType.P2MS:
            keys = re.search(f'{PKScriptType.P2MS}', self.pk_script).group(1)
            keys = split_public_keys(keys)
            return [bitcoin.pubkey_to_address(key) for key in keys]

        if self.pk_script_type == PKScriptType.P2PK:
            key = re.search(f'{PKScriptType.P2PK}', self.pk_script).group(1)
            if len(key) == 132:
                key = key[2:]
            elif len(key) != 130:
                raise ValueError('Unexpected key length')
            # todo: get rid of above
            return [bitcoin.pubkey_to_address(key)]

        return []

    @property
    def msg(self):
        if self.recipients:
            return None

        null_data = f'^{Opcode.RETURN}([0-9a-f]*)$'

        try:
            return re.search(null_data, self.pk_script).group(1)
        except AttributeError as exc:
            raise ValueError('Failed to match scriptPubKey against any known pattern.') from exc

    @classmethod
    def from_file(cls, file: IO, coinbase: bool = False):
        value = int.from_bytes(file.read(8), byteorder='little')
        pk_script_size = read_varint(file)
        pk_script = file.read(pk_script_size).hex()
        return cls(value, pk_script_size, pk_script, coinbase)

    def to_bytes(self):
        data = self.value.to_bytes(8, byteorder='little') + \
               varint_to_bytes(self.pk_script_size) + \
               bytes.fromhex(self.pk_script)
        return data


@dataclass(frozen=True)
class Transaction:
    # Intrinsic properties
    version: int
    # witness: bool
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
        version = int.from_bytes(file.read(4), byteorder='little')
        file.seek(-4, 1)

        # witness = file.read(2).hex()

        strategy_by_version = {1: NonWitTxFromBytesStrategy}
        self = strategy_by_version[version].from_bytes(file, coinbase)

        PastTransactions()[self.id] = self

        return self

    def to_bytes(self):
        data = self.version.to_bytes(4, byteorder='little') + \
               varint_to_bytes(self.input_count) + \
               b''.join(input_.to_bytes() for input_ in self.inputs) + \
               varint_to_bytes(self.output_count) + \
               b''.join(output.to_bytes() for output in self.outputs) + \
               bytes.fromhex(self.locktime)[::-1]
        return data


@dataclass(frozen=True)
class Header:
    version: int
    prev_block_hash: str
    merkle_root: str
    time_: int
    bits: int
    nonce: int

    @classmethod
    def from_file(cls, file: IO | fileinput.FileInput):
        version = int.from_bytes(file.read(4), byteorder='little')
        prev_block_hash = file.read(32)[::-1].hex()
        merkle_root = file.read(32)[::-1].hex()
        time_ = int.from_bytes(file.read(4), byteorder='little')
        bits = int.from_bytes(file.read(4), byteorder='little')
        nonce = int.from_bytes(file.read(4), byteorder='little')
        return cls(version, prev_block_hash, merkle_root, time_, bits, nonce)

    def to_bytes(self):
        # header
        version = self.version.to_bytes(4, byteorder='little')
        prev_block_hash = bytes.fromhex(self.prev_block_hash)[::-1]
        merkle_root = bytes.fromhex(self.merkle_root)[::-1]
        time_ = self.time_.to_bytes(4, byteorder='little')
        bits = self.bits.to_bytes(4, byteorder='little')
        nonce = self.nonce.to_bytes(4, byteorder='little')

        data = version + prev_block_hash + merkle_root + time_ + bits + nonce

        assert len(data) == 80

        return data


@dataclass(frozen=True)
class Block:
    # Intrinsic properties
    magic_bytes: str
    size: int  # number of bytes
    header: Header
    tx_count: int
    transactions: list[Transaction]

    # Derived properties
    height: int

    @property
    def id(self):
        header_bytes = self.header.to_bytes()
        return sha256(sha256(header_bytes).digest()).digest()[::-1].hex()

    @classmethod
    def from_file(cls, file: IO | fileinput.FileInput, height: int = 0):
        file.seek(4 + 4, 1)
        version = int.from_bytes(file.read(4), byteorder='little')
        file.seek(-(4 + 4 + 4), 1)

        strategy_by_version = {1: BlockV1FromBytesStrategy, 2: BlockV2FromBytesStrategy}
        self = strategy_by_version[version].from_bytes(file, height=height)

        PastBlocks()[self.id] = self

        return self

    def to_bytes(self):
        strategy_by_version = {1: BlockV1ToBytesStrategy, 2: BlockV2ToBytesStrategy}
        return strategy_by_version[self.header.version].to_bytes(self)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class PastBlocks(dict, metaclass=Singleton):
    pass


class PastTransactions(dict, metaclass=Singleton):
    pass


class BlockFromBytesStrategy:

    @staticmethod
    @abstractmethod
    def from_bytes(file: IO | fileinput.FileInput, **kwargs) -> Block:
        pass


class BlockV1FromBytesStrategy(BlockFromBytesStrategy):
    @staticmethod
    def from_bytes(file: IO | fileinput, height: int = None, **kwargs) -> Block:
        magic_bytes = file.read(4).hex()
        size = int.from_bytes(file.read(4), byteorder='little')
        header = Header.from_file(file)
        tx_count = read_varint(file)
        transactions = [Transaction.from_file(file, coinbase=i == 0) for i in range(tx_count)]

        return Block(magic_bytes, size, header, tx_count, transactions, height)


class BlockV2FromBytesStrategy(BlockFromBytesStrategy):
    @staticmethod
    def from_bytes(file: IO | fileinput.FileInput, **kwargs) -> Block:
        temp = BlockV1FromBytesStrategy.from_bytes(file)
        height = BlockV2FromBytesStrategy.read_height(temp.transactions[0].inputs[0].sig_script)
        return Block(temp.magic_bytes, temp.size, temp.header, temp.tx_count, temp.transactions, height)

    @staticmethod
    def read_height(sig_script: str):
        size = int(sig_script[:2], base=16)
        height = int.from_bytes(bytes.fromhex(sig_script[2:])[:size], byteorder='little')
        return height


class BlockToBytesStrategy:
    @staticmethod
    @abstractmethod
    def to_bytes(block) -> bytes:
        pass


class BlockV1ToBytesStrategy(BlockToBytesStrategy):
    @staticmethod
    def to_bytes(block) -> bytes:
        magic_bytes = bytes.fromhex(block.magic_bytes)
        size = block.size.to_bytes(4)
        header = block.header.to_bytes()
        tx_count = varint_to_bytes(block.tx_count)
        transactions = b''.join(tx.to_bytes() for tx in block.transactions)

        return magic_bytes + size + header + tx_count + transactions


class BlockV2ToBytesStrategy(BlockV1ToBytesStrategy):
    pass


class TransactionFromBytesStrategy:
    @staticmethod
    @abstractmethod
    def from_bytes(file: IO | fileinput.FileInput, **kwargs) -> Transaction:
        pass


class NonWitTxFromBytesStrategy(TransactionFromBytesStrategy):
    @staticmethod
    def from_bytes(file: IO | fileinput.FileInput, coinbase: bool = False, **kwargs) -> Transaction:
        version = int.from_bytes(file.read(4), byteorder='little')

        input_count = read_varint(file)
        inputs = [Input.from_file(file, coinbase) for _ in range(input_count)]

        output_count = read_varint(file)
        outputs = [Output.from_file(file, coinbase) for _ in range(output_count)]

        locktime = file.read(4)[::-1].hex()

        return Transaction(version, input_count, inputs, output_count, outputs, locktime, coinbase)
