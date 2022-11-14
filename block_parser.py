import hashlib
import warnings
from itertools import pairwise
from typing import IO


def make_merkle_root(lst):  # https://gist.github.com/anonymous/7eb080a67398f648c1709e41890f8c44
    if len(lst) == 1:
        return lst[0]
    if len(lst) % 2 == 1:
        lst.append(lst[-1])

    sha256d = lambda x: hashlib.sha256(x).digest()
    hash_pair = lambda x, y: sha256d(sha256d(x[::-1] + y[::-1]))[::-1]
    # todo: check if its addition or arr of x y

    return make_merkle_root([hash_pair(x, y) for x, y in pairwise(lst)])


def read_varint(file: IO) -> int:
    tx_count = file.read(1).hex()

    if tx_count.startswith('fd'):
        tx_count = file.read(2)[::-1].hex()
    elif tx_count.startswith('fe'):
        tx_count = file.read(4)[::-1].hex()
    elif tx_count.startswith('ff'):
        tx_count = file.read(8)[::-1].hex()

    return int(tx_count, base=16)


def read_tx_input(file: IO) -> dict:
    tx_id = file.read(32)[::-1].hex()
    vout = file.read(4)[::-1].hex()
    scriptSig_size = read_varint(file)
    scriptSig = file.read(scriptSig_size).hex()
    sequence = file.read(4)[::-1].hex()

    return {'tx_id': tx_id, 'vout': vout, 'scriptSig': scriptSig, 'sequence': sequence}


def read_tx_output(file: IO) -> dict:
    value = file.read(8)[::-1].hex()
    scriptPubKey_size = read_varint(file)
    scriptPubKey = file.read(scriptPubKey_size).hex()

    return {'value': value, 'scriptPubKey': scriptPubKey}


def read_transaction(file: IO) -> dict:
    version = file.read(4)[::-1].hex()

    input_count = read_varint(file)
    inputs = [read_tx_input(file) for _ in range(input_count)]

    output_count = read_varint(file)
    outputs = [read_tx_output(file) for _ in range(output_count)]

    locktime = file.read(4)[::-1].hex()

    return {'version': version, 'inputs': inputs, 'outputs': outputs, 'locktime': locktime}


# def read_block(block_dir):
#     block_dir = 'D:/Bitcoin/blocks/'  # Directory where blk*.dat files are stored
#     out_dir = './result/'  # Directory where to save parsing results
#
#     for filename in sorted(x for x in os.listdir(block_dir) if x.endswith('.dat') and x.startswith('blk')):
#         filepath = block_dir + filename
#
#         msg = f'Reading block from {filepath} at {datetime.datetime.now()}\n'
#         print(msg)


def read_block(file: IO) -> dict:
    magic_bytes = file.read(4)[::-1].hex()
    size = file.read(4)[::-1].hex()

    # block_header = f.read(80).hex()

    version = file.read(4)[::-1].hex()
    prev_block_hash = file.read(32)[::-1].hex()
    merkle_root = file.read(32)[::-1].hex()
    time_ = file.read(4)[::-1].hex()
    bits = file.read(4)[::-1].hex()
    nonce = file.read(4)[::-1].hex()

    tx_count = read_varint(file)
    transactions = [read_transaction(file) for _ in range(tx_count)]

    # if make_merkle_root(...) != merkle_root:
    #     warnings.warn('Merkle root does not match transactions. '
    #                   'Something is wrong if this is an official block.')

    return {'magic_bytes': magic_bytes, 'size': size, 'version': version, 'prev_block_hash': prev_block_hash,
            'merkle_root': merkle_root, 'time': time_, 'bits': bits, 'nonce': nonce, 'transactions': transactions}


def read_dat(filepath) -> [dict]:
    blocks = []

    with open(filepath, 'rb') as f:
        while True:
            try:
                blocks.append(read_block(f))
            except EOFError:
                break

    return blocks


def main():
    pass


if __name__ == '__main__':
    main()
