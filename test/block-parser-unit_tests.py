import pytest
import io

from block_parser import read_dat
from blockchain import read_varint, varint2Bytes


# from blockchain import key2Address


# def test_reverse():
#     from block_parser import reverse
#
#     in_ = ['012abc312g']
#     out = ['2g31bc2a01']
#
#     out_actual = [reverse(i) for i in in_]
#
#     assert out_actual == out


# def test_key2address():
#     keys = ['1e99423a4ed27608a15a2616a2b0e9e52ced330ac530edcc32c8ffc6a526aedd']
#     versions = [128]
#     addresses = ['5J3mBbAH58CpQ3Y5RNJpUKPE62SQ5tfcvU2JpbnkeyhfsYB1Jcn']
#
#     for key, version, address in zip(keys, versions, addresses):
#         assert key2Address(key, version=version) == address


def test_read_varint():
    return read_varint(io.BytesIO(bytes.fromhex('6a'))) == '6a'


def test_varint2Bytes():
    return varint2Bytes(106) == bytes.fromhex('6a')


def test_read_and_inverse_read():
    block_path = 'datasets/blocks/blk00000.dat'

    block = next(read_dat([block_path]))
    reconstructed = block.to_bytes()

    assert reconstructed == next(read_dat([block_path])).to_bytes()


# def test_read_and_inverse_read_all():
#     block_path = '../datasets/blocks/blk00000.dat'
#
#     blocks = list(read_dat([block_path]))
#     reconstructed = b''.join(block.to_bytes() for block in blocks)
#
#     with open(block_path, 'rb') as f:
#         bytes_read = f.read()
#
#     assert reconstructed == bytes_read


def test_index_and_read_from_index():
    from block_parser import build_index, read_dat_from_index, read_dat
    import glob
    import os

    blocks_dir = 'datasets/blocks'
    index_path = 'datasets/blocks/index.csv'

    blocks = sorted(list(read_dat(glob.glob(os.path.join(blocks_dir, 'blk*.dat')))), key=lambda x: x.time_)
    build_index(blocks_dir, index_path)
    print(f'{len(blocks)=}')

    assert blocks == list(read_dat_from_index(index_path))
