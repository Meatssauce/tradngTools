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


def test_txid():
    from hashlib import sha256

    txdata = '0100000001c997a5e56e104102fa209c6a852dd90660a20b2d9c352423edce25857fcd3704000000004847304402204e45e16932b8af514961a1d3a1a25fdf3f4f7732e9d624c6c61548ab5fb8cd410220181522ec8eca07de4860a4acdd12909d831cc56cbbac4622082221a8768d1d0901ffffffff0200ca9a3b00000000434104ae1a62fe09c5f51b13905f07f06b99a2f7159b2225f374cd378d71302fa28414e7aab37397f554a7df5f142c21c1b7303b8a0626f1baded5c72a704f7e6cd84cac00286bee0000000043410411db93e1dcdb8a016b49840f8c53bc1eb68a382e97b1482ecad7b148a6909a5cb2e0eaddfb84ccf9744464f82e160bfa9b8b64f9d4c03f999b8643f656b412a3ac00000000'
    id = sha256(sha256(bytes.fromhex(txdata)).digest()).digest().hex()

    assert id == '169e1e83e930853391bc6f35f605c6754cfead57cf8387639d3b4096c54f18f4'
