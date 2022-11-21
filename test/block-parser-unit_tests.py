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


def test_decompress_pk():
    from blockchain import decompress_pk
    inputs = ['0229b3e0919adc41a316aad4f41444d9bf3a9b639550f2aa735676ffff25ba3898',
              '02f15446771c5c585dd25d8d62df5195b77799aa8eac2f2196c54b73ca05f72f27']
    outputs = ['0429b3e0919adc41a316aad4f41444d9bf3a9b639550f2aa735676ffff25ba3898d6881e81d2e0163348ff07b3a9a3968401572aa79c79e7edb522f41addc8e6ce',
               '04f15446771c5c585dd25d8d62df5195b77799aa8eac2f2196c54b73ca05f72f274d335b71c85e064f80191e1f7e2437afa676a3e2a5a5fafcf0d27940cd33e4b4']

    for input_, output in zip(inputs, outputs):
        assert decompress_pk(input_) == output


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
    from blockchain import Transaction, varint2Bytes
    import io

    bytes_ = bytes.fromhex('01000000017967a5185e907a25225574544c31f7b059c1a191d65b53dcc1554d339c4f9efc010000006a47304402206a2eb16b7b92051d0fa38c133e67684ed064effada1d7f925c842da401d4f22702201f196b10e6e4b4a9fff948e5c5d71ec5da53e90529c8dbd122bff2b1d21dc8a90121039b7bcd0824b9a9164f7ba098408e63e5b7e3cf90835cceb19868f54f8961a825ffffffff014baf2100000000001976a914db4d1141d0048b1ed15839d0b7a4c488cd368b0e88ac00000000')
    transaction = Transaction.from_file(io.BytesIO(bytes_), False)

    assert transaction.id == 'c1b4e695098210a31fe02abffe9005cffc051bbe86ff33e173155bcbdc5821e3'
