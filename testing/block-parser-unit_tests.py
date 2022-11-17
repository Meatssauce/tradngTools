import pytest


# def test_reverse():
#     from block_parser import reverse
#
#     in_ = ['012abc312g']
#     out = ['2g31bc2a01']
#
#     out_actual = [reverse(i) for i in in_]
#
#     assert out_actual == out


from Block import key2Address

def test_key2address():
    keys = ['1e99423a4ed27608a15a2616a2b0e9e52ced330ac530edcc32c8ffc6a526aedd']
    versions = [128]
    addresses = ['5J3mBbAH58CpQ3Y5RNJpUKPE62SQ5tfcvU2JpbnkeyhfsYB1Jcn']

    for key, version, address in zip(keys, versions, addresses):
        assert key2Address(key, version=version) == address
