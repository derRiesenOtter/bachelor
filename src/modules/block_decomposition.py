import unittest

import numpy as np


def block_decomposition(
    sequence: str, balance: int, min_block_size: int, mapping: dict
):
    mapped_sequence = map_sequence(sequence, mapping)
    pass


def map_sequence(sequence: str, mapping: dict) -> np.ndarray:
    return np.array([mapping.get(char) for char in sequence])


def check_block(sequence: str, balance: int, min_block_size: int):
    pass


def create_dict_list(sequence: str, block_length: int):
    pass


def get_balance(factor_left: dict, factor_right: dict):
    max_balance = 0
    for key, value in factor_left.items():
        balance = abs(value - factor_right.get(key, 0))
        if balance > max_balance:
            max_balance = balance
    return max_balance


class TestBlockDecomposition(unittest.TestCase):
    def test_one(self):
        result = map_sequence("ABCD", {"A": 0, "B": 1, "C": 2, "D": 0})
        excepted = np.array([0, 1, 2, 0])
        np.testing.assert_array_equal(result, excepted)


class TestMapSequence(unittest.TestCase):
    pass


class TestCheckBlock(unittest.TestCase):
    pass


class TestCreateDictList(unittest.TestCase):
    pass


class TestGetBalance(unittest.TestCase):
    def test_one(self):
        result = get_balance({"0": 2, "1": 1}, {"0": 1, "1": 2})
        expected = 1
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
