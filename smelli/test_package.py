import unittest
import smelli


class TestPackage(unittest.TestCase):
    def test_flavio(self):
        self.assertTrue(smelli._check_flavio_version())
