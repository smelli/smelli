import unittest
import smelli


class TestPackage(unittest.TestCase):
    def test_flavio(self):
        self.assertTrue(smelli._flavio_up_to_date)
