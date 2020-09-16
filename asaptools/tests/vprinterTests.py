"""
Tests of the verbose printer utility

Copyright 2017, University Corporation for Atmospheric Research
See the LICENSE.txt file for details
"""

from __future__ import print_function

import sys
import unittest
from os import linesep

from asaptools import vprinter

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO


class VPrinterTests(unittest.TestCase):
    def setUp(self):
        self.header = '[1] '
        self.vprint = vprinter.VPrinter(header=self.header, verbosity=2)

    def testToStr(self):
        data = ['a', 'b', 'c', 1, 2, 3, 4.0, 5.0, 6.0]
        actual = self.vprint.to_str(*data)
        expected = ''.join([str(d) for d in data])
        self.assertEqual(actual, expected)

    def testToStrHeader(self):
        data = ['a', 'b', 'c', 1, 2, 3, 4.0, 5.0, 6.0]
        actual = self.vprint.to_str(*data, header=True)
        expected = self.header + ''.join([str(d) for d in data])
        self.assertEqual(actual, expected)

    def testVPrint(self):
        data = ['a', 'b', 'c', 1, 2, 3, 4.0, 5.0, 6.0]
        backup = sys.stdout
        sys.stdout = StringIO()
        self.vprint(*data)
        actual = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = backup
        expected = self.vprint.to_str(*data) + linesep
        self.assertEqual(actual, expected)

    def testVPrintHeader(self):
        data = ['a', 'b', 'c', 1, 2, 3, 4.0, 5.0, 6.0]
        backup = sys.stdout
        sys.stdout = StringIO()
        self.vprint(*data, header=True)
        actual = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = backup
        expected = self.vprint.to_str(*data, header=True) + linesep
        self.assertEqual(actual, expected)

    def testVPrintVerbosityCut(self):
        data = ['a', 'b', 'c', 1, 2, 3, 4.0, 5.0, 6.0]
        backup = sys.stdout
        sys.stdout = StringIO()
        self.vprint(*data, verbosity=3)
        actual = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = backup
        expected = ''
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
