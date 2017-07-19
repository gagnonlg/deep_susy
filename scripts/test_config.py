import re
import unittest

from deep_susy import utils

class Test_config(unittest.TestCase):
    def test_config(self):
        path = utils.project_path('datasets.config')
        with open(path, 'r') as cfg:
            lines = cfg.readlines()
            seen = []
            i = 0
            while i < len(lines):
                if lines[i].startswith('#'):
                    self.assertFalse(lines[i] in seen, lines[i])
                    seen.append(lines[i])
                m1 = re.match('^# Gtt_(.*)_5000_(.*)$', lines[i])
                if m1 is not None:
                    i += 1
                    m2 = re.search('ttn1_(.*)_5000_(.*)$', lines[i])
                    self.assertTrue(m2 is not None)
                    self.assertEqual(m1.group(1), m2.group(1))
                    self.assertEqual(m1.group(2), m2.group(2))
                i += 1

if __name__ == '__main__':
    unittest.main()
