#!/usr/bin/env python
import unittest, os, imp, sys

if __name__ == '__main__':
    sys.path.append('./')
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    file_dir = os.path.split(os.path.abspath(__file__))[0]
    for file in os.listdir(file_dir):
        path = os.path.join(file_dir, file)
        if os.path.isfile(path) and file.endswith('_test.py'):
            mod = imp.load_source(os.path.splitext(path)[0], path)
            suite.addTest(loader.loadTestsFromModule(mod))

    unittest.TextTestRunner(verbosity=1).run(suite)
