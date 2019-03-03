import unittest
from clf.create_train_test import process_frames
from cli import main

class TestFrameParser(unittest.TestCase):
    def test_empty_file(self):
        # correctly terminate if file does not exist
        frames = process_frames("./data/unittest/ab.mp4")
        self.assertEqual(len(frames), 0)

    def test_small_file(self):
        # file with few frames
        frames = process_frames("./data/unittest/traffic.mp4")
        self.assertEqual(len(frames), 300)


class CliTest(unittest.TestCase):
    def test_cli(self):
        # sanity check for cli
        label = main("./data/unittest/test.jpg")
        self.assertEqual(label,"outdoor")
        
if __name__ == "__main__":
    unittest.main()
