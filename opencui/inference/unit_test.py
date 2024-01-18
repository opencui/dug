import unittest
import shutil

from opencui.inference.index import indexing
from opencui.inference.serve import load_converter_from_meta

class AgentTest(unittest.TestCase):
    converter = None

    @classmethod
    def setUpClass(clsc):
        root = "./examples/agent"
        #indexing(root)

        AgentTest.converter = load_converter_from_meta(root)

    @classmethod
    def tearDownClass(cls):\
        pass

    def testDetectTriggerable(self):
        utterance = "I like to order some food"
        result = AgentTest.converter.detect_triggerables(utterance, [])
        print(result)


if __name__ == "__main__":
    unittest.main()