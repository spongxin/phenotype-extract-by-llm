from client import Client
import unittest
import os


class TestJsonExtract(unittest.TestCase):
    def setUp(self):
        self.resdir = os.path.join('output', 'llama3-70b-8192')
        
    def test_get_aviliable_json(self):
        for filename in os.listdir(self.resdir):
            print(filename)
            with open(os.path.join(self.resdir, filename), 'r', encoding='utf-8') as f:
                data = Client.extract_json_data(f.read())
                self.assertTrue(data is not None)
                if type(data) == str:
                    print(data)
                self.assertIsInstance(data, dict)


if __name__ == '__main__':
    unittest.main()