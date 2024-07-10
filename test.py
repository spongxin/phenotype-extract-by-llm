from client import Client
from control import QualityControl
import unittest
import os


class TestJsonExtract(unittest.TestCase):
    def setUp(self):
        self.resdir = os.path.join('output', 'llama3-70b-8192')
        
    def test_get_aviliable_json(self):
        for filename in [i for i in os.listdir(self.resdir) if i.endswith('.txt')]:
            with open(os.path.join(self.resdir, filename), 'r', encoding='utf-8') as f:
                data = Client.extract_json_data(f.read())
                self.assertTrue(data is not None)


class TestQualityControl(unittest.TestCase):
    def test_control_enzyme_list(self):
        enzyme_list = [
            'pyrroloquinoline quinone (PQQ)-linked methanol dehydrogenase', 
            'c-glutamylmethylamide synthetase', 
            'N-methylglutamate synthase/lyase', 
            'glutathione-dependent formaldehyde dehydrogenase', 
            'NAD-dependent formate dehydrogenase', 
            '3-hexulose phosphate synthase', 
            '2-keto-3-deoxy-6-phosphogluconate aldolase', 
            'glucose-6-phosphate dehydrogenase',
            '6-phosphogluconate dehydrogenase',
            'phosphoenolpyruvate carboxylase',
            'glutamate dehydrogenase'
        ]
        candidates = QualityControl.control_enzyme_list(enzyme_list)
        for e, c in zip(enzyme_list, candidates):
            print(f'{e} -> {c}')


if __name__ == '__main__':
    unittest.main()