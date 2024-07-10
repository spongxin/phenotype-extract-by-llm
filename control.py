from rapidfuzz import process, fuzz, utils
import os


class QualityControl:
    enzyme_list_path = os.path.join('reference', 'enzymes', 'enzymes.txt')

    @staticmethod
    def control_enzyme_list(enzymes: list) -> list[tuple[str, float, int]]:
        """Control the enzyme list with the given enzyme list.
        :param enzymes: list of enzyme names
        :return: list of tuples containing the enzyme name and its candidate
        """
        with open(QualityControl.enzyme_list_path, 'r', encoding='utf-8') as f:
            enzyme_list = [i.strip() for i in f.readlines()]
        candidates = [process.extractOne(enzyme, enzyme_list, scorer=fuzz.WRatio, processor=utils.default_process) for enzyme in enzymes]
        return candidates


