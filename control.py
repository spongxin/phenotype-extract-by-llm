from rapidfuzz import process, fuzz, utils
import os


class ItemsControl:
    enzyme_list_path = os.path.join('reference', 'enzymes', 'enzymes.txt')
    antibiotics_list_path = os.path.join('reference', 'antibiotics', 'antibiotics.txt')
    cnsource_list_path = os.path.join('reference', 'cnsource', 'cnsource.txt')
    
    with open(enzyme_list_path, 'r', encoding='utf-8') as f:
        enzyme_list = f.read().splitlines()
    with open(antibiotics_list_path, 'r', encoding='utf-8') as f:
        antibiotics_list = f.read().splitlines()
    with open(cnsource_list_path, 'r', encoding='utf-8') as f:
        cnsource_list = f.read().splitlines()

    @staticmethod
    def control_enzyme_list(enzymes: list) -> list[tuple[str, float, int]]:
        """Control the enzyme list with the given enzyme list.
        :param enzymes: list of enzyme names
        :return: list of tuples containing the enzyme name and its candidate
        """
        return ItemsControl._control_items_by_list(enzymes, ItemsControl.enzyme_list)
    
    @staticmethod
    def control_antibiotics_list(antibiotics: list) -> list[tuple[str, float, int]]:
        """Control the antibiotics list with the given antibiotics list.
        :param antibiotics: list of antibiotics names
        :return: list of tuples containing the antibiotics name and its candidate
        """
        return ItemsControl._control_items_by_list(antibiotics, ItemsControl.antibiotics_list)

    @staticmethod
    def control_cnsource_list(cnsources: list) -> list[tuple[str, float, int]]:
        """Control the cnsource list with the given cnsource list.
        :param cnsources: list of cnsource names
        :return: list of tuples containing the cnsource name and its candidate
        """
        return ItemsControl._control_items_by_list(cnsources, ItemsControl.cnsource_list)

    @staticmethod
    def _control_items_by_list(items: list, control_list: list) -> list[tuple[str, float, int]]:
        """Control the items with the given control list.
        :param items: list of items
        :param control_list: list of control items
        :return: list of tuples containing the item and its candidate
        """
        candidates = [process.extractOne(item, control_list, scorer=fuzz.WRatio, processor=utils.default_process) for item in items]
        return candidates
