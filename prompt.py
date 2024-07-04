import os


class Prompt:
    def __init__(self, path: str):
        """
        :param path: path to the prompt directory
        """
        self.path = path
        if not os.path.exists(path):
            raise ValueError(f"File {path} does not exist")
        self._load_prompt()
    
    def _load_prompt(self) -> dict:
        """
        Load all the prompts in the directory
        :return: a dictionary of prompts
        """
        prompts = {}
        for filename in os.listdir(self.path):
            if filename.endswith(".txt"):
                with open(os.path.join(self.path, filename), "r", encoding='utf-8') as f:
                    prompts[filename.removesuffix('.txt')] = f.read()
        self.names = list(prompts.keys())
        self.prompts = prompts

if __name__ == '__main__':
    prompt = Prompt("./prompts")
    print(prompt.names)