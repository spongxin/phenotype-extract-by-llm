import os

    
class LocalClient:
    def __init__(self, config: dict):
        from openai import OpenAI
        self.client = OpenAI(
            base_url = config.get("endpoint"),
            api_key = config.get("api_key"),
        )

    def chat(self, stop: list = ['<|eot_id|>'], **kwargs):
        return self.client.chat.completions.create(stop=stop, **kwargs)


class AzureGPTClient:
    def __init__(self, **kwargs):
        from openai import AzureOpenAI
        self.client = AzureOpenAI(
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
            api_version="2024-02-01"
        )
    
    def chat(self, messages, **kwargs):
        return self.client.chat.completions.create(
            model=kwargs.get('model'),
            messages=messages, 
            temperature=float(kwargs.get('temperature'))
        )

ClientList = [LocalClient, AzureGPTClient]