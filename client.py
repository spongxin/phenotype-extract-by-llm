from groq import Groq
import logging
import time
import os


class Client:
    interval_seconds = 30
    def __init__(self, api_keys_path: str = "api-keys.txt"):
        if not os.path.exists(api_keys_path):
            raise FileNotFoundError(f"API keys file not found: {api_keys_path}")
        with open(api_keys_path, "r", encoding='utf-8') as f:
            self._clients = [Groq(api_key=key.strip()) for key in f.readlines() if key.strip()]
        
        self._request_table = {idx: None for idx in range(len(self._clients))}
        self.clients_num = len(self._clients)
    
    def get_aviliable_client(self) -> Groq:
        min_waiting_time = self.interval_seconds
        current_time = time.time()
        for idx, client in enumerate(self._clients):
            if self._request_table[idx] is None or current_time - self._request_table[idx] >= self.interval_seconds:
                self._request_table[idx] = current_time
                return client
            waiting_time = self.interval_seconds - (current_time - self._request_table[idx])
            min_waiting_time = min(min_waiting_time, waiting_time)
        logging.warning(f"all clients are busy, waiting for {min_waiting_time} seconds")
        time.sleep(int(min_waiting_time)+1)
        return self.get_aviliable_client()