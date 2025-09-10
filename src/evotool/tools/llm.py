import json
import traceback
import http.client
from typing import Any, Tuple


class HttpsApi:
    def __init__(self, host, key, model, url, timeout=60, **kwargs):
        """
        Initialize the HttpsApi class.

        :param host: The host of the API.
        :param key: The API key.
        :param model: The model to use.
        :param url: The URL of the API.
        :param timeout: The timeout for the API request.
        :param kwargs: Additional keyword arguments.
        """
        self._host = host
        self._key = key
        self._model = model
        self._url = url
        self._timeout = timeout
        self._kwargs = kwargs
        self._max_retry = 10

    def get_response(self, prompt: str | Any, *args, **kwargs) -> Tuple[str, dict]:
        if isinstance(prompt, str):
            prompt = [{'role': 'user', 'content': prompt.strip()}]

        retry = 0
        while True:
            try:
                if self._model.startswith("o1-preview"):
                    for p in prompt:
                        if p['role'] == 'system':
                            p['role'] = 'user'

                conn = http.client.HTTPSConnection(self._host, timeout=self._timeout)
                payload = json.dumps({
                    # 'max_tokens': self._kwargs.get('max_tokens', 4096),
                    # 'top_p': self._kwargs.get('top_p', None),
                    'temperature': self._kwargs.get('temperature', 1.0),
                    'model': self._model,
                    'messages': prompt
                })
                headers = {
                    'Authorization': f'Bearer {self._key}',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }
                conn.request('POST', self._url, payload, headers)
                res = conn.getresponse()
                data = res.read().decode('utf-8')
                data = json.loads(data)
                response = data['choices'][0]['message']['content']
                usage = data['usage']
                # if self._model.startswith('claude'):
                #     response = data['content'][0]['text']
                # else:
                #     response = data['choices'][0]['message']['content']
                return response, usage
            except Exception as e:
                retry += 1
                if retry >= self._max_retry:
                    raise RuntimeError(
                        # f'{self.__class__.__name__} error: {traceback.format_exc()}.\n'
                        f'Model Response Error! You may check your API host and API key.'
                    )
                else:
                    print(f'Model Response Error! Retrying...')
                    # print(f'{self.__class__.__name__} error: {traceback.format_exc()}. Retrying...\n')
    def get_embedding(self, text: str | Any, *args, **kwargs) -> str:

        content_embedding = {'input': text, 'model': self._model}

        retry = 0
        while True:
            try:
                conn = http.client.HTTPSConnection(self._host, timeout=self._timeout)
                payload = json.dumps(content_embedding)
                headers = {
                    'Authorization': f'Bearer {self._key}',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }
                conn.request('POST', "/v1/embeddings", payload, headers)
                res = conn.getresponse()
                data = res.read().decode('utf-8')
                data = json.loads(data)
                response = data['data'][0]['embedding']
                # if self._model.startswith('claude'):
                #     response = data['content'][0]['text']
                # else:
                #     response = data['choices'][0]['message']['content']
                return response
            except Exception as e:
                retry += 1
                if retry >= self._max_retry:
                    raise RuntimeError(
                        f'{self.__class__.__name__} error: {traceback.format_exc()}.\n'
                        f'You may check your API host and API key.'
                    )
                else:
                    print(f'{self.__class__.__name__} error: {traceback.format_exc()}. Retrying...\n')