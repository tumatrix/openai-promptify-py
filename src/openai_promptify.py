import re

import openai
from openai import InvalidRequestError

model_prompt_limits = {
    'text-davinci-003': 4000,
    'gpt-3.5-turbo': 4000,
    'text-curie-001': 2048,
    'text-babbage-001': 2048,
    'text-ada-001': 2048,
    'gpt-4': 32000,
}

chat_models = ['gpt-3.5-turbo', 'gpt-4']


class Utils:
    PATTERN_WORD = re.compile(r'\W+')
    PATTERN_SENTENCE = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?])\s')

    def substitute(self, text, subs):
        for k, v in subs.items():
            subkey = '{' + k + '}'
            if subkey not in text:
                raise ValueError(f'Invalid prompt. Missing key {subkey}')
            text = text.replace(subkey, v)
        return text

    def sentences(self, text: str):
        if text is None:
            return []

        texts = text.split('\n')
        retval = []
        for text in texts:
            splits = self.PATTERN_SENTENCE.split(text.strip())
            retval.extend(splits)

        return [x.strip() for x in retval if x.strip() != '']

    def words(self, value) -> list:
        if value is None:
            return []
        retval = self.PATTERN_WORD.split(value)
        return [x for x in retval if x and x.strip() != '']

    def chunk(self, lines, chunk_size, force_split=False):
        if isinstance(lines, str):
            lines = self.sentences(lines)
        inputs = []
        ci = []
        current_size = 0
        for si in lines:
            w = self.words(si)
            l_words = len(w)
            if current_size + l_words > chunk_size:
                if force_split:
                    capacity = int(chunk_size - current_size)
                    ci.append(' '.join(w[:capacity]))
                    si = ' '.join(w[capacity:])
                    l_words = len(w[capacity:])
                current_size = 0
                inputs.append(ci)
                ci = []
            ci.append(si)
            current_size += l_words

        if len(ci) > 0:
            inputs.append(ci)

        inputs = [x for x in inputs if len(x)]
        return inputs

    def find_variables(self, text):
        pattern = r'\{(\w+)\}'
        matches = re.findall(pattern, text)
        return matches


_utils = Utils()


class OpenAIPromptify:

    def __init__(self, openai_key=None, model_repo: dict[str, dict] = None):
        assert openai_key is not None and openai_key != '', 'openai_key must be specified'
        assert model_repo is not None and len(model_repo) > 0, 'model_repo must be specified'
        for _, v in model_repo.items():
            assert v['model_name'] in model_prompt_limits.keys(), f'Invalid model {v["model_name"]}'

        openai.api_key = openai_key
        self.model_repo = model_repo

    def get_response(self, proxy_model_name=None, data: dict[str, str] = None, verbose=False) -> str:
        assert proxy_model_name is not None and proxy_model_name != '', 'model must be specified'
        assert proxy_model_name in self.model_repo.keys(), f'Invalid model {proxy_model_name}'

        data = data or {}

        model_config = self.model_repo[proxy_model_name]

        proxy = InfoExtractor(model_config, verbose=verbose)
        return proxy.execute(data)

    def get_variables(self, proxy_model_name):
        assert proxy_model_name is not None and proxy_model_name != '', 'model must be specified'
        assert proxy_model_name in self.model_repo.keys(), f'Invalid model {proxy_model_name}'

        model_config = self.model_repo[proxy_model_name]
        proxy = InfoExtractor(model_config)
        return proxy.get_variables()


class InfoExtractor:

    def __init__(self, model_config=None, verbose=False) -> None:
        self.verbose = verbose
        self.model_name = model_config['model_name']

        self.prompt = model_config['prompt']
        self.max_tokens = model_config.get('max_tokens', 256)
        self.temperature = model_config.get('temperature', 0.2)
        self.frequency_penalty = model_config.get('frequency_penalty', 1)
        self.presence_penalty = model_config.get('presence_penalty', 1)
        self.top_p = model_config.get('top_p', 1)
        self.chunk_size = (model_prompt_limits[self.model_name] - len(
            _utils.words(self.prompt)) - self.max_tokens) * .8

        self.chunkable = model_config.get('chunkable', False)
        self.chunk_param = model_config.get('chunkable_param', 'text')
        self.max_chunks = model_config.get('max_chunks', 10)

    def get_variables(self):
        return _utils.find_variables(self.prompt)

    def execute(self, data: dict[str, str], chunk_size=None) -> str:
        if self.chunkable:
            chunk_size = chunk_size or self.chunk_size
            text = data[self.chunk_param]
            inputs = _utils.sentences(text)
            inputs = _utils.chunk(inputs, chunk_size)
            if self.verbose:
                print(f'Chunks: {len(inputs)}')
            inputs = inputs[:self.max_chunks]
            m_inputs = []
            for i in inputs:
                m_data = data.copy()
                m_data[self.chunk_param] = '\n'.join(i)
                m_inputs.append(m_data)
            return self._execute_inputs(m_inputs, chunk_size=chunk_size)
        else:
            return self._execute_inputs([data], chunk_size=chunk_size)

    def _execute_inputs(self, inputs, chunk_size=None):
        substituted_inputs = [_utils.substitute(self.prompt, x) for x in inputs]
        responses = [self.call_openai_ex(x, y, chunk_size) for x, y in zip(inputs, substituted_inputs)]
        responses = [x for x in responses if x]
        return '\n'.join(responses)

    def call_openai_ex(self, input, text, current_chunk_size):
        try:
            return self.call_openai_api(f'{text}')
        except InvalidRequestError as e:
            if self.verbose:
                print(e)
            return self.try_splitting_further(input, current_chunk_size, )

    def try_splitting_further(self, input, current_chunk_size):
        chunk_size = current_chunk_size * .75
        return self.execute(input, chunk_size=chunk_size)

    def call_openai_api(self, prompt):
        if self.model_name in chat_models:
            messages = [
                {'role': 'user', 'content': prompt, }
            ]
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
            )
            usage = response['usage']['total_tokens']
            if self.verbose:
                print('\ttotal_tokens:', usage)
            response = [x['message']['content'] for x in response['choices']]

        else:
            response = openai.Completion.create(
                model=self.model_name,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
            )

            usage = response['usage']['total_tokens']
            if self.verbose:
                print('\ttotal_tokens:', usage)
            response = [x['text'] for x in response['choices']]
        response = response[0] if len(response) else ''
        return response
