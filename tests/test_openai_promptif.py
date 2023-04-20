from unittest import TestCase, mock

from openai import InvalidRequestError

from openai_promptify import OpenAIPromptify, Utils

_models = {
    'foo': {
        'model_name': 'text-ada-001',
        'prompt': 'Lazy brown {animal} jumped {location}',
    },

    'foo_chunk': {
        'model_name': 'text-ada-001',
        'prompt': '{key} the following text:\n{text}\n',
        'chunkable': True,
        'chunkable_param': 'text',
    },

    'foo_chat': {
        'model_name': 'gpt-3.5-turbo',
        'prompt': 'Lazy brown {animal} jumped {location}',
    },

    'no_variable_model': {
        'model_name': 'text-ada-001',
        'prompt': 'Lazy brown fox jumped over the moon',
    },
}


class TestPromptify(TestCase):

    def setUp(self) -> None:
        self.feature = OpenAIPromptify(openai_key='dummy', model_repo=_models)

    @mock.patch('openai_promptify.InfoExtractor.call_openai_api')
    def test_foo(self, mock_call_openai_api):

        def side_effect(*args, **kwargs):
            if args[0] == 'Lazy brown fox jumped over the moon':
                return 'This is correct statement'
            else:
                raise ValueError('Unexpected prompt', args[0])

        mock_call_openai_api.side_effect = side_effect

        actual = self.feature.get_response('foo', {'animal': 'fox', 'location': 'over the moon'})
        self.assertEqual('This is correct statement', actual)

    @mock.patch('openai_promptify.InfoExtractor.call_openai_api')
    def test_foo_chunk(self, mock_call_openai_api):

        def side_effect(*args, **kwargs):
            prompt = args[0]
            if not prompt.startswith('bar the following text:'):
                raise ValueError('Unexpected prompt', prompt)
            return 'This is correct statement'

        mock_call_openai_api.side_effect = side_effect

        text = [f'{i}: a really long text. ' for i in range(400)]
        text = ' '.join(text)
        actual = self.feature.get_response('foo_chunk', {'text': text, 'key': 'bar'})
        expected = ('This is correct statement\n' * 2).strip()
        self.assertEqual(expected, actual)
        self.assertEqual(2, mock_call_openai_api.call_count)

    @mock.patch('openai_promptify.InfoExtractor.call_openai_api')
    def test_foo_chunk_retry(self, mock_call_openai_api):
        error_triggerred_once = False

        def side_effect(*args, **kwargs):
            prompt = args[0]
            if not prompt.startswith('bar the following text:'):
                raise ValueError('Unexpected prompt', prompt)
            n_sentences = len(prompt.split('\n'))
            if n_sentences > 250:
                raise InvalidRequestError(f'too long prompt: {n_sentences}', -10)
            return 'This is correct statement'

        mock_call_openai_api.side_effect = side_effect

        text = [f'{i}: a really long text. ' for i in range(400)]
        text = ' '.join(text)
        text = 'there is a error_trigger in the text' + text
        actual = self.feature.get_response('foo_chunk', {'text': text, 'key': 'bar'})
        expected = ('This is correct statement\n' * 3).strip()
        self.assertEqual(expected, actual)
        self.assertEqual(4, mock_call_openai_api.call_count)  # 1 error, 2 break down, last one is ok

    @mock.patch('openai.ChatCompletion.create')
    def test_end_to_end_chat_model(self, mock):

        def side_effect(*args, **kwargs):
            assert kwargs['messages'][0]['role'] == 'user'
            assert kwargs['messages'][0]['content'] == 'Lazy brown fox jumped over the moon'
            assert len(kwargs['messages']) == 1
            assert kwargs['model'] == 'gpt-3.5-turbo'
            return {
                'usage': {'total_tokens': 10},
                'choices': [{'message': {'content': 'This is correct statement'}}]

            }

        mock.side_effect = side_effect

        self.feature = OpenAIPromptify(openai_key='dummy', model_repo=_models, )
        actual = self.feature.get_response('foo_chat', {'animal': 'fox', 'location': 'over the moon'})
        self.assertEqual('This is correct statement', actual)

    @mock.patch('openai.Completion.create')
    def test_end_to_end_completion_model(self, mock):

        def side_effect(*args, **kwargs):
            assert kwargs['prompt'] == 'Lazy brown fox jumped over the moon'
            assert kwargs['model'] == 'text-ada-001'
            return {
                'usage': {'total_tokens': 10},
                'choices': [{'text': 'This is correct statement'}]

            }

        mock.side_effect = side_effect

        self.feature = OpenAIPromptify(openai_key='dummy', model_repo=_models, )
        actual = self.feature.get_response('foo', {'animal': 'fox', 'location': 'over the moon'})
        self.assertEqual('This is correct statement', actual)

    def test_get_variables(self):
        actual = self.feature.get_variables(proxy_model_name='foo')
        expected = ['animal', 'location']
        self.assertEqual(expected, actual)

    def test_get_variables_no_variable(self):
        actual = self.feature.get_variables(proxy_model_name='no_variable_model')
        expected = []
        self.assertEqual(expected, actual)


class TestUtils(TestCase):
    feature = Utils()

    def test_substitute(self):
        text = 'Lazy brown {animal} jumped {location}'
        data = {'animal': 'fox', 'location': 'over the moon'}
        actual = self.feature.substitute(text, data)
        self.assertEqual('Lazy brown fox jumped over the moon', actual)

    def test_substitute_with_more_params(self):
        text = 'Lazy brown {animal} jumped {location}'
        data = {'animal': 'fox', 'location': 'over the moon', 'foo': 'bar'}
        with self.assertRaises(ValueError):
            _ = self.feature.substitute(text, data)

    def test_sentence_splitter(self):
        text = 'Lazy brown fox jumped over the moon. Lazy brown fox jumped over the moon.\nLorum ipsum'
        actual = self.feature.sentences(text)
        expected = ['Lazy brown fox jumped over the moon.', 'Lazy brown fox jumped over the moon.', "Lorum ipsum"]
        self.assertEqual(expected, actual)

    def test_chunks(self):
        text = [f'{i}: a really long text.' for i in range(10)]  # 5 * 10 = 50
        actual = self.feature.chunk(text, 20)
        self.assertEqual(3, len(actual))
        self.assertEqual([4, 4, 2], [len(x) for x in actual])

    def test_chunks_force_split(self):
        text = 'Lazy brown fox jumped over the moon without a statement end lorum ipsum and a really long text'
        actual = self.feature.chunk(text, 10, force_split=True)

        self.assertEqual(2, len(actual))
        self.assertEqual([1, 1], [len(x) for x in actual])

    def test_words(self):
        text = 'Lazy brown fox jumped over the moon. Lazy brown fox jumped over the moon.\nLorum ipsum'
        actual = self.feature.words(text)
        expected = ['Lazy', 'brown', 'fox', 'jumped', 'over', 'the', 'moon', 'Lazy', 'brown', 'fox', 'jumped', 'over',
                    'the', 'moon', 'Lorum', 'ipsum']
        self.assertEqual(expected, actual)

    def test_words_with_punc(self):
        text = '1: a really long text.\n2: really long text'
        actual = self.feature.words(text)
        expected = ['1', 'a', 'really', 'long', 'text', '2', 'really', 'long', 'text']
        self.assertEqual(expected, actual)
