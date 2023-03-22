import os
from unittest import TestCase

from openai_promptify import OpenAIPromptify


class TestE2E(TestCase):

    def x_test_e2e(self):
        models = {
            'foo': {
                'model_name': 'text-ada-001',
                'prompt': 'Lazy brown {animal} jumped {location}',
            },
        }

        feature = OpenAIPromptify(openai_key=os.environ.get('OPENAI_KEY'), model_repo=models)
        response = feature.get_response('foo', {'animal': 'fox', 'location': 'over the moon'})
        self.assertNotEqual('', response)
        self.assertIsNotNone(response)
        print(response)
