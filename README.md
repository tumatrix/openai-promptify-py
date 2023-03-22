# openai-promptify-py

**Important Alert: This library is not supported by OpenAI. But only a utility wrapper maintained by certain human
beings.**

OpenAI Promptify Py is a Python library that provides a wrapper on top of OpenAI API models and abstracts out some
details, such as retrying and chunking into smaller calls. The library also enables users to give model configurations
as a dictionary, including the OpenAI model name, temperature, frequency penalty, presence penalty, and a prompt. Users
can then call the API with a "proxy model name" and get the results without having to provide all configurations again
and again.

## Installation

You can install OpenAI Promptify Py using pip:

    pip install openai-promptify-py @ git+https://github.com/tumatrix/openai-promptify-py.git

_pip repository coming soon_

## Usage

Here's an example of how to use OpenAI Promptify Py:

    from openai_promptify_py import OpenAIPromptifyPy

        models = {
            'foo': {
                'model_name': 'text-ada-001',
                'prompt': 'Lazy brown {animal} jumped {location}',
            },
        }

        feature = OpenAIPromptify(openai_key=os.environ.get('OPENAI_KEY'), model_repo=models)
        response = feature.get_response('foo', {'animal': 'fox', 'location': 'over the moon'})

## Contributing

If you'd like to contribute to OpenAI Promptify Py, please open an issue or pull request on our GitHub repository.

## License

OpenAI Promptify Py is licensed under the MIT License.
