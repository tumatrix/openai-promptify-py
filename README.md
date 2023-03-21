# openai-promptify-py
OpenAI Promptify Py is a Python library that provides a wrapper on top of OpenAI API models and abstracts out some of the details, such as retrying and chunking into smaller calls. The library also enables users to give model configurations as a dictionary, including the OpenAI model name, temperature, frequency penalty, presence penalty, and a prompt. Users can then call the API with a "proxy model name" and get the results without having to provide all configurations again and again.

## Installation
You can install OpenAI Promptify Py using pip:

```
pip install openai-promptify-py
```

## Usage
Here's an example of how to use OpenAI Promptify Py:

```
from openai_promptify_py import OpenAIPromptifyPy

promptify = OpenAIPromptifyPy(api_key="your-api-key", model_config={
    "open-ai-model-name": "davinci",
    "temperature": 0.5,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.5,
    "prompt": "The quick brown {animal}"
})

data = {"animal": "fox"}
response = promptify.get_response(proxy_model_name="my-proxy-model", data=data)
print(response)
```

## Contributing
If you'd like to contribute to OpenAI Promptify Py, please open an issue or pull request on our GitHub repository.

## License
OpenAI Promptify Py is licensed under the MIT License.
