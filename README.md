# STT - LLM - TTS

The purpose of this repository is to demonstrate the Speech to Text (STT) with Azure OpenAI transcribe model and Azure Speech, calling to a Azure OpenAI text model to answer the user's question providing the response in streaming, and the Text to Speech (TTS) capabilities with Azure OpenAI TTS model and Azure Speech. 

## Python scripts:
- stt-llm-tts.py: STT and TTS with Azure OpenAI but the LLM text model does not provide the answer in streaming
- stt-llm-tts_streaming.py: STT and TTS with Azure OpenAI but the LLM text model provides the answer in streaming
- azure_speech_demo.py: STT and TTS with Azure Speech service

## Prerequisites
+ An Azure subscription, with [access to Azure OpenAI](https://aka.ms/oai/access).
+ An Azure OpenAI service with the service name and an API key.
+ A deployment of GPT-4.1 or GPT-4o model on the Azure OpenAI Service.
+ A deployment of Transcribe model on the Azure OpenAI Service.
+ A deployment of TTS model on the Azure OpenAI Service.
+ An instance of Azure Speech service.

I used Python 3.12.10, [Visual Studio Code with the Python extension](https://code.visualstudio.com/docs/python/python-tutorial), and the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) to test the notebooks.

### Set up a Python virtual environment in Visual Studio Code

1. Open the Command Palette (Ctrl+Shift+P).
2. Search for **Python: Create Environment**.
3. Select **Venv**.
4. Select a Python interpreter. Choose 3.12 or later.

It can take a minute to set up. If you run into problems, see [Python environments in VS Code](https://code.visualstudio.com/docs/python/environments).

### Environment Configuration

Create a `.env` file in the root directory of your project with the following content. You can use the provided [`.env-sample`](.env-sample) as a template.

The needed libraries are specified in [requirements.txt](requirements.txt).
