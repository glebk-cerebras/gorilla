import json
import os
import time

from model_handler.constant import GORILLA_TO_OPENAPI
from model_handler.handler import BaseHandler
from model_handler.model_style import ModelStyle
from model_handler.utils import (
    convert_to_tool,
    convert_to_function_call,
    func_doc_language_specific_pre_processing,
)
from openai import OpenAI


class CerebrasHandler(BaseHandler):
    def __init__(self, model_name, temperature=0.001, top_p=1, max_tokens=1000) -> None:
        super().__init__(model_name, temperature, top_p, max_tokens)
        self.model_style = ModelStyle.OpenAI

        self.client = OpenAI(
            base_url="http://api.cerebras.ai/v1/",
            api_key=os.getenv("CEREBRAS_API_KEY"),
        )

    def inference(self, prompt, functions, test_category):
        functions = func_doc_language_specific_pre_processing(
            functions, test_category
        )

        models = {
            "cerebras-llama3.1-8b": "llama3.1-8b",
            "cerebras-llama3.1-70b": "llama3.1-70b",
        }

        if self.model_name not in models:
            raise ValueError(
                f"Model {self.model_name} not supported. Supported models are {models.keys()}"
            )

        cerebras_model_name = models[self.model_name]

        message = prompt
        oai_tool = convert_to_tool(
            functions, GORILLA_TO_OPENAPI, self.model_style, test_category
        )
        start_time = time.time()
        if len(oai_tool) > 0:
            response = self.client.chat.completions.create(
                messages=message,
                model=cerebras_model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                tools=oai_tool,
            )
        else:
            response = self.client.chat.completions.create(
                messages=message,
                model=self.cerebras_model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
        latency = time.time() - start_time
        try:
            result = [
                {func_call.function.name: func_call.function.arguments}
                for func_call in response.choices[0].message.tool_calls
            ]
        except:
            result = response.choices[0].message.content
        metadata = {}
        metadata["input_token_count"] = response.usage.prompt_tokens
        metadata["output_token_count"] = response.usage.completion_tokens
        metadata["latency"] = latency
        metadata["processed_message"] = message
        metadata["processed_tool"] = oai_tool

        return result, metadata
    
    def decode_ast(self, result, language="Python"):
        decoded_output = []
        for invoked_function in result:
            name = list(invoked_function.keys())[0]
            params = json.loads(invoked_function[name])
            decoded_output.append({name: params})
        return decoded_output

    def decode_execute(self, result):
        function_call = convert_to_function_call(result)
        return function_call
