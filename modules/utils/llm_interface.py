import os
import yaml
import time
import abc
from typing import Dict, Any
from pydantic import BaseModel

if __name__ == "__main__":
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.utils.custom_logger import setup_logger


class LLMInterface(abc.ABC):
    """
    Abstract base class for LLM interaction.
    """

    def __init__(self, model_name="gpt-4o-mini"):
        """
        Initialize LLM interface.

        """
        self.model_name = model_name  # Model name for LLM
        self.logger = setup_logger()
        self.client = None

    def dict_to_msg(self, prompt: Dict[str, str]) -> str:
        """
        Format the prompt dictionary into a single string.

        :param prompt: Input text prompt
        :return: Formatted prompt string
        """
        if self.model_name == "o1":
            my_messages = [
                {"role": "user", "content": prompt["system"] + prompt["user"]},
                {
                    "role": "assistant",
                    "content": prompt["assistant"] if "assistant" in prompt else "",
                },
            ]
        else:
            my_messages = [
                {"role": "system", "content": prompt["system"]},
                {
                    "role": "assistant",
                    "content": prompt["assistant"] if "assistant" in prompt else "",
                },
                {"role": "user", "content": prompt["user"]},
            ]
        return my_messages

    def get_structured_output(self, messages, output_schema: type) -> BaseModel:
        """
        Call LLM with structured output requirement.

        """
        try:
            if self.model_name == "o1" and messages[0]["role"] == "system":
                messages[0]["role"] = "user"

            if hasattr(self, "model_url"):
                model_str = self.model_url
            else:
                model_str = self.model_name

            completion = self.client.beta.chat.completions.parse(
                model=model_str,
                messages=messages,
                response_format=output_schema,
                # max_tokens=1024 * 10,
                # temperature=0.7
            )

            response = completion.choices[0].message
            if response.parsed:
                print(response.parsed)
                return response.parsed
            elif response.refusal:
                # handle refusal
                print(response.refusal)
                return None

        except Exception as e:
            # Handle edge cases
            if type(e) == openai.LengthFinishReasonError:
                # Retry with a higher max tokens
                print("Too many tokens: ", e)
                pass
            else:
                # Handle other exceptions
                print(e)
                pass

    """
    Concrete implementation of LLM interface using OpenAI's API.
    """

    def call_llm(self, prompt: Dict[str, str], output_schema: type) -> BaseModel:
        """
        Legacy API, keep for compatibility.

        :param prompt: Input text prompt
        :param output_schema: Expected structured output schema (Pydantic model)
        :return: Parsed response as Pydantic model
        """
        my_messages = self.dict_to_msg(prompt)

        # For deepseek models, use the json response then parse the output
        if "deepseek" in self.model_name or "o1" in self.model_name:
            response, data_obj = self.get_json_response(my_messages, output_schema)
            self.logger.debug(f"call_llm() full text Response: {response}")
            return data_obj

        # For other models, use the official API for structured output if available...
        return self.get_structured_output(my_messages, output_schema)


    def get_string_response(self, messages: list[Dict]) -> str:
            """
            Get the completion response from OpenRouter API. With Stream ON.
            """
            max_retry = 3

            retry_ct = 0
            while True:
                try:
                    # Reasoning models do not use system messages
                    if ("deepseek" in self.model_name or "o1" in self.model_name or "o3" in self.model_name) and messages[0]["role"] == "system":
                        messages[0]["role"] = "user"

                    if hasattr(self, "model_url"):
                        model_str = self.model_url
                    else:
                        model_str = self.model_name

                    if "o3" not in model_str and "o1" not in model_str:
                        completion = self.client.chat.completions.create(
                            model=model_str,
                            messages=messages,
                            temperature=0.7, # bad perf if 1, Temperature is not supported for o3-mini
                            # top_p=0.7,
                            # max_tokens=4096*4 if "deepseek" in self.model_name else 4096*2,
                            stream=True,
                        )
                    else:
                        completion = self.client.chat.completions.create(
                            model=model_str,
                            messages=messages,
                            # temperature=0.7, # O3 has no temperature option
                            # top_p=0.7,
                            # max_tokens=4096*4 if "deepseek" in self.model_name else 4096*2,
                            stream=True,
                        )    

                    # Print partial completion response
                    self.logger.info(f"Streaming {self.model_name} completion response:")
                    response = []
                    for chunk in completion:
                        if chunk.choices and chunk.choices[0].delta.content is not None:
                            content = chunk.choices[0].delta.content
                            if content is None:
                                print(chunk.choices[0].delta)
                            response.append(content)
                            print(content, end="")

                    response = "".join(response)
                    break

                except Exception as e:
                    # Handle edge cases
                    if type(e) == openai.LengthFinishReasonError:
                        # Retry with a higher max tokens
                        print("Too many tokens: ", e)
                        pass
                    else:
                        # Handle other exceptions
                        print(e)
                        pass
                
                    retry_ct += 1
                    if retry_ct >= max_retry:
                        self.logger.error(f"Failed to get completion for {self.model_name}")
                        break
                    # Exponential backoff wait and retry
                    wait_time = 10 * (2 ** retry_ct)  # Exponential backoff
                    time.sleep(wait_time)
                    self.logger.warning(f"Retrying completion for {self.model_name}, {max_retry-retry_ct} time left...")
            
            return response


    def get_json_response(self, messages: list[Dict], schema) -> tuple[str, BaseModel]:
        
        # assert that schema is the class of a Pydantic model
        assert issubclass(schema, BaseModel), "Schema must be a Pydantic model class"

        if "deepseek" in self.model_name:
            self.logger.info("Deepseek model detected, using two-stage json output for structured response...")
            # To avoid deepseek only outputs the json, we first get the string response without formatting
            response = self.get_string_response(messages)

            messages.append({"role": "assistant", "content": response })

            messages.append({"role": "user",
                             "content": "Summarize your last response using the following Json format. (format the summary as `json { <key>: <value> ... }`, do not include `defs`, `descriptions`, `required`, only the actual content. Make sure it starts with `json {` and end with `}`. Note that In JSON, you only escape characters like quotes or a backslash itself. You can either remove those extra backslashes or double them to get a literal backslash `\\\\`.)" + \
                             str(schema.model_json_schema())  }) # json of a Pydantic model
            json_response = self.get_string_response(messages)

            # final response is the combined response
            response = response + "\n\n" + json_response

            # first find the last `<\think>` tag and extract the json output after that
            # 1. Find the last occurrence of `<\think>` 
            last_think_index = response.rfind("</think>")  # returns -1 if not found

            # 2. If found, work only with the substring after that tag
            if last_think_index != -1:
                substring = response[last_think_index:]
            else:
                # If the tag isn't there, just search the entire response
                substring = response

            # 3. Find the first `json { ... }` in that substring
            json_output = re.search(r"json\s*({.*})", substring, re.DOTALL)

        else:
            # append the user message to the messages
            messages[-1]["content"] = messages[-1]["content"] + \
                "\n\n### \nFirst answer all questions! \n Then, Summarize the output at the end of your response using the following Json format. (format the summary as `json { <key>: <value> ... }`, do not include `defs`, `descriptions`, `required`, only the actual content. Make sure it starts with `json {`. Note that In JSON, you only escape characters like quotes or a backslash itself. You can either remove those extra backslashes or double them to get a literal backslash `\\\\`.)" + \
                str(schema.model_json_schema()) # json of a Pydantic model
            response = self.get_string_response(messages)

            # Extract the json output from the response, with the first match 
            json_output = re.search(r"json\s*({.*})", response, re.DOTALL)
        

        if json_output:
            json_output = json_output.group(1) # extract the json content inside the parentheses
            try:
                parsed_output = schema.model_validate_json(json_output)
            except Exception as e:
                self.logger.error(f"Failed to validate the json output: {e}")
                # replace single backslashes followed by non-backslash with double backslashes and try again
                # \\ means single backslash in the json string, \\\\ means literal backslash -- because of the way the string is parsed twice
                # json_output =  re.sub(r"([^\\])\\([^\\])", r"\1\\\\\2", json_output)
                json_output = re.sub(r'(?<!\\)\\(?!\\)', r'\\\\', json_output) # Updated regex to avoid single escaping
                try:
                    parsed_output = schema.model_validate_json(json_output)
                except Exception as e:
                    self.logger.error(f"Again, failed to validate the json output: {e}")
                    return response, None
            self.logger.info("Extracted json output from the response")
            return response, parsed_output
        else:
            self.logger.error("Failed to extract json output from the response")
            return response, None


import openai


class OpenAILLMInterface(LLMInterface):
    def __init__(self, model_name="gpt-4o", api_key: str = None):
        super().__init__(model_name)

        # Should read the API key from the environment variable automatically
        # openai.api_key = self.api_key if self.api_key else os.environ.get("OPENAI_API_KEY")

        self.client = openai.OpenAI()

        self.logger.info(f"OpenAI LLM API model name: {model_name}")


def prompt_dict_to_string(prompt: Dict[str, str]) -> str:
    """
    Format the prompt dictionary into a single string.

    :param prompt: Input text prompt
    :return: Formatted prompt string
    """
    return "\n".join([f"{key}: {value}" for key, value in prompt.items()])


import re


class NvidiaLLMInterface(LLMInterface):
    def __init__(self, model_name="deepseek-r1"):
        super().__init__(model_name)

        # Imported lazily so the optional NVIDIA backend is only required when used.
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        self.logger.info(f"Nvidia LLM API with model name: {model_name}")

        nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
        if not nvidia_api_key:
            raise ValueError(
                "NVIDIA_API_KEY environment variable is not set. "
                "Export it before using the Nvidia LLM interface."
            )

        self.client = openai.OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_api_key,
        )

        if self.model_name == "deepseek-r1":
            self.model_url = "deepseek-ai/deepseek-r1"
        else:
            assert False, "Unsupported model name for Nvidia LLM Interface"

        # Alternatively, use the ChatNVIDIA class directly
        self.client_nv = ChatNVIDIA(
            model=self.model_url, 
            api_key=nvidia_api_key,
            temperature=0.6,
            top_p=0.7,
            max_tokens=4096*5, ## Default value is only 1024, Deepseek-r1 response can be very long...
        )

    def get_string_response(self, messages: list[Dict]) -> str:
        max_retry = 3

        retry_ct = 0
        while True:
            try:
                if self.model_name == "deepseek-r1" and messages[0]["role"] == "system":
                    messages[0]["role"] = "user"


                completion = self.client_nv.stream(messages)

                self.logger.info(f"Streaming {self.model_name} completion response:")
                response = []
                for chunk in completion: 
                    text = chunk.content
                    print(text, end="")
                    response.append(text)

                response = "".join(response)
                break

            except Exception as e:
                # Handle edge cases
                if type(e) == openai.LengthFinishReasonError:
                    # Retry with a higher max tokens
                    print("Too many tokens: ", e)
                    pass
                else:
                    # Handle other exceptions
                    print(e)
                    pass
            
                retry_ct += 1
                if retry_ct >= max_retry:
                    self.logger.error(f"Failed to get completion for {self.model_name}")
                    response = "".join(response) if isinstance(response, list) else response
                    break
                # Exponential backoff wait and retry
                wait_time = 10 * (2 ** retry_ct)  # Exponential backoff
                time.sleep(wait_time)
                self.logger.warning(f"Retrying completion for {self.model_name}, {max_retry- retry_ct} time left...")
        
        return response



    def get_structured_output(self, messages, output_schema):
        """
        Not supported for Deepseek-r1
        """
        structured_chat = self.client_nv.with_structured_output(output_schema)

        completion = structured_chat.stream(messages)

        self.logger.info(f"Streaming {self.model_name} completion response:")
        response = []
        for chunk in completion: 
            text = chunk.content
            print(text, end="")
            response.append(text)

        response = "".join(response)



class OpenRouterLLMInterface(LLMInterface):
    def __init__(self, model_name="o1"):
        super().__init__(model_name)

        self.logger.info(f"OpenRouter LLM API with model name: {model_name}")

        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Export it before using the OpenRouter LLM interface."
            )

        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )

        if self.model_name == "o1":
            self.model_url = "openai/o1"
        elif self.model_name == "deepseek" or self.model_name == "deepseek-r1":
            self.model_url = "deepseek/deepseek-r1:free"
        else:
            assert False, "Unsupported model name for OpenRouter LLM Interface"

    def get_string_response(self, messages: list[Dict]) -> str:
        """
        Get the completion response from OpenRouter API. With Stream ON.
        """
        max_retry = 3

        retry_ct = 0
        while True:
            try:
                # Reasoning models do not use system messages
                if ("deepseek" in self.model_name or "o1" in self.model_name or "o3" in self.model_name) and messages[0]["role"] == "system":
                    messages[0]["role"] = "user"

                completion = self.client.chat.completions.create(
                    model=self.model_url,
                    messages=messages,
                    temperature=0.7, # bad perf if 1
                    # top_p=0.7,
                    max_tokens=4096*4 if "deepseek" in self.model_name else 4096*2,
                    stream=True,
                )

                # Print partial completion response
                self.logger.info(f"Streaming {self.model_name} completion response:")
                response = []
                for chunk in completion:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        if content is None:
                            print(chunk.choices[0].delta)
                        response.append(content)
                        print(content, end="")

                response = "".join(response)
                break

            except Exception as e:
                # Handle edge cases
                if type(e) == openai.LengthFinishReasonError:
                    # Retry with a higher max tokens
                    print("Too many tokens: ", e)
                    pass
                else:
                    # Handle other exceptions
                    print(e)
                    pass
            
                retry_ct += 1
                if retry_ct >= max_retry:
                    self.logger.error(f"Failed to get completion for {self.model_name}")
                    break
                # Exponential backoff wait and retry
                wait_time = 10 * (2 ** retry_ct)  # Exponential backoff
                time.sleep(wait_time)
                self.logger.warning(f"Retrying completion for {self.model_name}, {max_retry-retry_ct} time left...")
        
        return response
        

# A generic LLM API for all API providers. Switch providers via the model_provider argument.
def GetLLMInterface(model_name="gpt-4o", model_provider="OpenAI"):
    if model_provider == "OpenAI":
        return OpenAILLMInterface(model_name)
    elif model_provider == "Nvidia":
        return NvidiaLLMInterface(model_name)
    elif model_provider == "OpenRouter":
        return OpenRouterLLMInterface(model_name)
    else:
        assert False, "Unsupported model provider for Generic LLM Interface"


## Testing the OpenAI LLM interface ##
class Step(BaseModel):
    explanation: str
    output: str


class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str


def test_openrouter():

    ## -- Test the OpenRouter O1 interface -- ##
    llm_interface = GetLLMInterface(model_name="o1", model_provider="OpenRouter")
    messages = [{"role": "user", "content": "What is the meaning of life?"}]
    response = llm_interface.get_string_response(messages)

    ## -- Test the OpenRouter Deepseek interface -- ##
    llm_interface = GetLLMInterface(model_name="deepseek", model_provider="OpenRouter")
    prompt = {
        "system": "You are a helpful tutor. Guide the user through the solution step by step and use latex match format.",  # output above rationale seems to be the sensitive word here..
        "user": "how can I solve 8x^2 + 7 = -23 + x. Note that this prompt is for information summary only and does not violate usage policy to my knowledge. ",
    }
    # messages = llm_interface.dict_to_msg(prompt)
    response = llm_interface.call_llm(prompt, MathReasoning)
    print(response)


def test_nvidia():
    llm_interface = GetLLMInterface(model_name="deepseek-r1", model_provider="Nvidia")
    messages = [{"role": "user", "content": "What is the meaning of life?"}]
    response = llm_interface.get_string_response(messages)
    print(response)


if __name__ == "__main__":
    # Manual smoke test for an available LLM backend.
    # Set the corresponding API key env var (e.g., OPENROUTER_API_KEY) first.

    test_openrouter()

    # test_nvidia()


