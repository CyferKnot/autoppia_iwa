# llm_service.py


import httpx
import json
import re
import ast
from pathlib import Path
from openai import AsyncOpenAI, OpenAI
from datetime import datetime, timezone

from autoppia_iwa.src.llms.domain.interfaces import ILLM, LLMConfig
from autoppia_iwa.src.shared.log_writer import log_jsonl, write_log

class OpenAIService(ILLM):
    """
    Simple OpenAI-based LLM.
    Uses OpenAI (sync) and AsyncOpenAI (async) clients.
    """

    def __init__(self, config: LLMConfig, api_key: str):
        self.config = config
        self.sync_client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)

    def _prepare_payload(self, messages: list[dict[str, str]], json_format: bool = False, schema: dict | None = None) -> dict:
        """
        Prepares the payload for OpenAI API requests.
        """
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if json_format and schema:
            payload["response_format"] = {"type": "json_object"}
            # Add system message for JSON structure
            messages.insert(0, {"role": "system", "content": f"You must respond with JSON that matches this schema: {schema}"})
            payload["messages"] = messages
        return payload

    def predict(self, messages: list[dict[str, str]], json_format: bool = False, schema: dict | None = None, return_raw: bool = False) -> str:
        """
        Synchronous prediction using OpenAI's API.
        """
        try:
            payload = self._prepare_payload(messages, json_format, schema)
            response = self.sync_client.chat.completions.create(**payload)
            if return_raw:
                return response
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI Sync Error: {e}") from e

    async def async_predict(self, messages: list[dict[str, str]], json_format: bool = False, schema: dict | None = None, return_raw: bool = False) -> str:
        """
        Asynchronous prediction using OpenAI's API.
        """
        try:
            payload = self._prepare_payload(messages, json_format, schema)
            response = await self.async_client.chat.completions.create(**payload)
            if return_raw:
                return response
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI Async Error: {e}") from e


class LocalLLMService(ILLM):
    """
    Simple local (self-hosted) LLM that communicates via HTTP.
    Uses HTTPX for sync and async calls.
    """

    def __init__(self, config: LLMConfig, endpoint_url: str):
        """
        :param config: LLMConfig object with model details, max_tokens, temperature, etc.
        :param endpoint_url: The HTTP endpoint for single-request generation (e.g. /generate).
        """
        self.config = config
        self.endpoint_url = endpoint_url

    @staticmethod
    def clean_analysis_fields(data: dict) -> dict:
        # Curiosities
        if isinstance(data.get("curiosities"), list):
            data["curiosities"] = " ".join(str(x) for x in data["curiosities"])
        elif not isinstance(data.get("curiosities"), (str, type(None))):
            data["curiosities"] = str(data["curiosities"])

        # Functionality: stringified dicts or plain strings
        functionality = data.get("functionality")
        if isinstance(functionality, list):
            cleaned_functionality = []
            for item in functionality:
                if isinstance(item, dict):
                    cleaned_functionality.append(str(item.get("use_case") or item.get("description") or str(item)))
                elif isinstance(item, str):
                    try:
                        parsed = ast.literal_eval(item)
                        # Handle known keys
                        if isinstance(parsed, dict):
                            if "use_cases" in parsed:
                                cleaned_functionality.extend(str(x) for x in parsed["use_cases"])
                            elif "use_case" in parsed:
                                cleaned_functionality.append(str(parsed["use_case"]))
                            elif "description" in parsed:
                                cleaned_functionality.append(str(parsed["description"]))
                            else:
                                cleaned_functionality.append(str(parsed))
                        else:
                            cleaned_functionality.append(str(parsed))
                    except Exception:
                        cleaned_functionality.append(item)
                else:
                    cleaned_functionality.append(str(item))
            data["functionality"] = cleaned_functionality
        elif isinstance(functionality, str):
            data["functionality"] = [functionality]
        else:
            data["functionality"] = []

        # Accessibility
        accessibility = data.get("accessibility")
        if isinstance(accessibility, dict):
            data["accessibility"] = [str(v) for v in accessibility.values()]
        elif isinstance(accessibility, list):
            data["accessibility"] = [str(x) for x in accessibility]
        elif accessibility is not None and not isinstance(accessibility, list):
            data["accessibility"] = [str(accessibility)]

        # Key words
        key_words = data.get("key_words")
        if isinstance(key_words, list):
            data["key_words"] = [str(x) for x in key_words]
        elif key_words is not None:
            data["key_words"] = [str(key_words)]

        # media_files_description / relevant_fields
        for key in ["media_files_description", "relevant_fields"]:
            val = data.get(key)
            if val is not None and not isinstance(val, list):
                data[key] = None
            elif isinstance(val, list) and not all(isinstance(item, dict) for item in val):
                data[key] = None

        return data

    def _prepare_payload(
        self,
        messages: list[dict[str, str]],
        json_format: bool = False,
        schema: dict | None = None,
        system_prompt_override: str | None = None
    ) -> dict:
        prompt_text = "\n".join([m["content"] for m in messages if m["role"] == "user"])

        if system_prompt_override:
            system_prompt = system_prompt_override
        else:
            system_prompt_path = Path("/home/cefus/ai/prompts/json_analysis_prompt.txt")
            system_prompt = system_prompt_path.read_text()

        payload = {
            "model": "NousResearch/Hermes-3-Llama-3.1-8B",
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": prompt_text }
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "num_ctx": self.config.max_tokens
            }
        }

        if json_format:
            payload["format"] = "json"
        if schema:
            payload["schema"] = schema

        return payload


    def predict(self, messages: list[dict[str, str]], json_format: bool = False, schema: dict | None = None, return_raw: bool = False, system_prompt_override: str | None = None) -> str:
        try:
            with httpx.Client(timeout=180.0) as client:
                payload = self._prepare_payload(messages, json_format, schema, system_prompt_override)
                system_message = payload["messages"][0]["content"]
                user_prompt = payload["messages"][1]["content"]
                model_name = payload["model"]

                # Log the initial request
                log_jsonl({
                    "event": "llm_request",
                    "source": "llm_service.predict",
                    "model": model_name,
                    "system_prompt": system_message,
                    "user_prompt": user_prompt,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

                # Execute LLM request
                response = client.post(self.endpoint_url, json=payload)
                response.raise_for_status()
                response_data = response.json()
                raw_content = response_data["choices"][0]["message"]["content"]

                # Clean and parse LLM output
                parsed_json = None
                try:
                    cleaned_text = re.sub(r'//.*', '', raw_content)
                    cleaned_text = re.sub(r'^```json\s*|\s*```$', '', cleaned_text.strip(), flags=re.IGNORECASE | re.MULTILINE)

                    # # Manual sanity check before parsing
                    # if '"attributes": {' in cleaned_text and cleaned_text.count('"attributes": {') > 1:
                    #     raise ValueError("Invalid multiple attribute blocks detected")

                    parsed_json = json.loads(cleaned_text)

                except (json.JSONDecodeError, ValueError) as decode_err:
                    log_jsonl({
                        "event": "llm_response_parse_failure" if isinstance(decode_err, json.JSONDecodeError) else "llm_schema_validation_error",
                        "source": "llm_service.predict",
                        "error": str(decode_err),
                        "raw_response": raw_content,
                        "prompt": user_prompt,
                        "model": model_name,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    return {}

                if parsed_json is None:
                    return {}

                # Clean result before returning
                cleaned_content = self.clean_analysis_fields(parsed_json)

                # Log cleaned output
                print("[DEBUG] Cleaned LLM output:", json.dumps(cleaned_content, indent=2)[:1000])
                log_jsonl({
                    "event": "llm_response",
                    "source": "llm_service.predict",
                    "model": model_name,
                    "response": cleaned_content,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

                return json.dumps(cleaned_content)

        except httpx.HTTPError as http_err:
            write_log("HTTP error during LLM request", {
                "error": str(http_err),
                "url": self.endpoint_url
            })
            raise RuntimeError(f"Local LLM Sync Error: {http_err}") from http_err

        except Exception as ex:
            print("[ERROR] Unhandled exception in predict():", ex)
            write_log("Unexpected exception during predict()", {
                "error": str(ex)
            })
            return {}



    async def async_predict(self, messages: list[dict[str, str]], json_format: bool = False, schema: dict | None = None, return_raw: bool = False, system_prompt_override: str | None = None) -> str:
        """
        Asynchronous prediction using the local LLM endpoint.
        """
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = self._prepare_payload(messages, json_format, schema, system_prompt_override)
                response = await client.post(self.endpoint_url, json=payload)
                response.raise_for_status()
                json_data = response.json()

                content = ""
                try:
                    content = json_data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as e:
                    print(f"[ERROR] Could not extract content from LLM response: {e}")
                    log_jsonl({
                        "event": "llm_async_response_parse_failure",
                        "source": "llm_service.async_predict",
                        "error": str(e),
                        "raw_json_data": json_data,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })

                log_jsonl({
                    "event": "llm_async_response",
                    "source": "llm_service.async_predict",
                    "response": json_data,
                    "extracted_content": content[:300],  # preview
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

                return content
        except httpx.HTTPError as e:
            write_log("HTTP error in async_predict", {
                "error": str(e),
                "url": self.endpoint_url
            })
            raise RuntimeError(f"Local LLM Async Error: {e}") from e


class LLMFactory:
    """
    Simple factory to build the right LLM implementation
    based on the llm_type.
    """

    @staticmethod
    def create_llm(llm_type: str, config: LLMConfig, **kwargs) -> ILLM:
        if llm_type.lower() == "openai":
            return OpenAIService(config, api_key=kwargs.get("api_key"))
        elif llm_type.lower() == "local":
            return LocalLLMService(config, endpoint_url=kwargs.get("endpoint_url"))
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
