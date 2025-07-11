import copy
import json
from typing import Any

from dependency_injector.wiring import Provide

from autoppia_iwa.src.di_container import DIContainer
from autoppia_iwa.src.llms.domain.interfaces import ILLM
from autoppia_iwa.src.web_analysis.domain.analysis_classes import LLMWebAnalysis
from autoppia_iwa.src.web_analysis.domain.classes import Element
from autoppia_iwa.src.web_analysis.domain.prompt_llm_template import PromptLLMTemplate

CONVERT_RESPONSE_TO_JSON_PROMPT = """
You are an expert JSON content reviewer tasked with analyzing the given RAW JSON/Unstructured
 segment of a webpage and providing a strictly valid JSON-formatted analysis.

Important Requirements:
- Return only one JSON object (no arrays, no multiple objects).
- The output must be valid JSON that can be directly parsed by `json.loads` without modification.
- Use double quotes for all keys and string values.
- Do not include trailing commas.
- Do not include any text or explanation outside of the JSON object.
- If something is not relevant, omit it entirely rather than returning empty lists or objects.
- Do not include comments or additional text outside the JSON.
- Do not include code fences (```).

If the input cannot be summarized into a valid JSON object, return an empty JSON object: {}.
"""


class WebLLMAnalyzer:
    def __init__(self, llm_service: ILLM = Provide[DIContainer.llm_service]):
        """
        Initialize the web page structure extractor with a start URL.

        Args:
            llm_service (ILLM): the model to extract data from.
        """
        self.llm_service: ILLM = llm_service

    def analyze_element(self, element: Element) -> LLMWebAnalysis:
        template = PromptLLMTemplate.get_instance_from_file(
            "config/prompts/web_analysis/analyze_element.txt",
            "config/schemas/web_analysis/analyze_element_schema.json",
            {
                "element": element.to_dict(),
            },
        )
        return self._analyze_prompt_template(template=template)

    def analyze_element_parent(self, element: Element, children_analysis: list) -> LLMWebAnalysis:
        element_without_children = copy.deepcopy(element)
        del element_without_children.children

        # Text analysis about the sub segment
        template = PromptLLMTemplate.get_instance_from_file(
            "config/prompts/web_analysis/analyze_element_parent.txt",
            "config/schemas/web_analysis/analyze_element_parent_schema.json",
            {
                "element_without_children": element.to_dict(),
                "children_analysis": children_analysis,
            },
        )
        return self._analyze_prompt_template(template=template)

    def summarize_web_page(self, domain: str, page_url: str, elements_analysis_result) -> LLMWebAnalysis:
        for element in elements_analysis_result:
            if isinstance(element.get("analysis"), LLMWebAnalysis):
                element["analysis"] = element["analysis"].model_dump()
        template = PromptLLMTemplate.get_instance_from_file(
            "config/prompts/web_analysis/analyze_page_url.txt",
            "config/schemas/web_analysis/analyze_page_url_schema.json",
            {
                "domain": domain,
                "page_url": page_url,
                "html_page_analysis": elements_analysis_result,
            },
        )
        return self._analyze_prompt_template(template=template)

    def _analyze_prompt_template(self, template: PromptLLMTemplate) -> LLMWebAnalysis | None:
        prompt = PromptLLMTemplate.clean_prompt(template.current_prompt)
        json_schema = template.get_schema()
        llm_message = self._create_llm_message(prompt)

        analysis: LLMWebAnalysis | None = None
        tries = 3
        for _i in range(tries):
            try:
                response: str | dict = self.llm_service.predict(llm_message, json_format=True, schema=json_schema)

                # If response is already a dict, skip parsing
                if isinstance(response, dict):
                    json_result = response
                else:
                    try:
                        json_result = self._parse_json_response(response)
                    except Exception as e:
                        print(f"Failed to parse response string as JSON: {e}")
                        return None

                # Unwrap if the LLM returned {"element": {...}} instead of flat object
                if isinstance(json_result, dict) and "element" in json_result:
                    json_result = json_result["element"]

                # Handle legacy edge case: relevant_fields was a dict not list
                if isinstance(json_result.get("relevant_fields"), dict):
                    rf = json_result["relevant_fields"]
                    json_result["relevant_fields"] = [{"type": k, "attributes": v} for k, v in rf.items()]

                try:
                    if isinstance(json_result, dict):
                        # Unwrap 'element' or 'analysis' if present
                        if "element" in json_result and isinstance(json_result["element"], dict):
                            json_result = json_result["element"]
                        if "analysis" in json_result and isinstance(json_result["analysis"], dict):
                            json_result = json_result["analysis"]

                    def sanitize_json_result(json_result: dict) -> dict:
                        def is_valid_str(v):
                            return isinstance(v, str) and v.strip() != ""

                        def is_valid_list_of_strs(v):
                            return isinstance(v, list) and all(isinstance(i, str) and i.strip() for i in v)

                        def is_valid_functionality(v):
                            return isinstance(v, dict) and all(is_valid_str(k) and is_valid_str(val) for k, val in v.items())

                        def clean_media_files(media):
                            if not isinstance(media, list):
                                return None
                            cleaned = []
                            for m in media:
                                if (
                                    isinstance(m, dict)
                                    and is_valid_str(m.get("description"))
                                    and is_valid_str(m.get("alt", ""))
                                    and is_valid_str(m.get("src", ""))
                                ):
                                    cleaned.append(m)
                            return cleaned or None

                        def clean_relevant_fields(fields):
                            if not isinstance(fields, list):
                                return None
                            cleaned = []
                            for f in fields:
                                if (
                                    isinstance(f, dict)
                                    and is_valid_str(f.get("type"))
                                    and isinstance(f.get("attributes"), list)
                                    and all(is_valid_str(attr) for attr in f["attributes"])
                                ):
                                    cleaned.append(f)
                            return cleaned or None

                        sanitized = {
                            "one_phrase_summary": json_result.get("one_phrase_summary") if is_valid_str(json_result.get("one_phrase_summary")) else "",
                            "summary": json_result.get("summary") if is_valid_str(json_result.get("summary")) else "",
                            "categories": json_result.get("categories") if is_valid_list_of_strs(json_result.get("categories")) else [],
                            "functionality": json_result.get("functionality") if is_valid_functionality(json_result.get("functionality")) else {},
                            "media_files_description": clean_media_files(json_result.get("media_files_description")),
                            "key_words": json_result.get("key_words") if is_valid_list_of_strs(json_result.get("key_words")) else [],
                            "relevant_fields": clean_relevant_fields(json_result.get("relevant_fields")),
                            "curiosities": json_result.get("curiosities") if is_valid_str(json_result.get("curiosities")) else None,
                            "accessibility": json_result.get("accessibility") if is_valid_list_of_strs(json_result.get("accessibility")) else None
                        }

                        return sanitized

                    def sanitize_functionality(value):
                        result = {}
                        if isinstance(value, dict):
                            for k, v in value.items():
                                if isinstance(k, str):
                                    if isinstance(v, str):
                                        result[k] = [v.strip()]
                                    elif isinstance(v, list):
                                        result[k] = [str(item).strip() for item in v if isinstance(item, str)]
                        elif isinstance(value, list):
                            # Merge list of dicts (common LLM hallucination)
                            for item in value:
                                if isinstance(item, dict):
                                    for k, v in item.items():
                                        if isinstance(k, str):
                                            if isinstance(v, str):
                                                result.setdefault(k, []).append(v.strip())
                                            elif isinstance(v, list):
                                                result.setdefault(k, []).extend(str(i).strip() for i in v if isinstance(i, str))
                        return result



                    # Unwrap 'element' or 'analysis' if present — do this BEFORE sanitizing
                    if isinstance(json_result, dict):
                        if "element" in json_result and isinstance(json_result["element"], dict):
                            json_result = json_result["element"]
                        if "analysis" in json_result and isinstance(json_result["analysis"], dict):
                            json_result = json_result["analysis"]

                    # Now sanitize the final unwrapped dictionary
                    json_result = sanitize_json_result(json_result)

                    # Sanity patch for "functionality" again after sanitization (just in case)
                    if "functionality" in json_result:
                        json_result["functionality"] = sanitize_functionality(json_result["functionality"])


                    analysis = LLMWebAnalysis(**json_result)
                except Exception as e:
                    print(f"Error while parsing llm response into LLMWebAnalysis instance: {e}")
                    return None
                break
            except Exception as e:
                print(f"Error while parsing llm response into LLMWebAnalysis instance: {e}")
        return analysis

    @staticmethod
    def _create_llm_message(prompt: str, system_instructions: str = CONVERT_RESPONSE_TO_JSON_PROMPT) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_instructions.strip()},
            {"role": "user", "content": prompt.strip()},
        ]

    @staticmethod
    def _parse_json_response(response: str) -> dict[Any, Any]:
        """Parses a JSON response from the LLM."""
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}") from e
