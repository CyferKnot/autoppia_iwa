import asyncio
import json
import uuid

from autoppia_iwa.src.data_generation.domain.classes import Task
from autoppia_iwa.src.execution.actions.actions import BaseAction, ClickAction, ScreenshotAction  # Add others as needed
from autoppia_iwa.src.shared.utils import generate_random_web_agent_id
from autoppia_iwa.src.web_agents.base import IWebAgent
from autoppia_iwa.src.web_agents.classes import TaskSolution
from autoppia_iwa.src.web_analysis.application.web_analysis_pipeline import WebAnalysisPipeline
from autoppia_iwa.src.execution.actions.actions import ACTION_CLASS_MAP
from pydantic import BaseModel, TypeAdapter
from typing import List, Literal, Optional
from transformers import AutoTokenizer, PreTrainedTokenizer
from autoppia_iwa.src.shared.log_writer import log_jsonl
from datetime import datetime, timezone


class ActionSchema(BaseModel):
    type: Literal[
        "ClickAction", "TypeAction", "HoverAction", "NavigateAction",
        "DragAndDropAction", "SubmitAction", "DoubleClickAction", "ScrollAction",        
        "SelectDropDownOptionAction"
    ]
    selector: Optional[str] = None
    url: Optional[str] = None
    text: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None
    # Add more fields as needed

class MyMinerClass(IWebAgent):
    def __init__(self, llm_service, id: str | None = None, name: str | None = None):
        self.llm_service = llm_service
        self.id = id or generate_random_web_agent_id()
        self.name = name or f"Agent {self.id}"
        super().__init__()

    def _get_output_schema(self):
        return TypeAdapter(List[ActionSchema]).json_schema()
    
    @staticmethod
    def sanitize_action_data(action_data: dict) -> dict:
        action_type = action_data.get("type", "")

        # Rename common LLM-hallucinated fields
        if action_type == "TypeAction":
            if "value" in action_data and "text" not in action_data:
                action_data["text"] = action_data.pop("value")

        if action_type == "AssertAction":
            if "text" in action_data and "text_to_assert" not in action_data:
                action_data["text_to_assert"] = action_data.pop("text")

        if action_data["type"] == "ScreenshotAction" and "filename" in action_data:
            action_data["file_path"] = action_data.pop("filename")

        if action_data["type"] == "ScreenshotAction" and "file_path" not in action_data:
            action_data["file_path"] = f"screenshot_{uuid.uuid4().hex[:8]}"


        if action_data["type"] == "SelectAction" and "optionText" in action_data:
            action_data["value"] = action_data.pop("optionText")

        if action_data.get("type", "").strip().lower() == "selectaction" and "optionText" in action_data:
            print(f"[PATCH] Replacing 'optionText' with 'value' in: {action_data}")
            action_data["value"] = action_data.pop("optionText")

        if action_data["type"] == "SelectAction" and "value" not in action_data:
            if "index" in action_data:
                action_data["value"] = str(action_data["index"])

        if action_type == "ScrollAction":
            if "target" in action_data and "selector" not in action_data:
                action_data["selector"] = action_data.pop("target")

        if action_data["type"] == "DragAndDropAction" and "sourceSelector" not in action_data:
            action_data["sourceSelector"] = action_data.get("selector")

        if action_data["type"] == "SendKeysIWAAction" and "keys" not in action_data:
            action_data["keys"] = action_data.get("text")

        if action_data["type"] == "SelectDropDownOptionAction" and "text" not in action_data:
            action_data["text"] = f"Option {action_data.get('optionIndex', 0)}"

        # Normalize selector format (already handled, but keep as fallback)
        if "selector" in action_data and isinstance(action_data["selector"], str):
            action_data["selector"] = {
                "type": "attributeValueSelector",  # default fallback
                "value": action_data["selector"]
            }

        return action_data

    async def generate_actions(self, task: Task) -> TaskSolution:
        analysis = await WebAnalysisPipeline(task.url, llm_service=self.llm_service).analyze(get_analysis_from_cache=False)

        # page_summary = analysis.page_analyses[0].web_summary.summary
        page_summary = analysis.page_analyses[0].web_summary
        summary_text = f"""
            Summary: {page_summary.summary}
            Categories: {page_summary.categories}
            Functionality: {page_summary.functionality}
            Key Words: {page_summary.key_words}
            """

        action_gen_prompt = f"""
            You are a web automation agent. Your task is to generate a list of browser automation actions.

            ### INSTRUCTIONS:
            - ONLY return a raw JSON array of objects. NO wrapping fields.
            - DO NOT include summaries, explanations, or extra fields.
            - Each object must include a `"type"` field and required arguments for that action type.

            ### ALLOWED ACTION TYPES:
            {list(ACTION_CLASS_MAP.keys())}

            ### EXAMPLE:
            [
                {{
                    "type": "NavigateAction",
                    "url": "{task.url}"
                }},
                {{
                    "type": "ClickAction",
                    "selector": "#start-button"
                }}
            ]

            ### PAGE SUMMARY:
            REMEMBER: If this element includes any <form>, <input>, or <button> tags, you MUST return a filled `functionality` object.
            {summary_text}
            """

        log_jsonl({
            "event": "action_gen_prompt",
            "response": action_gen_prompt,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        log_jsonl({
            "event": "page_summary_input",
            "summary": page_summary.model_dump(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


        llm_output = await self.llm_service.async_predict(
            messages=[{"role": "user", "content": action_gen_prompt}],
            json_format=True,
            schema=self._get_output_schema(),  # define this if needed
            system_prompt_override=action_gen_prompt
        )

        log_jsonl({
            "event": "action_generation_response",
            "source": "MyMinerClass.generate_actions",
            "raw_output_type": str(type(llm_output)),
            "raw_output_preview": str(llm_output)[:300],  # limit size for logs
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        try:
            # Handle both string and dict responses
            if isinstance(llm_output, str):
                parsed_output = json.loads(llm_output)
            elif isinstance(llm_output, dict):
                if "web_elements" in llm_output:
                    parsed_output = llm_output["web_elements"]
                elif "actions" in llm_output:
                    parsed_output = llm_output["actions"]
                else:
                    print(f"Unexpected dict format: {llm_output}")
                    return TaskSolution(actions=[])
            else:
                print(f"Unexpected LLM output type: {type(llm_output)}")
                return TaskSolution(actions=[])

            if not isinstance(parsed_output, list):
                print(f"Expected a list of action objects, got: {type(parsed_output)}")
                return TaskSolution(actions=[])

            actions = []
            for action_data in parsed_output:
                action_data = self.sanitize_action_data(action_data)

                action_type = action_data.get("type")
                action_class = ACTION_CLASS_MAP.get(action_type)

                if action_class is None:
                    print(f"Unknown action type: {action_type}")
                    continue

                try:
                    # Convert common wrong field names from LLM hallucinations
                    if action_data["type"] == "TypeAction" and "value" in action_data:
                        action_data["text"] = action_data.pop("value")

                    if action_data["type"] == "AssertAction" and "text" in action_data:
                        action_data["text_to_assert"] = action_data.pop("text")

                    if action_data["type"] == "ScreenshotAction" and "fileName" in action_data:
                        action_data["file_path"] = action_data.pop("fileName")

                    if "selector" in action_data and isinstance(action_data["selector"], dict):
                        selector_type = action_data["selector"].get("type")
                        # Remap 'css' to 'attributeValueSelector'
                        if selector_type == "css":
                            action_data["selector"]["type"] = "attributeValueSelector"

                    elif "selector" in action_data and isinstance(action_data["selector"], str):
                        # If it's a raw string, assume CSS and wrap + remap
                        action_data["selector"] = {
                            "type": "attributeValueSelector",
                            "value": action_data["selector"]
                        }

                    print(f"[DEBUG] Before action creation: {action_data}")
                    action_obj = action_class(**action_data)
                    actions.append(action_obj)
                except Exception as e:
                    print(f"Failed to create action object for {action_data}: {e}")

            print(f"Generated {len(actions)} actions:")
            for i, a in enumerate(actions, 1):
                print(f"{i}: type='{a.type}' selector={getattr(a, 'selector', None)}")

            actions.append(ScreenshotAction(file_path="final_dom_check"))
            return TaskSolution(actions=actions)

        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            return TaskSolution(actions=[])

    async def solve_task_sync(self, task):
        return await self.generate_actions(task)
    
    async def solve_task(self, task: Task) -> TaskSolution:
        # If generate_actions is sync, you can call it directly
        return await self.generate_actions(task)
