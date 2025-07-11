import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import asyncio
import json

from autoppia_iwa.src.bootstrap import AppBootstrap
from autoppia_iwa.src.demo_webs.config import demo_web_projects
from autoppia_iwa.src.demo_webs.utils import initialize_demo_webs_projects
from autoppia_iwa.src.web_agents.my_miner_class import MyMinerClass
from autoppia_iwa.src.evaluation.evaluator.evaluator import ConcurrentEvaluator
from autoppia_iwa.src.evaluation.classes import EvaluatorConfig
from autoppia_iwa.src.shared.utils_entrypoints.tasks import generate_tasks_for_project
from autoppia_iwa.src.web_agents.classes import TaskSolution
import logging

from pydantic.json import pydantic_encoder
from autoppia_iwa.src.evaluation.evaluator.evaluator import ConcurrentEvaluator
from autoppia_iwa.src.evaluation.classes import EvaluatorConfig
from collections import Counter


CACHE_PATH = "./tmp_tasks_cache/autoppia_cinema_tasks.json"
NUM_TASKS = 3
action_type_counts = Counter()

async def run_bulk_evaluation(num_tasks=NUM_TASKS):
    print("🔄 Generating tasks for Autoppia Cinema...")

    # Ensure cache directory exists
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    bootstrap = AppBootstrap()
    llm_service = bootstrap.container.llm_service()
    web_agent = MyMinerClass(llm_service=llm_service)
    # Load demo projects
    projects = await initialize_demo_webs_projects(demo_web_projects)
    demo_project = projects[0]  # Use the first demo project

    # Generate tasks
    tasks = await generate_tasks_for_project(
        demo_project=demo_project,
        use_cached_tasks=False,
        task_cache_dir="./tmp_tasks_cache",
        prompts_per_url=3,
        num_of_urls=1,
    )

    if not tasks:
        print("❌ No tasks generated.")
        return

    try:
        with open(CACHE_PATH, "w") as f:
            json.dump([task.model_dump(mode="json") for task in tasks], f, indent=2, default=pydantic_encoder)
        print(f"✅ Tasks saved to {CACHE_PATH}")
    except Exception as e:
        print(f"⚠️ Error saving tasks to {CACHE_PATH}: {e}")

    for i, task in enumerate(tasks, 1):
        print(f"\n🔁 Running Task {i}/{len(tasks)} → {task.prompt}")
        try:
            solution = await web_agent.solve_task_sync(task)
            print(f"✅ Solution for Task {task.id}: {solution}")

            # Define action types you want to track specifically
            UNCONFIRMED_ACTIONS = {
                "AssertAction",
                "SendKeysIWAAction",
                "SelectDropDownOptionAction",
                "DragAndDropAction",
                "UndefinedAction",
            }

            # Print and flag any unconfirmed action types in this task
            if solution and solution.actions:
                print(f"🔎 Checking actions for task {task.id}...")
                with open("unconfirmed_actions.log", "a") as log_file:
                    for idx, action in enumerate(solution.actions, 1):
                        action_type = action.type
                        action_type_counts[action.type] += 1
                        if action_type in UNCONFIRMED_ACTIONS:
                            msg = f"⚠️  [UNCONFIRMED ACTION] Task {task.id} #{idx} → {action}"
                            print(msg)
                            log_file.write(msg + "\n")

                print("\n📊 Action Type Summary:")
                for action_type, count in action_type_counts.items():
                    print(f"- {action_type}: {count}")

        except Exception as e:
            print(f"❌ Error solving task {task.id}: {e}")

if __name__ == "__main__":
    asyncio.run(run_bulk_evaluation(NUM_TASKS))
    
    print("\n📊 Action Type Summary:")
    for action_type, count in action_type_counts.items():
        print(f"- {action_type}: {count}")