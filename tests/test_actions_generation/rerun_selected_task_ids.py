import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import asyncio
import json
import logging
from collections import Counter

from autoppia_iwa.src.bootstrap import AppBootstrap
from autoppia_iwa.src.web_agents.my_miner_class import MyMinerClass
from autoppia_iwa.src.data_generation.domain.classes import Task

CACHE_PATH = "./tmp_tasks_cache/autoppia_cinema_tasks.json"
LOG_PATH = "unconfirmed_actions_rerun.log"

# 👇 Paste your selected task IDs here
TASK_IDS_TO_RERUN = [
    "0578365e-d985-45e9-a2cf-aeffc4e6ac34",
    "235f58cc-aeb2-47af-979b-6a19e2056690",
    "6cdfffc8-3645-4667-8b11-46229dc75ea3",
    "1c92ebd6-6697-4399-9fbf-0639b803afbe",
    "e81a1625-b68e-4cce-81d7-26e7964fb2c4",
    "4409d78f-b3d9-46d0-b4f0-2a8947c8b9cf",
    "2b0bf09b-9fb3-4acc-a9c1-546ea4039416",
    # etc...
]

UNCONFIRMED_ACTIONS = {
    "AssertAction",
    "SendKeysIWAAction",
    "SelectDropDownOptionAction",
    "DragAndDropAction",
    "UndefinedAction",
}

action_type_counts = Counter()

async def run_selected_tasks():
    logging.basicConfig(level=logging.INFO)
    print(f"📂 Loading tasks from {CACHE_PATH}...")

    if not os.path.exists(CACHE_PATH):
        print(f"❌ Task cache not found: {CACHE_PATH}")
        return

    with open(CACHE_PATH, "r") as f:
        cache_data = json.load(f)

    tasks_raw = cache_data.get("tasks", [])
    selected_tasks = [Task.deserialize(t) for t in tasks_raw if t["id"] in TASK_IDS_TO_RERUN]

    if not selected_tasks:
        print("⚠️ No matching tasks found.")
        return

    bootstrap = AppBootstrap()
    llm_service = bootstrap.container.llm_service()
    web_agent = MyMinerClass(llm_service=llm_service)

    print(f"🔁 Rerunning {len(selected_tasks)} selected tasks...")

    with open(LOG_PATH, "w") as log_file:
        for i, task in enumerate(selected_tasks, 1):
            print(f"\n🔁 Task {i}/{len(selected_tasks)} → {task.prompt}")
            try:
                solution = await web_agent.solve_task_sync(task)
                print(f"✅ Solution for Task {task.id}: {solution}")

                if solution and solution.actions:
                    print(f"🔎 Checking actions for task {task.id}...")
                    for idx, action in enumerate(solution.actions, 1):
                        action_type = action.type
                        action_type_counts[action_type] += 1
                        if action_type in UNCONFIRMED_ACTIONS:
                            msg = f"⚠️  [UNCONFIRMED ACTION] Task {task.id} #{idx} → {action}"
                            print(msg)
                            log_file.write(msg + "\n")

            except Exception as e:
                print(f"❌ Error solving task {task.id}: {e}")

    print("\n📊 Action Type Summary:")
    for action_type, count in action_type_counts.items():
        print(f"- {action_type}: {count}")

if __name__ == "__main__":
    asyncio.run(run_selected_tasks())
