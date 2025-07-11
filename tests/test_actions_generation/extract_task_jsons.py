import json

TASK_IDS = {
    "0578365e-d985-45e9-a2cf-aeffc4e6ac34",
    "4409d78f-b3d9-46d0-b4f0-2a8947c8b9cf",
    "2b0bf09b-9fb3-4acc-a9c1-546ea4039416",
}

input_path = "log_stream.jsonl"
output_path = "recovered_tasks.json"

found_tasks = []

with open(input_path, "r") as infile:
    for line in infile:
        try:
            data = json.loads(line)
            # Look for the task object or task ID in typical structures
            task = None

            if isinstance(data, dict) and "id" in data and data["id"] in TASK_IDS:
                task = data
            elif "task" in data and isinstance(data["task"], dict) and data["task"].get("id") in TASK_IDS:
                task = data["task"]

            if task:
                found_tasks.append(task)
        except json.JSONDecodeError:
            continue

with open(output_path, "w") as out:
    json.dump(found_tasks, out, indent=2)

print(f"✅ Recovered {len(found_tasks)} task(s) to {output_path}")
