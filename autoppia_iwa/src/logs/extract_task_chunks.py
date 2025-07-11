import json

TASK_IDS = {
    "0578365e-d985-45e9-a2cf-aeffc4e6ac34",
    "4409d78f-b3d9-46d0-b4f0-2a8947c8b9cf",
    "2b0bf09b-9fb3-4acc-a9c1-546ea4039416"
}

with open("log_stream.jsonl") as f:
    entries = [json.loads(line) for line in f if line.strip()]

for task_id in TASK_IDS:
    found = False
    with open(f"task_{task_id[:4]}.json", "w") as out:
        for entry in entries:
            text = json.dumps(entry)
            if task_id in text:
                out.write(json.dumps(entry, indent=2) + "\n")
                found = True
    if found:
        print(f"✅ Extracted matches for {task_id}")
    else:
        print(f"⚠️  No entries found for {task_id}")
