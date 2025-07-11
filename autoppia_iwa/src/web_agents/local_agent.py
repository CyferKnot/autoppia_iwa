from autoppia_iwa.src.data_generation.domain.classes import Task
from autoppia_iwa.src.web_agents.base import IWebAgent
from autoppia_iwa.src.web_agents.classes import TaskSolution


class LocalWebAgent(IWebAgent):
    def __init__(self, llm_service):
        self.llm = llm_service

    def solve_task_sync(self, task: Task) -> TaskSolution:
        messages = [{"role": "user", "content": task.prompt}]
        
        # Manually include model param expected by Ollama
        response = self.llm.predict(
            messages=[{"role": "user", "content": task.prompt}],
            return_raw=True  # get full response, in case you need to debug
        )

        # print("LLM response:", response)

        # Dummy return
        return TaskSolution(task_id=task.id, actions=[], web_agent_id="local_direct_agent")

    async def solve_task(self, task: Task) -> TaskSolution:
        return self.solve_task_sync(task)
