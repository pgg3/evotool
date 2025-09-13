import re
from abc import ABC, abstractmethod
from typing import List
from evotool.task.base_task import Es1p1Adapter, Solution


class Es1p1PythonAdapter(Es1p1Adapter):
    def __init__(self, task_info: dict):
        super().__init__(task_info)
    
    def get_prompt(self, best_sol: Solution|None) -> List[dict]:
        task_description = self._get_base_task_description()

        prompt = f"""{task_description}

Here is the Python code example you need to optimize:
```python
{best_sol.sol_string}
```
Propose a new Python code which performs better than the above code.
"""
        prompt_content = [{'role': 'user', 'content': prompt}]
        return prompt_content

    def parse_response(self, response_str: str) -> Solution:
        # Try different code block patterns in order of preference
        patterns = [
            r'```python\s*\n(.*?)\n```',
            r'```Python\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```'  # generic code block
        ]

        # Find all matches using case insensitive search
        for pattern in patterns:
            matches = re.findall(pattern, response_str, re.DOTALL | re.IGNORECASE)
            if matches:
                # Return the longest match (likely the most complete implementation)
                return Solution(max(matches, key=len).strip())
        # Last resort: return stripped response
        return Solution(response_str.strip())