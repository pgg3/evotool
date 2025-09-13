import re
from abc import ABC, abstractmethod
from typing import List
from evotool.task.base_task import FunSearchAdapter, Solution, EvaluationResult


class FunSearchPythonAdapter(FunSearchAdapter):
    """FunSearch Adapter for Python code optimization tasks.
    
    This class provides common FunSearch logic for Python tasks.
    Subclasses only need to implement _get_system_prompt() to define task-specific instructions.
    """
    
    def __init__(self, task_info: dict):
        super().__init__(task_info)
    
    def get_prompt(self, solutions: List[Solution]) -> List[dict]:
        """Generate prompt based on multiple solutions (similar to CUDA implementation)"""
        task_description = self._get_base_task_description()
        if len(solutions) == 1:
            prompt = f"""{task_description}

You are a Machine Learning Engineer trying to optimize Python code for better performance. Make sure the code returns the correct result and maintains the same functionality. Focus on algorithmic improvements, data structure optimizations, and efficient Python patterns.

Answer using the following schema:

```python
[Your Python implementation]
```

MAKE SURE THE PROPOSAL CODE IS VALID PYTHON CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.

Here is the Python code example you need to optimize:
```python
{solutions[0].sol_string}
```

Propose a new Python code which aims to improve the performance of the operation, while ensuring the code returns the correct result.
"""
        elif len(solutions) >= 2:
            prompt = f"""{task_description}

You are a Machine Learning Engineer trying to optimize Python code for better performance. Make sure the code returns the correct result and maintains the same functionality. Focus on algorithmic improvements, data structure optimizations, and efficient Python patterns.

Answer using the following schema:

```python
[Your Python implementation]
```

MAKE SURE THE PROPOSAL CODE IS VALID PYTHON CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.

Here is a Python code example:
```python
{solutions[0].sol_string}
```

A better version of the Python code example is as follows:
```python
{solutions[1].sol_string}
```

Propose a new Python code which aims to improve the performance of the operation, while ensuring the code returns the correct result.
"""
        else:
            # Fallback if no solutions provided
            prompt = f"""{task_description}

You are a Machine Learning Engineer trying to optimize Python code for better performance. Make sure the code returns the correct result and maintains the same functionality. Focus on algorithmic improvements, data structure optimizations, and efficient Python patterns.

Answer using the following schema:

```python
[Your Python implementation]
```

MAKE SURE THE PROPOSAL CODE IS VALID PYTHON CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.

Here is the original Python code:
```python
{self.task_info["python_code"]}
```

Propose an optimized Python code which aims to improve the performance of the operation, while ensuring the code returns the correct result.
"""
        
        prompt_content = [{'role': 'user', 'content': prompt}]
        return prompt_content
    
    def parse_response(self, response_str: str) -> Solution:
        """Parse LLM response to extract CUDA code"""
        # Try different code block patterns in order of preference
        patterns = [
            r'```python\s*\n(.*?)\n```',
            r'```Python\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```'          # generic code block
        ]

        # Find all matches using case insensitive search
        for pattern in patterns:
            matches = re.findall(pattern, response_str, re.DOTALL | re.IGNORECASE)
            if matches:
                # Return the longest match (likely the most complete implementation)
                return Solution(max(matches, key=len).strip())

        # Last resort: return stripped response
        return Solution(response_str.strip())
