import re
from evotool.task.base_task import Es1p1Adapter, Solution, EvaluationResult
from  typing import List

from evotool.task.cuda_engineering import CudaAdapter


class Es1p1CudaAdapter(CudaAdapter, Es1p1Adapter):
    def __init__(self, task_info: dict):
        Es1p1Adapter.__init__(self, task_info)

    def get_prompt(self, best_sol:Solution) -> List[dict]:
        prompt = f"""
You are a Machine Learning Engineer trying to reduce the runtime of a kernel in CUDA. Make sure the kernel returns the correct result. Do not use any alternative precision that could result in an incorrect result. The kernel will be run on a RTX 4090 GPU with CUDA 12.4.

Answer using the following schema:

```cpp
[Your kernel implementation]
```

The pybind11 cuda module name has to be the same as in the example.
MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.

Here is the CUDA kernel code example you need to optimize:
```cpp
{best_sol.sol_string}
```
Propose a new CUDA kernel code which aims to reduce the runtime of the operation, while ensuring the kernel returns the correct result.
"""
        prompt_content = [{'role': 'user', 'content': prompt}]
        return prompt_content

    def parse_response(self, response_str: str) -> Solution:
        # Try different code block patterns in order of preference
        patterns = [
            r'```cpp\s*\n(.*?)\n```',      # cpp
            r'```c\+\+\s*\n(.*?)\n```',    # c++
            r'```cuda\s*\n(.*?)\n```',     # cuda
            r'```c\s*\n(.*?)\n```',        # c
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

