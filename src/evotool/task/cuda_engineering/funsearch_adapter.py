import re
from evotool.task.base_task import FunSearchAdapter, Solution, EvaluationResult
from typing import List


class FunSearchCudaAdapter(FunSearchAdapter):
    """FunSearch adapter for CUDA kernel optimization"""
    
    def __init__(self, task_info: dict):
        super().__init__(task_info)
    
    def make_init_sol(self) -> Solution:
        """Create initial solution from cuda_code"""
        init_sol = Solution(self.task_info["cuda_code"])
        evaluation_res = EvaluationResult(
            valid=True,
            score=-self.task_info["cuda_info"]["runtime"],
            additional_info_dict=dict()
        )
        init_sol.evaluation_res = evaluation_res
        return init_sol
    
    def get_prompt(self, solutions: List[Solution]) -> List[dict]:
        """Generate prompt based on multiple solutions (similar to reference implementation)"""
        if len(solutions) == 1:
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
{solutions[0].sol_string}
```

Propose a new CUDA kernel code which aims to reduce the runtime of the operation, while ensuring the kernel returns the correct result.
"""
        elif len(solutions) >= 2:
            prompt = f"""
You are a Machine Learning Engineer trying to reduce the runtime of a kernel in CUDA. Make sure the kernel returns the correct result. Do not use any alternative precision that could result in an incorrect result. The kernel will be run on a RTX 4090 GPU with CUDA 12.4.

Answer using the following schema:

```cpp
[Your kernel implementation]
```

The pybind11 cuda module name has to be the same as in the example.
MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.

Here is a CUDA kernel code example:
```cpp
{solutions[0].sol_string}
```

A better version of the CUDA kernel code example is as follows:
```cpp
{solutions[1].sol_string}
```

Propose a new CUDA kernel code which aims to reduce the runtime of the operation, while ensuring the kernel returns the correct result.
"""
        else:
            # Fallback if no solutions provided
            prompt = f"""
You are a Machine Learning Engineer trying to reduce the runtime of a kernel in CUDA. Make sure the kernel returns the correct result. Do not use any alternative precision that could result in an incorrect result. The kernel will be run on a RTX 4090 GPU with CUDA 12.4.

Answer using the following schema:

```cpp
[Your kernel implementation]
```

The pybind11 cuda module name has to be the same as in the example.
MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.

Here is the original CUDA kernel code:
```cpp
{self.task_info["cuda_code"]}
```

Propose an optimized CUDA kernel code which aims to reduce the runtime of the operation, while ensuring the kernel returns the correct result.
"""
        
        prompt_content = [{'role': 'user', 'content': prompt}]
        return prompt_content
    
    def parse_response(self, response_str: str) -> str:
        """Parse LLM response to extract CUDA code"""
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
                return max(matches, key=len).strip()
        
        # Last resort: return stripped response
        return response_str.strip()