from abc import ABC, abstractmethod
from typing import List
from evotool.task.base_task import Es1p1Adapter, Solution


class Es1p1PythonAdapter(Es1p1Adapter, ABC):
    """ES(1+1) Adapter for Python code optimization tasks.
    
    This class provides common ES(1+1) logic for Python tasks.
    Subclasses only need to implement _get_system_prompt() to define task-specific instructions.
    """
    
    def __init__(self, task_info: dict):
        super().__init__(task_info)
    
    def get_prompt(self, best_sol: Solution|None) -> List[dict]:
        """Generate prompt for Python code optimization."""
        content = self._get_system_prompt() + "\n\n"
        
        current_score = best_sol.evaluation_res.score if best_sol.evaluation_res else 0
        method = best_sol.other_info.get('method', 'unknown') if best_sol.other_info else 'unknown'
        additional_info = best_sol.evaluation_res.additional_info if best_sol.evaluation_res else {}
        
        content += f"""Task: Improve the algorithm implementation.

Current best solution ({method}, Score = {current_score:.4f}):
```python
{best_sol.sol_string}
```

Performance metrics: {additional_info}

Improve this implementation by:
- Using better algorithmic approaches
- Optimizing the computational process
- Adding mathematical enhancements
- Ensuring numerical stability

Return only the improved function implementation."""

        return [{"role": "user", "content": content}]
    
    def parse_response(self, response_str: str) -> str:
        """Parse LLM response to extract Python code."""
        lines = response_str.strip().split('\n')
        
        # Look for python code blocks
        in_code_block = False
        code_lines = []
        
        for line in lines:
            if line.strip().startswith('```python') or line.strip().startswith('```'):
                in_code_block = True
                continue
            elif line.strip() == '```' and in_code_block:
                break
            elif in_code_block:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        else:
            # If no code block found, return the whole response
            return response_str.strip()
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get system prompt for the specific Python task.
        
        This should include:
        - Task-specific requirements and constraints
        - Function interfaces and expected behavior
        - Domain-specific guidelines
        - Performance optimization goals
        """
        pass