from abc import ABC, abstractmethod
from typing import List
from evotool.task.base_task import FunSearchAdapter, Solution


class FunSearchPythonAdapter(FunSearchAdapter, ABC):
    """FunSearch Adapter for Python code optimization tasks.
    
    This class provides common FunSearch logic for Python tasks.
    Subclasses only need to implement _get_system_prompt() to define task-specific instructions.
    """
    
    def __init__(self, task_info: dict):
        super().__init__(task_info)
    
    def get_prompt(self, best_sols: List[Solution]) -> List[dict]:
        """Generate prompt for Python code optimization using multiple solutions."""
        content = self._get_system_prompt() + "\n\n"
        content += "Task: Analyze multiple implementations and create an improved hybrid approach.\n\n"
        
        for i, sol in enumerate(best_sols, 1):
            score = sol.evaluation_res.score if sol.evaluation_res else 0
            method = sol.other_info.get('method', 'unknown') if sol.other_info else 'unknown'
            additional_info = sol.evaluation_res.additional_info if sol.evaluation_res else {}
            
            content += f"Implementation {i} ({method}, Score = {score:.4f}):\n"
            if additional_info:
                content += f"  Metrics: {additional_info}\n"
            content += f"```python\n{sol.sol_string}\n```\n\n"
        
        content += """Analysis Strategy: Create a novel approach that combines the best aspects of these solutions.

Consider:
- Learning from the strengths of each implementation
- Combining different algorithmic foundations
- Creating hybrid approaches that merge successful techniques
- Introducing innovations that go beyond the existing solutions

Return an improved implementation that synthesizes insights from all approaches."""

        return [{"role": "user", "content": content}]
    
    def parse_response(self, response_str: str) -> str:
        """Parse LLM response to extract Python code."""
        lines = response_str.strip().split('\n')
        
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
            return response_str.strip()
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get system prompt for the specific Python task.
        
        This should include:
        - Task-specific requirements and constraints
        - Function interfaces and expected behavior
        - Domain-specific guidelines
        - Synthesis and combination strategies
        """
        pass