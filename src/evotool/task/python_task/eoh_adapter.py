from abc import ABC, abstractmethod
from typing import List
from evotool.task.base_task import EohAdapter, Solution


class EohPythonAdapter(EohAdapter, ABC):
    """EOH Adapter for Python code optimization tasks.
    
    This class provides common operator logic for Python tasks.
    Subclasses only need to implement _get_system_prompt() to define task-specific instructions.
    """
    
    def __init__(self, task_info: dict):
        super().__init__(task_info)
    
    def get_prompt_i1(self) -> List[dict]:
        """Generate initialization prompt (I1 operator)."""
        content = self._get_system_prompt() + "\n\n"
        content += """Task: Initialize diverse algorithms.

Create an innovative implementation using different mathematical approaches and algorithmic foundations.

FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
{algorithmic description and reasoning}
```python
implementation code
```"""
        
        return [{"role": "user", "content": content}]
    
    def get_prompt_e1(self, selected_individuals: List[Solution]) -> List[dict]:
        """Generate E1 (crossover) prompt."""
        content = self._get_system_prompt() + "\n\n"
        content += "Task: Cross-breed algorithms to create hybrid approaches.\n\nParent Solutions:\n"
        
        for i, sol in enumerate(selected_individuals, 1):
            score = sol.evaluation_res.score if sol.evaluation_res else 0
            method = sol.other_info.get('method', 'unknown') if sol.other_info else 'unknown'
            
            content += f"\nParent {i} ({method}, Score = {score:.4f}):\n"
            content += f"```python\n{sol.sol_string}\n```\n"
        
        content += """
Crossover Strategy: Combine the best features of these algorithms to create a novel hybrid.
Consider mixing different mathematical foundations, combining preprocessing steps, and merging optimization approaches.

Create a hybrid algorithm that inherits strengths from both parents.

FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
{algorithmic description and reasoning}
```python
implementation code
```"""
        
        return [{"role": "user", "content": content}]
    
    def get_prompt_e2(self, selected_individuals: List[Solution]) -> List[dict]:
        """Generate E2 (guided crossover) prompt."""
        content = self._get_system_prompt() + "\n\n"
        content += "Task: Guided crossover with performance analysis.\n\nCandidate Solutions:\n"
        
        for i, sol in enumerate(selected_individuals, 1):
            score = sol.evaluation_res.score if sol.evaluation_res else 0
            method = sol.other_info.get('method', 'unknown') if sol.other_info else 'unknown'
            additional_info = sol.evaluation_res.additional_info if sol.evaluation_res else {}
            
            content += f"\nCandidate {i} ({method}, Score = {score:.4f}):\n"
            if additional_info:
                content += f"  Metrics: {additional_info}\n"
            content += f"```python\n{sol.sol_string}\n```\n"
        
        content += """
Guided Crossover: Analyze the performance patterns and create a targeted hybrid.

Analysis Focus:
- Which components contribute most to high scores?
- What causes performance issues?
- How do different approaches handle edge cases?
- Which algorithmic principles show most promise?

Create a principled combination that addresses identified weaknesses while preserving strengths.

FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
{algorithmic description and reasoning}
```python
implementation code
```"""
        
        return [{"role": "user", "content": content}]
    
    def get_prompt_m1(self, individual: Solution) -> List[dict]:
        """Generate M1 (mutation) prompt."""
        content = self._get_system_prompt() + "\n\n"
        score = individual.evaluation_res.score if individual.evaluation_res else 0
        method = individual.other_info.get('method', 'unknown') if individual.other_info else 'unknown'
        
        content += f"""Task: Mutate algorithm for improvement.

Current Solution ({method}, Score = {score:.4f}):
```python
{individual.sol_string}
```

Mutation Strategy: Make focused improvements to the algorithm.
Consider:
- Algorithmic refinements and optimizations
- Mathematical enhancements
- Implementation improvements
- Novel variations of the core approach

Create a meaningful variation that could improve performance.

FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
{{algorithmic description and reasoning}}
```python
implementation code
```"""
        
        return [{"role": "user", "content": content}]
    
    def get_prompt_m2(self, individual: Solution) -> List[dict]:
        """Generate M2 (parameter mutation) prompt."""
        content = self._get_system_prompt() + "\n\n"
        score = individual.evaluation_res.score if individual.evaluation_res else 0
        method = individual.other_info.get('method', 'unknown') if individual.other_info else 'unknown'
        
        content += f"""Task: Parameter-focused mutation.

Current Solution ({method}, Score = {score:.4f}):
```python
{individual.sol_string}
```

Parameter Mutation: Modify algorithmic parameters and hyperparameters.
Focus on:
- Numerical constants and thresholds
- Regularization parameters  
- Tolerance values and convergence criteria
- Complexity parameters
- Weighting schemes

Adjust parameters to potentially improve performance or numerical stability.

FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
{{algorithmic description and reasoning}}
```python
implementation code
```"""
        
        return [{"role": "user", "content": content}]
    
    def parse_response(self, response_str: str) -> tuple[str, str]:
        """Parse LLM response to extract Python code and algorithm description."""
        import re
        
        # Extract algorithm/thought from response using pattern matching (same as CUDA EOH)
        try:
            pattern = r'\{.*?\}'
            bracketed_texts = re.findall(pattern, response_str, re.DOTALL)
            algorithm = bracketed_texts[0] if bracketed_texts else None
            if algorithm:
                # Remove the outer braces
                algorithm = algorithm[1:-1].strip()
        except:
            algorithm = None
        
        # Remove the algorithm part from response before code extraction
        response_without_algorithm = response_str
        if algorithm:
            # Remove all {algorithm} parts from the response
            response_without_algorithm = re.sub(r'\{.*?\}', '', response_str, flags=re.DOTALL)
        
        # Extract Python code block
        code_patterns = [
            r'```python\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```'
        ]
        
        code = None
        for pattern in code_patterns:
            matches = re.findall(pattern, response_without_algorithm, re.DOTALL)
            if matches:
                code = matches[0].strip()
                break
        
        # Fallback if no code block found
        if not code:
            code = response_without_algorithm.strip()
        
        # Fallback if no algorithm found
        if not algorithm:
            algorithm = "Algorithm description not provided"
        
        return code, algorithm
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get system prompt for the specific Python task.
        
        This should include:
        - Task-specific requirements and constraints
        - Function interfaces and expected behavior
        - Domain-specific guidelines
        - Output format requirements
        """
        pass