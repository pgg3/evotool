import re
from evotool.task.base_task import EvoEngineerAdapter, Solution, Operator
from typing import List

def _make_task_description(operation_name: str, GPU_TYPE: str, CUDA_VER: str) -> str:
    return f"""You are a Machine Learning Engineer trying to reduce the runtime of a {operation_name} kernel in CUDA. 
Make sure the kernel returns the correct result. Do not use any alternative precision that could result in an incorrect result. 
The kernel will be run on a {GPU_TYPE} GPU with CUDA version {CUDA_VER}.

The PYBIND11_MODULE has to be the same as in the example.
MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
"""


class EvoEngineerCudaAdapter(EvoEngineerAdapter):
    def __init__(self, task_info: dict):
        super().__init__(task_info)
    
    def _get_base_task_description(self) -> str:
        """Get the base task description using task info"""
        operation_name = self.task_info.get("operation_name", "operation")
        gpu_type = self.task_info.get("gpu_type", "RTX 4090")
        cuda_version = self.task_info.get("cuda_version", "12.4.1")
        
        return _make_task_description(
            operation_name=operation_name,
            GPU_TYPE=gpu_type,
            CUDA_VER=cuda_version
        )

    def make_init_sol(self) -> Solution:
        """Create initial solution from the baseline CUDA code"""
        from evotool.task.base_task import EvaluationResult

        other_info = {'name': "Baseline", "thought": "Baseline"}
        init_sol = Solution(self.task_info["cuda_code"], other_info)
        evaluation_res = EvaluationResult(
            valid=True,
            score=-self.task_info["cuda_info"]["runtime"],  # Negative because lower runtime is better
            additional_info={
                "prof_string": self.task_info["cuda_info"]["prof_string"],
            }
        )
        init_sol.evaluation_res = evaluation_res
        return init_sol

    def get_init_operators(self) -> List[Operator]:
        """Get initialization operators for CUDA optimization"""
        return [
            Operator("init", 0)
        ]
    
    def get_offspring_operators(self) -> List[Operator]:
        """Get offspring operators for CUDA optimization"""
        return [
            Operator("crossover", 2),
            Operator("mutation", 1),
            Operator("rewrite", 1),
            Operator("init", 0)
        ]

    def get_operator_prompt(self, operator_name: str, selected_individuals: List[Solution], current_best_sol: Solution, random_thoughts: List[str], **kwargs) -> List[dict]:
        """Generate prompt for any operator"""
        task_description = self._get_base_task_description()

        if current_best_sol is None:
            current_best_sol = self.make_init_sol()
        
        if operator_name == "init":
            # Build the thoughts section if available
            thoughts_section = ""
            if random_thoughts and len(random_thoughts) > 0:
                thoughts_list = "\n".join([f"- {thought}" for thought in random_thoughts])
                thoughts_section = f"""

Helpful thoughts to consider:
{thoughts_list}

"""
            
            prompt = f"""
{task_description}

Here is the CUDA kernel code you need to optimize:

<current_kernel>
<name>{current_best_sol.other_info['name']}</name>
<thought>{current_best_sol.other_info['thought']}</thought>
<code>
```c++
{current_best_sol.sol_string}
```
</code>
<runtime>{-current_best_sol.evaluation_res.score:.5f} milliseconds</runtime>
<profile_info>
{current_best_sol.evaluation_res.additional_info['prof_string']}
</profile_info>
</current_kernel>{thoughts_section}

Based on the current kernel{' and the helpful thoughts above' if random_thoughts and len(random_thoughts) > 0 else ''}, propose a new CUDA kernel that:
1. {'Analyzes how the thoughts can be applied to optimize performance' if random_thoughts and len(random_thoughts) > 0 else 'Analyzes the current implementation to identify optimization opportunities'}
2. {'Incorporates relevant insights from the thoughts into your design' if random_thoughts and len(random_thoughts) > 0 else 'Applies proven CUDA optimization techniques'}
3. {'Explains in your "thought" field how you utilized the provided thoughts' if random_thoughts and len(random_thoughts) > 0 else 'Explains your optimization rationale clearly'}

The new kernel should aim to reduce the runtime of the operation while ensuring it returns the correct result.

Answer using the following schema:

name: A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
code: The proposed cuda script in code.
thought: The rationale for the improvement idea.

"""
            return [{'role': 'user', 'content': prompt}]

        elif operator_name == "crossover":
            # Build the thoughts section if available
            thoughts_section = ""
            if random_thoughts and len(random_thoughts) > 0:
                thoughts_list = "\n".join([f"- {thought}" for thought in random_thoughts])
                thoughts_section = f"""

Helpful thoughts to consider:
{thoughts_list}

"""
            
            # Build XML-structured individuals
            indivs_xml = ""
            for i, indi in enumerate(selected_individuals):
                name = indi.other_info.get('name', f"kernel_{i+1}")
                runtime = -indi.evaluation_res.score
                prof_string = indi.evaluation_res.additional_info.get('prof_string', 'No profile info available')
                thought = indi.other_info.get('thought', 'No thought provided')
                
                indivs_xml += f"""
<kernel_{i+1}>
<name>{name}</name>
<thought>{thought}</thought>
<code>
```c++
{indi.sol_string}
```
</code>
<runtime>{runtime:.5f} milliseconds</runtime>
<profile_info>
{prof_string}
</profile_info>
</kernel_{i+1}>"""
            
            prompt = f"""
{task_description}

Here is the current best CUDA kernel to optimize:

<current_kernel>
<name>{current_best_sol.other_info.get('name', 'current_best')}</name>
<thought>{current_best_sol.other_info.get('thought', 'Current best implementation')}</thought>
<code>
```c++
{current_best_sol.sol_string}
```
</code>
<runtime>{-current_best_sol.evaluation_res.score:.5f} milliseconds</runtime>
<profile_info>
{current_best_sol.evaluation_res.additional_info['prof_string']}
</profile_info>
</current_kernel>

I have {len(selected_individuals)} existing kernel implementations to learn from:
{indivs_xml}{thoughts_section}

Based on the current kernel, the existing implementations{' and the helpful thoughts above' if random_thoughts and len(random_thoughts) > 0 else ''}, propose a new CUDA kernel that:
1. {'Analyzes how the thoughts can guide the combination of ideas from different implementations' if random_thoughts and len(random_thoughts) > 0 else 'Analyzes different approaches from the existing implementations'}
2. {'Incorporates relevant insights from both the thoughts and the kernel implementations' if random_thoughts and len(random_thoughts) > 0 else 'Combines the best ideas from the existing implementations'}  
3. {'Explains how you utilized the provided thoughts and which implementation ideas you merged' if random_thoughts and len(random_thoughts) > 0 else 'Explains which implementation ideas you combined and why'}

The new kernel should aim to reduce the runtime by combining ideas from the existing implementations while ensuring it returns the correct result.

Answer using the following schema:

name: A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
code: The proposed cuda script in code.
thought: The rationale for the improvement idea.

"""
            return [{'role': 'user', 'content': prompt}]
        elif operator_name == "mutation":
            individual = selected_individuals[0]
            
            # Build the thoughts section if available
            thoughts_section = ""
            if random_thoughts and len(random_thoughts) > 0:
                thoughts_list = "\n".join([f"- {thought}" for thought in random_thoughts])
                thoughts_section = f"""

Helpful thoughts to consider:
{thoughts_list}

"""
            
            name = individual.other_info.get('name', 'mutation_base')
            runtime = -individual.evaluation_res.score
            prof_string = individual.evaluation_res.additional_info.get('prof_string', 'No profile info available')
            thought = individual.other_info.get('thought', 'No thought provided')
            
            prompt = f"""
{task_description}

Here is the kernel implementation to mutate:

<source_kernel>
<name>{name}</name>
<thought>{thought}</thought>
<code>
```c++
{individual.sol_string}
```
</code>
<runtime>{runtime:.5f} milliseconds</runtime>
<profile_info>
{prof_string}
</profile_info>
</source_kernel>{thoughts_section}

Based on the source kernel{' and the helpful thoughts above' if random_thoughts and len(random_thoughts) > 0 else ''}, propose a mutated CUDA kernel that:
1. {'Uses the thoughts to guide specific modifications to the implementation' if random_thoughts and len(random_thoughts) > 0 else 'Makes targeted modifications to improve performance'}
2. {'Applies insights from the thoughts while maintaining the core algorithmic approach' if random_thoughts and len(random_thoughts) > 0 else 'Keeps the core approach but optimizes specific aspects'}
3. {'Explains how the thoughts influenced your mutation strategy' if random_thoughts and len(random_thoughts) > 0 else 'Explains what specific changes you made and why'}

The mutated kernel should have a different form but maintain the same functionality while aiming for better runtime performance.

Answer using the following schema:

name: A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
code: The proposed cuda script in code.
thought: The rationale for the improvement idea.

"""
            return [{'role': 'user', 'content': prompt}]
            
        elif operator_name == "parameter_mutation":
            individual = selected_individuals[0]
            if 'algorithm' in individual.other_info and individual.other_info['algorithm']:
                algorithm_desc = individual.other_info['algorithm']
            else:
                algorithm_desc = "Current kernel implementation"
            
            prompt = f"""{task_description}

I have one kernel implementation with its code as follows. Kernel implementation description:
{algorithm_desc}
Code:
{individual.sol_string}

Please identify the main kernel implementation parameters and assist me in creating a new kernel implementation that has a different parameter settings of the kernel implementation provided.
1. First, describe your new kernel implementation and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the kernel:
```cpp
[Your kernel implementation]
```
Do not give additional explanations.
"""
            return [{'role': 'user', 'content': prompt}]
            
        elif operator_name == "rewrite":
            individual = selected_individuals[0]
            
            # Build the thoughts section if available
            thoughts_section = ""
            if random_thoughts and len(random_thoughts) > 0:
                thoughts_list = "\n".join([f"- {thought}" for thought in random_thoughts])
                thoughts_section = f"""

Helpful thoughts to consider:
{thoughts_list}

"""
            
            name = individual.other_info.get('name', 'rewrite_base')
            runtime = -individual.evaluation_res.score
            prof_string = individual.evaluation_res.additional_info.get('prof_string', 'No profile info available')
            thought = individual.other_info.get('thought', 'No thought provided')
            
            prompt = f"""
{task_description}

Here is the kernel implementation to rewrite:

<source_kernel>
<name>{name}</name>
<thought>{thought}</thought>
<code>
```c++
{individual.sol_string}
```
</code>
<runtime>{runtime:.5f} milliseconds</runtime>
<profile_info>
{prof_string}
</profile_info>
</source_kernel>{thoughts_section}

Based on the source kernel{' and the helpful thoughts above' if random_thoughts and len(random_thoughts) > 0 else ''}, propose a completely rewritten CUDA kernel that:
1. {'Uses the thoughts to inspire a fundamentally different algorithmic approach' if random_thoughts and len(random_thoughts) > 0 else 'Takes a fundamentally different algorithmic approach'}
2. {'Applies insights from the thoughts to redesign the implementation from scratch' if random_thoughts and len(random_thoughts) > 0 else 'Redesigns the implementation using different optimization techniques'}
3. {'Explains how the thoughts led you to this new approach' if random_thoughts and len(random_thoughts) > 0 else 'Explains your new approach and why it should be faster'}

The rewritten kernel should achieve the same functionality but with a significantly different implementation strategy that aims for better performance.

Answer using the following schema:

name: A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
code: The proposed cuda script in code.
thought: The rationale for the improvement idea.

"""
            return [{'role': 'user', 'content': prompt}]
            
        else:
            raise ValueError(f"Unknown operator: {operator_name}")

    def parse_response(self, response_str: str) -> Solution:
        """Parse LLM response to extract solution string"""
        import json
        
        if response_str is None:
            return Solution("")

        proposed_content = response_str

        # Initialize variables to prevent undefined errors
        name = ""
        thought = ""
        cleaned_code = ""

        # if surrounded by ```json ```
        if proposed_content.strip().startswith("```json") and proposed_content.endswith("```"):
            content_inside = proposed_content.strip()[7:-3]
            try:
                json_dict = json.loads(content_inside)
                code_block_pattern = re.compile(
                    r'\s*(?:```[^\n]*)?\n?(.*?)(```|$)',
                    re.DOTALL
                )

                cleaned_code = code_block_pattern.search(json_dict["code"]).group(1)
                json_dict["code"] = cleaned_code
                del json_dict["code"]  # Fix: use del instead of remove

                # Return Solution with the cleaned code
                return Solution(cleaned_code)
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                # Fall through to regex parsing if JSON parsing fails
                pass

        name_heading = r"(?:name|Name|NAME)\s*:?"
        thought_heading = r"(?:thought|Thought|THOUGHT)\s*:?"
        code_heading = r"(?:code|Code|CODE)\s*:?"

        # Extract name
        name_pattern = re.compile(r"" + name_heading + r"\s*(.*?)" + code_heading, re.DOTALL)
        match = name_pattern.search(proposed_content)
        if match:
            name = match.group(1).strip()

        # Extract code
        code_pattern = re.compile(r"" + code_heading + r"\s*(.*)" + thought_heading, re.DOTALL)
        match = code_pattern.search(proposed_content)
        if match:
            code = match.group(1).strip()
            code_block_pattern = re.compile(
                r'\s*(?:```[^\n]*)?\n?(.*?)(```|$)',
                re.DOTALL
            )
            code_match = code_block_pattern.search(code)
            if code_match:
                cleaned_code = code_match.group(1)
            else:
                cleaned_code = code

        # Extract thought
        thought_pattern = re.compile(r"" + thought_heading + r"\s*(.*?)$", re.DOTALL)
        match = thought_pattern.search(proposed_content)
        if match:
            thought = match.group(1).strip()

        # Return Solution with cleaned code and other info
        other_info = {
            "name": name,
            "thought": thought
        }
        return Solution(cleaned_code if cleaned_code else proposed_content.strip(), other_info=other_info)

