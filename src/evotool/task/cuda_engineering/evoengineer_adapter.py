import re
from evotool.task.base_task import EvoEngineerAdapter, Solution, Operator
from typing import List

def _make_task_description(operation_name: str, GPU_TYPE: str, CUDA_VER: str) -> str:
    return f"""You are a Machine Learning Engineer trying to reduce the runtime of a {operation_name} kernel in CUDA. 
Make sure the kernel returns the correct result. Do not use any alternative precision that could result in an incorrect result. 
The kernel will be run on a {GPU_TYPE} GPU with CUDA version {CUDA_VER}.

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
            Operator("mutation", 1)
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

Reference insights (consider if relevant):
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

Think deeply about how to optimize this CUDA kernel. {'Reference insights are provided above - use them as inspiration if they seem relevant to your optimization approach.' if random_thoughts and len(random_thoughts) > 0 else ''} Propose a new CUDA kernel that:
1. Analyzes the current implementation to identify optimization opportunities
2. Applies proven CUDA optimization techniques based on your analysis
3. Explains your optimization rationale and approach clearly

The new kernel should aim to reduce the runtime of the operation while ensuring it returns the correct result.
The PYBIND11_MODULE has to be the same as in the example.
MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.

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

Reference insights (consider if relevant):
{thoughts_list}

"""
            
            # Build XML-structured individuals
            indivs_xml = ""
            for i, indi in enumerate(selected_individuals):
                name = indi.other_info.get('name', f"kernel_{i+1}")
                runtime = -indi.evaluation_res.score
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

I have {len(selected_individuals)} other kernel implementations to learn from:
{indivs_xml}{thoughts_section}

Analyze the current best kernel and these alternative implementations, then think deeply about how to combine their best ideas. {'Reference insights are provided above - use them as inspiration if they seem relevant to your crossover approach.' if random_thoughts and len(random_thoughts) > 0 else ''} Propose a new CUDA kernel that:
1. Analyzes the different optimization approaches from the existing implementations
2. Combines the most effective ideas and techniques from multiple kernels
3. Explains your crossover rationale and which implementation ideas you merged

The new kernel should aim to reduce the runtime by combining ideas from the existing implementations while ensuring it returns the correct result.
The PYBIND11_MODULE has to be the same as in the example.
MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.

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

Reference insights (consider if relevant):
{thoughts_list}

"""
            
            name = individual.other_info.get('name', 'mutation_base')
            runtime = -individual.evaluation_res.score
            thought = individual.other_info.get('thought', 'No thought provided')
            
            prompt = f"""
{task_description}

Here is the current best CUDA kernel for reference:

<current_best>
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
</current_best>

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
</source_kernel>{thoughts_section}

Think creatively about how to fundamentally reimagine this kernel, using the current best as your correctness reference. {'Reference insights are provided above - use them as inspiration if they seem relevant to your mutation approach.' if random_thoughts and len(random_thoughts) > 0 else ''} Propose a deeply mutated CUDA kernel that:
1. Explores radical changes in algorithmic approach, memory access patterns, or parallelization strategy
2. May completely restructure the computation while maintaining functional correctness against the current best
3. Explains your bold mutation concept and the reasoning behind this fundamental transformation

The mutated kernel should have a different form but maintain the same functionality while aiming for better runtime performance than the current best.
The PYBIND11_MODULE has to be the same as in the example.
MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.

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

