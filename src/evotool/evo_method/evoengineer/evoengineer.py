import concurrent.futures
from typing import List, Type

from .run_config import EvoEngineerConfig
from .run_state_dict import EvoEngineerRunStateDict
from ..base_method import Method
from evotool.task.base_task import Solution
from ..base_run_state_dict import BaseRunStateDict


class EvoEngineer(Method):
    def __init__(self, config: EvoEngineerConfig):
        super().__init__(config)
        self.config = config

    def run(self):
        """Main EvoEngineer algorithm execution"""
        self.verbose_title("EvoEngineer ALGORITHM STARTED")

        if "sample" not in self.run_state_dict.usage_history:
            self.run_state_dict.usage_history["sample"] = []

        # Initialize with seed solution if sol_history is empty
        if len(self.run_state_dict.sol_history) == 0:
            initial_sol = self.config.adapter.make_init_sol()
            # Don't use _register_solution as this doesn't consume samples
            self.run_state_dict.sol_history.append(initial_sol)
            self.run_state_dict.population.append(initial_sol)
            self._save_run_state_dict()
            self.verbose_info(f"Initialized with baseline solution (score: {initial_sol.evaluation_res.score if initial_sol.evaluation_res else 'None'})")

        # Initialize population if starting from scratch
        if self.run_state_dict.generation == 0:
            self._initialize_population()
            
        # Check if we have enough individuals for selection
        valid_population = self._get_valid_population(self.run_state_dict.population)
        if len(valid_population) < 2:  # Need at least 2 for selection
            self.verbose_info(f'The search is terminated since EvoEngineer unable to obtain 2 feasible algorithms during initialization.')
            return
        
        # Main evolution loop - moved loop control logic here
        while (self.run_state_dict.generation < self.config.max_generations) and (self.run_state_dict.tot_sample_nums < self.config.max_sample_nums):
            try:
                self.verbose_info(f"Generation {self.run_state_dict.generation} - Sample {self.run_state_dict.tot_sample_nums + 1} - {self.run_state_dict.tot_sample_nums + self.config.num_samplers} / {self.config.max_sample_nums or 'unlimited'}")
                
                # Apply offspring operators in parallel for this generation
                self._apply_operators_parallel(self.config.get_offspring_operators(), f"Gen {self.run_state_dict.generation}")
                
                # Manage population size - keep only the best pop_size individuals
                self._manage_population_size()
                
                self.run_state_dict.generation += 1
                self._save_run_state_dict()
                
            except KeyboardInterrupt:
                self.verbose_info("Evolution interrupted by user")
                break
            except Exception as e:
                self.verbose_info(f"Evolution error: {str(e)}")
                continue
        
        # Mark as done and save final state
        self.run_state_dict.is_done = True
        self._save_run_state_dict()

    def _initialize_population(self):
        """Initialize population using init operators - keep generating until we have enough valid solutions"""
        self.verbose_info("Initializing population...")
        
        initial_sample_limit = self.config.max_sample_nums # Reasonable limit
        
        # Keep generating until we have pop_size valid solutions or hit sample limit
        while (len(self._get_valid_population(self.run_state_dict.population)) < self.config.pop_size and 
               self.run_state_dict.tot_sample_nums < initial_sample_limit):
            
            # Apply init operators in parallel
            self._apply_operators_parallel(self.config.get_init_operators(), "Init")
            
            valid_count = len(self._get_valid_population(self.run_state_dict.population))
            self.verbose_info(f"Valid solutions: {valid_count}/{self.config.pop_size}")
            
            self._save_run_state_dict()
        
        valid_population = self._get_valid_population(self.run_state_dict.population)
        if len(valid_population) >= 2:  # Need at least 2 for selection
            self.run_state_dict.generation = 1
            self._save_run_state_dict()
            self.verbose_info(f"Initialization completed with {len(valid_population)} valid solutions")
        else:
            self.verbose_info(f"Warning: Only {len(valid_population)} valid solutions obtained, need at least 2")

    def _get_valid_population(self, population: List[Solution]) -> List[Solution]:
        """Get valid solutions from population"""
        return [sol for sol in population if sol.evaluation_res and sol.evaluation_res.valid]

    def _get_best_valid_sol(self, sol_history: List[Solution]) -> Solution:
        """Get the best valid solution from sol_history"""
        valid_sols = [sol for sol in sol_history if sol.evaluation_res and sol.evaluation_res.valid and sol.evaluation_res.score is not None]
        if valid_sols:
            return max(valid_sols, key=lambda x: x.evaluation_res.score)
        return sol_history[-1] if sol_history else None
    
    def _register_solution(self, solution: Solution):
        """Register a new solution to both sol_history and population"""
        self.run_state_dict.sol_history.append(solution)
        self.run_state_dict.population.append(solution)
        self.run_state_dict.tot_sample_nums += 1

    def _apply_operators_parallel(self, operators: List, generation_label: str = ""):
        """Apply operators in parallel and register solutions"""
        if not operators:
            return
            
        # Single executor for both generation and evaluation
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.num_samplers + self.config.num_evaluators) as executor:
            generate_futures = []
            eval_futures = []
            
            # Submit generation tasks up to num_samplers limit
            operators_to_use = operators[:self.config.num_samplers]
            for i, operator in enumerate(operators_to_use):
                selected_individuals = self._select_individuals_for_operator(operator)
                future = executor.submit(self._generate_single_solution, operator, selected_individuals, i)
                generate_futures.append((operator.name, future))
            
            # Process generations as they complete and immediately submit for evaluation
            future_to_operator = {future: operator_name for operator_name, future in generate_futures}
            for future in concurrent.futures.as_completed([f for _, f in generate_futures]):
                operator_name = future_to_operator[future]
                try:
                    solution, usage = future.result()
                    
                    # Add usage history
                    self.run_state_dict.usage_history["sample"].append(usage)
                    
                    # Immediately submit for evaluation without waiting
                    if solution.sol_string.strip():
                        eval_future = executor.submit(self.config.evaluator.evaluate_code, solution.sol_string)
                        eval_futures.append((eval_future, solution, operator_name))
                    else:
                        self._register_solution(solution)
                        # Log result for empty solution
                        self.verbose_info(f"{operator_name} {generation_label} - Score: None (Invalid)")
                    
                except Exception as e:
                    self.verbose_info(f"Error generating {operator_name}: {str(e)}")
                    continue
            
            # Collect evaluation results
            eval_future_to_info = {eval_future: (solution, operator_name) for eval_future, solution, operator_name in eval_futures}
            for eval_future in concurrent.futures.as_completed([ef for ef, _, _ in eval_futures]):
                solution, operator_name = eval_future_to_info[eval_future]
                try:
                    evaluation_res = eval_future.result()
                    solution.evaluation_res = evaluation_res
                    self._register_solution(solution)
                    
                    # Log result
                    score_str = "None" if not solution.evaluation_res or solution.evaluation_res.score is None else f"{solution.evaluation_res.score}"
                    valid_str = "Valid" if solution.evaluation_res and solution.evaluation_res.valid else "Invalid"
                    self.verbose_info(f"{operator_name} {generation_label} - Score: {score_str} ({valid_str})")
                    
                except Exception as e:
                    self.verbose_info(f"Error evaluating {operator_name}: {str(e)}")
                    self._register_solution(solution)  # Add with no evaluation result
                    continue

    def _manage_population_size(self):
        """Manage population size - keep only the best pop_size individuals"""
        if len(self.run_state_dict.population) <= self.config.pop_size:
            return
            
        # Separate valid and invalid solutions
        valid_solutions = self._get_valid_population(self.run_state_dict.population)
        invalid_solutions = [sol for sol in self.run_state_dict.population if sol not in valid_solutions]
        
        # Sort valid solutions by score (descending - higher is better)
        valid_solutions.sort(key=lambda x: x.evaluation_res.score if x.evaluation_res and x.evaluation_res.score is not None else float('-inf'), reverse=True)
        
        # Keep the best valid solutions + some invalid ones if needed
        new_population = []
        
        # First, add the best valid solutions
        valid_to_keep = min(len(valid_solutions), self.config.pop_size)
        new_population.extend(valid_solutions[:valid_to_keep])
        
        # If we need more individuals, add some invalid ones (most recent)
        remaining_slots = self.config.pop_size - len(new_population)
        if remaining_slots > 0 and invalid_solutions:
            new_population.extend(invalid_solutions[-remaining_slots:])  # Keep most recent invalid ones
        
        self.run_state_dict.population = new_population
        
        valid_count = len(self._get_valid_population(new_population))
        self.verbose_info(f"Population managed: {len(new_population)} total ({valid_count} valid, {len(new_population)-valid_count} invalid)")

    def _select_individuals_for_operator(self, operator) -> List[Solution]:
        """Select individuals for an operator based on its selection_size"""
        if operator.selection_size == 0:
            return []  # Init operators don't need individuals
            
        valid_population = [sol for sol in self.run_state_dict.population 
                          if sol.evaluation_res and sol.evaluation_res.valid and sol.evaluation_res.score is not None]
        
        if len(valid_population) == 0:
            # Fallback to any available solutions
            return self.run_state_dict.population[:operator.selection_size] if self.run_state_dict.population else []
        
        # Sort by score (assuming higher is better)
        valid_population.sort(key=lambda x: x.evaluation_res.score, reverse=True)
        
        # Select top individuals
        if len(valid_population) >= operator.selection_size:
            return valid_population[:operator.selection_size]
        else:
            return valid_population

    def _generate_single_solution(self, operator, selected_individuals: List[Solution], sampler_id: int) -> tuple[Solution, dict]:
        """Generate a single solution using an operator"""
        try:
            prompt_content = self.config.adapter.get_operator_prompt(operator.name, selected_individuals)
            response, usage = self.config.running_llm.get_response(prompt_content)
            new_sol = self.config.adapter.parse_response(response)
            self.verbose_info(f"Sampler {sampler_id}: Generated {operator.name} solution")
            return new_sol, usage
        except Exception as e:
            self.verbose_info(f"Sampler {sampler_id}: Failed to generate {operator.name} solution - {str(e)}")
            return Solution(""), {}

    def _get_run_state_class(self) -> Type[BaseRunStateDict]:
        return EvoEngineerRunStateDict