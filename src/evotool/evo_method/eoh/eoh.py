import concurrent.futures
from typing import List

from .run_config import EohConfig
from .run_state_dict import EohRunStateDict
from ..base_method import Method
from evotool.task.base_task import Solution

class Eoh(Method):
    def __init__(self, config: EohConfig):
        super().__init__(config)
        self.config = config

    def run(self):
        """Main EoH algorithm execution"""
        self.verbose_title("EOH ALGORITHM STARTED")

        if "sample" not in self.run_state_dict.usage_history:
            self.run_state_dict.usage_history["sample"] = []

        # Initialize with seed solution if sol_history is empty
        if len(self.run_state_dict.sol_history) == 0:
            initial_sol = self.config.adapter.make_init_sol()
            self.run_state_dict.sol_history.append(initial_sol)
            self.run_state_dict.population.append(initial_sol)
            self._save_run_state_dict()
            self.verbose_info(f"Initialized with baseline solution (score: {initial_sol.evaluation_res.score if initial_sol.evaluation_res else 'None'})")

        # Initialize population if starting from scratch
        if self.run_state_dict.generation == 0:
            self._initialize_population()
            
        # Check if we have enough individuals for selection
        valid_population = self._get_valid_population(self.run_state_dict.population)
        if len(valid_population) < self.config.selection_num:
            self.verbose_info(f'The search is terminated since EoH unable to obtain {self.config.selection_num} feasible algorithms during initialization.')
            return
        
        # Main evolution loop - moved loop control logic here
        while (self.run_state_dict.generation < self.config.max_generations) and (self.run_state_dict.tot_sample_nums < self.config.max_sample_nums):
            try:
                self.verbose_info(f"Generation {self.run_state_dict.generation} - Sample {self.run_state_dict.tot_sample_nums + 1} - {self.run_state_dict.tot_sample_nums + self.config.num_samplers} / {self.config.max_sample_nums or 'unlimited'}")
                
                # Apply operators in parallel for this generation
                new_solutions = self._apply_operators_parallel()
                
                # Add new solutions to both sol_history and population
                for sol in new_solutions:
                    self.run_state_dict.sol_history.append(sol)
                    self.run_state_dict.population.append(sol)
                    self.run_state_dict.tot_sample_nums += 1
                
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
        """Initialize population using i1 prompt - keep generating until we have enough valid solutions"""
        self.verbose_info("Initializing population...")
        
        initial_sample_limit = min(self.config.max_sample_nums or float('inf'), 200)  # Reasonable limit
        
        # Keep generating until we have pop_size valid solutions or hit sample limit
        while (len(self._get_valid_population(self.run_state_dict.population)) < self.config.pop_size and 
               self.run_state_dict.tot_sample_nums < initial_sample_limit):
            
            # Generate multiple solutions in parallel
            new_solutions = self._generate_initial_solutions()
            
            # Evaluate solutions in parallel
            evaluated_solutions = self._evaluate_solutions(new_solutions)
            
            # Add all solutions to both sol_history and population
            for sol in evaluated_solutions:
                self.run_state_dict.sol_history.append(sol)
                self.run_state_dict.population.append(sol)
                self.run_state_dict.tot_sample_nums += 1
                
                score_str = "None" if not sol.evaluation_res or sol.evaluation_res.score is None else f"{sol.evaluation_res.score}"
                valid_str = "Valid" if sol.evaluation_res and sol.evaluation_res.valid else "Invalid"
                self.verbose_info(f"Initial sample {self.run_state_dict.tot_sample_nums} - Score: {score_str} ({valid_str})")
            
            valid_count = len(self._get_valid_population(self.run_state_dict.population))
            self.verbose_info(f"Valid solutions: {valid_count}/{self.config.pop_size}")
            
            self._save_run_state_dict()
        
        valid_population = self._get_valid_population(self.run_state_dict.population)
        if len(valid_population) >= self.config.selection_num:
            self.run_state_dict.generation = 1
            self._save_run_state_dict()
            self.verbose_info(f"Initialization completed with {len(valid_population)} valid solutions")
        else:
            self.verbose_info(f"Warning: Only {len(valid_population)} valid solutions obtained, need at least {self.config.selection_num}")

    def _generate_initial_solutions(self) -> List[Solution]:
        """Generate initial solutions using multithreading"""
        new_solutions = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.num_samplers) as executor:
            futures = []
            
            # Generate multiple solutions in parallel
            for sampler_id in range(self.config.num_samplers):
                future = executor.submit(self._generate_single_initial_solution, sampler_id)
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    new_sol, usage = future.result()
                    self.run_state_dict.usage_history["sample"].append(usage)
                    new_solutions.append(new_sol)
                except Exception as e:
                    self.verbose_info(f"Initial solution generation failed: {str(e)}")
                    continue
        
        return new_solutions
    
    def _generate_single_initial_solution(self, sampler_id: int) -> tuple[Solution, dict]:
        """Generate a single initial solution"""
        try:
            prompt_content = self.config.adapter.get_prompt_i1()
            response, usage = self.config.running_llm.get_response(prompt_content)
            parsed_response, algorithm = self.config.adapter.parse_response(response)
            other_info = {"algorithm": algorithm}
            new_sol = Solution(parsed_response, other_info=other_info)
            self.verbose_info(f"Sampler {sampler_id}: Generated initial solution")
            return new_sol, usage
        except Exception as e:
            self.verbose_info(f"Sampler {sampler_id}: Failed to generate initial solution - {str(e)}")
            return Solution(""), {}
    
    def _evaluate_solutions(self, solutions: List[Solution]) -> List[Solution]:
        """Evaluate solutions using multithreading"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.num_evaluators) as executor:
            futures = []
            for i, solution in enumerate(solutions):
                if solution.sol_string.strip():  # Only evaluate non-empty solutions
                    future = executor.submit(self.config.evaluator.evaluate_code, solution.sol_string)
                    futures.append((future, i))
            
            # Collect results
            for future, i in futures:
                try:
                    evaluation_res = future.result()
                    solutions[i].evaluation_res = evaluation_res
                except Exception as e:
                    self.verbose_info(f"Evaluation failed: {str(e)}")
                    continue
        
        return solutions
    
    def _get_valid_population(self, population: List[Solution]) -> List[Solution]:
        """Get valid solutions from population"""
        return [sol for sol in population if sol.evaluation_res and sol.evaluation_res.valid]

    def _get_best_valid_sol(self, sol_history: List[Solution]) -> Solution:
        """Get the best valid solution from sol_history"""
        valid_sols = [sol for sol in sol_history if sol.evaluation_res and sol.evaluation_res.valid and sol.evaluation_res.score is not None]
        if valid_sols:
            return max(valid_sols, key=lambda x: x.evaluation_res.score)
        return sol_history[-1] if sol_history else None


    def _apply_operators_parallel(self) -> List[Solution]:
        """Apply all operators in parallel and return new solutions"""
        new_solutions = []
        
        # Prepare operator tasks
        operator_tasks = []
        
        # E1 operator - crossover
        selected_individuals = self._select_individuals(self.config.selection_num)
        if selected_individuals:
            prompt_content = self.config.adapter.get_prompt_e1(selected_individuals)
            operator_tasks.append(("E1", prompt_content))
        
        # E2 operator - guided crossover
        if self.config.use_e2_operator:
            selected_individuals = self._select_individuals(self.config.selection_num)
            if selected_individuals:
                prompt_content = self.config.adapter.get_prompt_e2(selected_individuals)
                operator_tasks.append(("E2", prompt_content))
        
        # M1 operator - mutation
        if self.config.use_m1_operator:
            selected_individuals = self._select_individuals(1)
            if selected_individuals:
                selected_individual = selected_individuals[0]
                prompt_content = self.config.adapter.get_prompt_m1(selected_individual)
                operator_tasks.append(("M1", prompt_content))
        
        # M2 operator - parameter mutation
        if self.config.use_m2_operator:
            selected_individuals = self._select_individuals(1)
            if selected_individuals:
                selected_individual = selected_individuals[0]
                prompt_content = self.config.adapter.get_prompt_m2(selected_individual)
                operator_tasks.append(("M2", prompt_content))
        
        # Execute operators in parallel using num_samplers
        if operator_tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.num_samplers) as executor:
                futures = []
                
                # Submit tasks up to num_samplers limit
                for i, (operator_name, prompt_content) in enumerate(operator_tasks[:self.config.num_samplers]):
                    future = executor.submit(self._generate_single_operator_solution, prompt_content, operator_name, i)
                    futures.append((operator_name, future))
                
                # Collect results
                for operator_name, future in futures:
                    try:
                        solution, usage = future.result()
                        
                        # Add usage history
                        self.run_state_dict.usage_history["sample"].append(usage)
                        
                        # Evaluate solution
                        if solution.sol_string.strip():
                            evaluation_res = self.config.evaluator.evaluate_code(solution.sol_string)
                            solution.evaluation_res = evaluation_res
                        
                        new_solutions.append(solution)
                        
                        # Log result
                        score_str = "None" if not solution.evaluation_res or solution.evaluation_res.score is None else f"{solution.evaluation_res.score}"
                        valid_str = "Valid" if solution.evaluation_res and solution.evaluation_res.valid else "Invalid"
                        self.verbose_info(f"{operator_name} Gen {self.run_state_dict.generation} - Score: {score_str} ({valid_str})")
                        
                    except Exception as e:
                        self.verbose_info(f"Error in {operator_name}: {str(e)}")
                        continue
        
        return new_solutions
    

    def _manage_population_size(self, run_state_dict: EohRunStateDict):
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



    def _select_individuals(self, num_select: int) -> List[Solution]:
        """Select individuals from population for operators"""
        valid_population = [sol for sol in self.run_state_dict.population 
                          if sol.evaluation_res and sol.evaluation_res.valid and sol.evaluation_res.score is not None]
        
        if len(valid_population) == 0:
            # Fallback to any available solutions
            return self.run_state_dict.population[:num_select] if self.run_state_dict.population else []
        
        # Sort by score (assuming higher is better)
        valid_population.sort(key=lambda x: x.evaluation_res.score, reverse=True)
        
        # Tournament selection or simple top selection
        if len(valid_population) >= num_select:
            return valid_population[:num_select]
        else:
            return valid_population

    
    def _generate_single_operator_solution(self, prompt_content: List[dict], operator_type: str, sampler_id: int) -> tuple[Solution, dict]:
        """Generate a single solution for an operator"""
        try:
            response, usage = self.config.running_llm.get_response(prompt_content)
            parsed_response, algorithm = self.config.adapter.parse_response(response)
            other_info = {"algorithm": algorithm}
            new_sol = Solution(parsed_response, other_info=other_info)
            self.verbose_info(f"Sampler {sampler_id}: Generated {operator_type} solution")
            return new_sol, usage
        except Exception as e:
            self.verbose_info(f"Sampler {sampler_id}: Failed to generate {operator_type} solution - {str(e)}")
            return Solution(""), {}