import concurrent.futures
from typing import Type

from .run_config import Es1p1Config
from .run_state_dict import Es1p1RunStateDict
from ..base_method import Method
from evotool.task.base_task import Solution
from ..base_run_state_dict import BaseRunStateDict


class Es1p1(Method):
    def __init__(self, config: Es1p1Config):
        super().__init__(config)
        self.config = config
    
    def run(self):
        """Main ES(1+1) algorithm execution"""
        self.verbose_title("ES(1+1) ALGORITHM STARTED")

        if "sample" not in self.run_state_dict.usage_history:
            self.run_state_dict.usage_history["sample"] = []

        if len(self.run_state_dict.sol_history) == 0:
            # Try to create and evaluate initial solution up to 3 times
            init_sol = None
            for attempt in range(3):
                try:
                    candidate_sol = self.config.adapter.make_init_sol()
                    if candidate_sol.evaluation_res is None:
                        candidate_sol.evaluation_res = self.config.evaluator.evaluate_code(candidate_sol.sol_string)
                    
                    if candidate_sol.evaluation_res and candidate_sol.evaluation_res.valid:
                        init_sol = candidate_sol
                        break
                    else:
                        self.verbose_info(f"Initial solution attempt {attempt + 1} failed: invalid evaluation result")
                except Exception as e:
                    self.verbose_info(f"Initial solution attempt {attempt + 1} failed with exception: {e}")
            
            if init_sol is None:
                print("Warning: Failed to create valid initial solution after 3 attempts. Exiting.")
                return
            
            self.run_state_dict.sol_history.append(init_sol)
            self._save_run_state_dict()
        
        # Main evolution loop
        while self.run_state_dict.tot_sample_nums < self.config.max_sample_nums:
            try:
                start_sample = self.run_state_dict.tot_sample_nums + 1
                end_sample = self.run_state_dict.tot_sample_nums + self.config.num_samplers
                self.verbose_info(
                    f"Samples  {start_sample} - {end_sample} / {self.config.max_sample_nums or 'unlimited'} "
                )

                best_sol = self._get_best_sol(self.run_state_dict.sol_history)

                # Async propose and evaluate - single executor for both
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.num_samplers + self.config.num_evaluators) as executor:
                    # Submit all propose tasks
                    propose_futures = []
                    eval_futures = []
                    
                    for i in range(self.config.num_samplers):
                        future = executor.submit(self._propose_sample, best_sol, i)
                        propose_futures.append(future)
                    
                    # Process proposals as they complete and immediately submit for evaluation
                    for future in concurrent.futures.as_completed(propose_futures):
                        try:
                            new_sol, usage = future.result()
                            self.run_state_dict.usage_history["sample"].append(usage)
                            
                            # Immediately submit for evaluation without waiting
                            eval_future = executor.submit(self.config.evaluator.evaluate_code, new_sol.sol_string)
                            eval_futures.append((eval_future, new_sol))
                        except Exception as e:
                            self.verbose_info(f"Propose failed: {str(e)}")
                            continue
                    
                    # Collect evaluation results as they complete
                    for eval_future, sol in eval_futures:
                        try:
                            evaluation_res = eval_future.result()
                            sol.evaluation_res = evaluation_res
                            score_str = "None" if evaluation_res.score is None else f"{evaluation_res.score}"
                            self.verbose_info(f"Sample evaluated - Score: {score_str}")
                            
                            # Add to history
                            self.run_state_dict.sol_history.append(sol)
                            self.run_state_dict.tot_sample_nums += 1
                            self._save_run_state_dict()
                        except Exception as e:
                            self.verbose_info(f"Evaluation failed: {str(e)}")
                            continue

                            

            except KeyboardInterrupt:
                self.verbose_info("Interrupted by user")
                break
            except Exception as e:
                self.verbose_info(f"Sampling error: {str(e)}")
                continue
        
        # Mark as done and save final state
        self.run_state_dict.is_done = True
        self._save_run_state_dict()
    
    def _propose_sample(self, best_sol: Solution, sampler_id: int) -> tuple[Solution, dict]:
        try:
            prompt_content = self.config.adapter.get_prompt(best_sol)

            response, usage = self.config.running_llm.get_response(prompt_content)

            new_sol = self.config.adapter.parse_response(response)

            self.verbose_info(f"Sampler {sampler_id}: Generated a sample.")
            return new_sol, usage
        except Exception as e:
            self.verbose_info(f"Sampler {sampler_id}: Failed to generate a samples - {str(e)}")
            return Solution(""), {}

    def _get_run_state_class(self) -> Type[BaseRunStateDict]:
        return Es1p1RunStateDict

