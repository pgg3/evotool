import concurrent.futures

from .run_config import Es1p1Config
from .run_state_dict import Es1p1RunStateDict
from ..base_method import Method
from evotool.task.base_task import Solution

class Es1p1(Method):
    def __init__(self, config: Es1p1Config):
        super().__init__(config)
        self.config = config
    
    def run(self, run_state_dict: Es1p1RunStateDict):
        """Main ES(1+1) algorithm execution"""
        self.verbose_title("ES(1+1) ALGORITHM STARTED")
        self._save_run_state(run_state_dict)

        if "sample" not in run_state_dict.usage_history:
            run_state_dict.usage_history["sample"] = []

        if len(run_state_dict.sol_history) == 0:
            run_state_dict.sol_history.append(self.config.adapter.make_init_sol())
            self._save_run_state(run_state_dict)
        
        # Main evolution loop
        while ((self.config.max_sample_nums is None) or 
               (run_state_dict.tot_sample_nums < self.config.max_sample_nums)):
            try:
                start_sample = run_state_dict.tot_sample_nums + 1
                end_sample = run_state_dict.tot_sample_nums + self.config.num_samplers
                self.verbose_info(
                    f"Samples  {start_sample} - {end_sample} / {self.config.max_sample_nums or 'unlimited'} "
                )

                best_sol = self._get_best_valid_sol(run_state_dict.sol_history)

                # Multi-threaded propose and evaluate
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.num_samplers) as propose_executor:
                    # Submit multiple propose tasks in parallel
                    propose_futures = []
                    for i in range(self.config.num_samplers):
                        future = propose_executor.submit(self._propose_sample, best_sol, i)
                        propose_futures.append(future)
                    
                    # Collect all proposed samples
                    sol_to_eval = []
                    
                    for future in concurrent.futures.as_completed(propose_futures):
                        new_sol, usage = future.result()
                        run_state_dict.usage_history["sample"].append(usage)
                        sol_to_eval.append(new_sol)

                # Parallel evaluation using ThreadPoolExecutor
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.num_evaluators) as eval_executor:
                    futures = []
                    for i, sol in enumerate(sol_to_eval):
                        future = eval_executor.submit(self.config.evaluator.evaluate_code, sol.sol_string)
                        futures.append((future, i))
                    
                    # Collect results and update best
                    for future, i in futures:
                        try:
                            evaluation_res = future.result()
                            sol_to_eval[i].evaluation_res = evaluation_res
                            score_str = "None" if evaluation_res.score is None else f"{evaluation_res.score}"
                            self.verbose_info(f"Sample evaluated - Score: {score_str}")
                            
                            # Add to history
                            run_state_dict.sol_history.append(sol_to_eval[i])
                            run_state_dict.tot_sample_nums += 1
                            self._save_run_state(run_state_dict)
                            
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
        run_state_dict.is_done = True
        self._save_run_state(run_state_dict)
    
    def _propose_sample(self, best_sol: Solution, sampler_id: int) -> tuple[Solution, dict]:
        try:
            prompt_content = self.config.adapter.get_prompt(best_sol)

            response, usage = self.config.running_llm.get_response(prompt_content)

            parsed_response = self.config.adapter.parse_response(response)

            new_sol = Solution(parsed_response)

            self.verbose_info(f"Sampler {sampler_id}: Generated a sample.")
            return new_sol, usage
        except Exception as e:
            self.verbose_info(f"Sampler {sampler_id}: Failed to generate a samples - {str(e)}")
            return Solution(""), usage

