import concurrent.futures

from .run_config import FunSearchConfig
from .run_state_dict import FunSearchRunStateDict
from .programs_database import ProgramsDatabase
from ..base_method import Method
from evotool.task.base_task import Solution

class FunSearch(Method):
    def __init__(self, config: FunSearchConfig):
        super().__init__(config)
        self.config = config
    
    def run(self, run_state_dict: FunSearchRunStateDict):
        """Main FunSearch algorithm execution"""
        self.verbose_title("FUNSEARCH ALGORITHM STARTED")
        self._save_run_state(run_state_dict)
        
        # Initialize usage history
        if "sample" not in run_state_dict.usage_history:
            run_state_dict.usage_history["sample"] = []
        
        # Initialize or restore programs database
        if run_state_dict.has_database_state(self.config.output_path):
            # Restore from saved database file
            database_dict = run_state_dict.load_database_state(self.config.output_path)
            if database_dict:
                programs_db = ProgramsDatabase.from_dict(database_dict)
                self.verbose_info("Restored programs database from saved state")
            else:
                # Failed to load, create new database
                programs_db = ProgramsDatabase(
                    num_islands=self.config.num_islands,
                    solutions_per_prompt=self.config.programs_per_prompt,
                    reset_period=4 * 60 * 60,  # 4 hours
                )
                self.verbose_info("Failed to restore database, initialized new one")
        else:
            # Initialize new database
            programs_db = ProgramsDatabase(
                num_islands=self.config.num_islands,
                solutions_per_prompt=self.config.programs_per_prompt,
                reset_period=4 * 60 * 60,  # 4 hours
            )
            self.verbose_info("Initialized new programs database")
        
        # Initialize with seed program if sol_history is empty
        if len(run_state_dict.sol_history) == 0:
            initial_sol = self.config.adapter.make_init_sol()
            programs_db.register_solution(initial_sol)  # Register to all islands
            run_state_dict.sol_history.append(initial_sol)  # Add to sol_history but don't count in sample_nums
            
            # Save database state to separate file
            run_state_dict.save_database_state(programs_db.to_dict(), self.config.output_path)
            self._save_run_state(run_state_dict)
            
            self.verbose_info(f"Initialized with seed program (score: {initial_sol.evaluation_res.score if initial_sol.evaluation_res else 'None'})")
        else:
            self.verbose_info(f"Continuing from sample {run_state_dict.tot_sample_nums} with {len(run_state_dict.sol_history)} solutions in history")
            
            # Rebuild database from sol_history if needed
            if not run_state_dict.has_database_state(self.config.output_path):
                self.verbose_info("Rebuilding database from solution history...")
                for solution in run_state_dict.sol_history:
                    if solution.evaluation_res and solution.evaluation_res.valid:
                        programs_db.register_solution(solution)
        
        # Main sampling loop
        while ((self.config.max_sample_nums is None) or 
               (run_state_dict.tot_sample_nums < self.config.max_sample_nums)):
            try:
                start_sample = run_state_dict.tot_sample_nums + 1
                end_sample = run_state_dict.tot_sample_nums + self.config.num_samplers
                self.verbose_info(
                    f"Samples {start_sample} - {end_sample} / {self.config.max_sample_nums or 'unlimited'}"
                )
                
                # Get prompt solutions from random island
                prompt_solutions, island_id = programs_db.get_prompt_solutions()
                if not prompt_solutions:
                    self.verbose_info("No solutions available for prompting")
                    continue
                
                self.verbose_info(f"Selected {len(prompt_solutions)} solutions from island {island_id}")
                
                # Generate new programs using LLM
                new_programs = self._generate_programs(prompt_solutions, run_state_dict)
                
                # Evaluate programs
                evaluated_programs = self._evaluate_programs(new_programs)
                
                # Process all evaluated programs
                for program in evaluated_programs:
                    # Add ALL programs (valid/invalid) to sol_history
                    run_state_dict.sol_history.append(program)
                    run_state_dict.tot_sample_nums += 1
                    
                    # Only register valid programs to the database/island
                    if program.evaluation_res and program.evaluation_res.valid:
                        programs_db.register_solution(program, island_id)
                        
                        score_str = f"{program.evaluation_res.score:.6f}" if program.evaluation_res.score is not None else "None"
                        self.verbose_info(f"Registered valid program to island {island_id} (score: {score_str})")
                    else:
                        self.verbose_info(f"Added invalid program to history (sample {run_state_dict.tot_sample_nums})")
                
                # Log current best
                best_solution = programs_db.get_best_solution()
                if best_solution and best_solution.evaluation_res:
                    best_score_str = f"{best_solution.evaluation_res.score:.6f}" if best_solution.evaluation_res.score is not None else "None"
                    self.verbose_info(f"Current best score: {best_score_str}")
                
                # Show database statistics periodically
                if run_state_dict.tot_sample_nums % 50 == 0:
                    stats = programs_db.get_statistics()
                    self.verbose_info(f"Database stats: {stats['total_programs']} total programs, {stats['num_islands']} islands, best score: {stats['global_best_score']:.6f}")
                
                # Save database state to separate file
                run_state_dict.save_database_state(programs_db.to_dict(), self.config.output_path)
                self._save_run_state(run_state_dict)
                
            except KeyboardInterrupt:
                self.verbose_info("Interrupted by user")
                break
            except Exception as e:
                self.verbose_info(f"Sampling error: {str(e)}")
                continue
        
        # Mark as done and save final state with database
        run_state_dict.is_done = True
        run_state_dict.save_database_state(programs_db.to_dict(), self.config.output_path)
        self._save_run_state(run_state_dict)
        
        # Log final statistics
        final_stats = programs_db.get_statistics()
        self.verbose_info(f"Final stats: {final_stats['total_programs']} programs, best score: {final_stats['global_best_score']:.6f}")
        
        # Show database file location
        self.verbose_info(f"Programs database saved to: {run_state_dict.database_file}")
    
    def _generate_programs(self, prompt_solutions, run_state_dict) -> list[Solution]:
        """Generate new programs using LLM based on prompt solutions"""
        new_programs = []
        
        # Multi-threaded program generation
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.num_samplers) as executor:
            futures = []
            
            # Generate multiple programs per sampler
            for sampler_id in range(self.config.num_samplers):
                future = executor.submit(self._generate_single_program, prompt_solutions, sampler_id)
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    new_program, usage = future.result()
                    run_state_dict.usage_history["sample"].append(usage)
                    new_programs.append(new_program)
                except Exception as e:
                    self.verbose_info(f"Program generation failed: {str(e)}")
                    continue
        
        return new_programs
    
    def _generate_single_program(self, prompt_solutions: list[Solution], sampler_id: int) -> tuple[Solution, dict]:
        """Generate single program variant using LLM based on prompt solutions"""
        usage = {}
        try:
            # Get prompt from adapter based on selected solutions
            prompt_content = self.config.adapter.get_prompt(prompt_solutions)
            
            response, usage = self.config.running_llm.get_response(prompt_content)
            
            parsed_response = self.config.adapter.parse_response(response)
            
            new_sol = Solution(parsed_response)
            
            self.verbose_info(f"Sampler {sampler_id}: Generated a program variant.")
            return new_sol, usage
        except Exception as e:
            self.verbose_info(f"Sampler {sampler_id}: Failed to generate program - {str(e)}")
            return Solution(""), usage  # Return usage even if failed
    
    def _evaluate_programs(self, programs: list[Solution]) -> list[Solution]:
        # Parallel evaluation using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.num_evaluators) as executor:
            futures = []
            for i, program in enumerate(programs):
                future = executor.submit(self.config.evaluator.evaluate_code, program.sol_string)
                futures.append((future, i))
            
            # Collect results
            for future, i in futures:
                try:
                    evaluation_res = future.result()
                    programs[i].evaluation_res = evaluation_res
                    score_str = "None" if evaluation_res.score is None else f"{evaluation_res.score}"
                    self.verbose_info(f"Program evaluated - Score: {score_str}")
                except Exception as e:
                    self.verbose_info(f"Evaluation failed: {str(e)}")
                    continue
        
        return programs