import os
import time
import numpy as np
from concurrent import futures

from .prompt_maker import PromptMaker
from .response_parser import ResponseParser
from .run_state_dict import AiCudaEngineerRunStateDict
from .run_config import AiCudaEngineerConfig
from evotool.evo_method.base_method import Method

class AiCudaEngineer(Method):
    def __init__(self, config:AiCudaEngineerConfig):
        super().__init__(config)
        self.config = config

    def run(self, run_state_dict: AiCudaEngineerRunStateDict, hist_best_kernel_list):
        self.verbose_title("AI CUDA ENGINEER STARTED")
        while not run_state_dict.is_done:
            if run_state_dict.run_stage == "0":
                self.conversion(run_state_dict)
            elif run_state_dict.run_stage == "1":
                self.translation(run_state_dict)
            elif run_state_dict.run_stage == "2":
                self.evolution(run_state_dict)
            elif run_state_dict.run_stage == "3":
                self.rag_evolution(run_state_dict, hist_best_kernel_list)
            else:
                return

    def conversion(self, run_state_dict: AiCudaEngineerRunStateDict):
        self.verbose_stage("Stage 0: Converting the function into functional code")
        if run_state_dict.task_info["func_py_code"] is None:
            parsed_convert_response = None
            error_msg = None
            error_restart = 0
            convert_success = False
            run_state_dict.usage_history["0"] = []
            self.verbose_info("Converting the function into functional code...")
            while not convert_success:
                convert_prompt = PromptMaker.make_convert_prompt(run_state_dict.task_info["org_py_code"], parsed_convert_response, error_msg)
                convert_response, convert_usage = self.config.conversion_llm.get_response(convert_prompt)
                parsed_convert_response = ResponseParser.parse_convert_response(convert_response)
                run_state_dict.usage_history["0"].append(convert_usage)
                convert_result_dict = self.config.evaluator.compare_py_code_sandbox(run_state_dict.task_info["org_py_code"], parsed_convert_response)
                convert_success, error_msg = convert_result_dict["correctness"], convert_result_dict["error_msg"]

                if not convert_success:
                    error_restart += 1
                    self.verbose_info("Conversion failed, retrying...")

                if error_restart % 3 == 0:
                    error_msg = None

                if error_restart > 10:
                    run_state_dict.run_stage = "4"
                    run_state_dict.is_done = True
                    self._save_run_state(run_state_dict)
                    return
            self.verbose_info("Conversion successful!")
            run_state_dict.task_info["func_py_code"] = parsed_convert_response
            run_state_dict.run_stage = "1"
            self._save_run_state(run_state_dict)
        else:
            self.verbose_info("Conversion successful! (func_file already converted)")
            run_state_dict.run_stage = "1"
            self._save_run_state(run_state_dict)
            return

    def translation(self, run_state_dict: AiCudaEngineerRunStateDict):
        self.verbose_stage("Stage 1: Translating the functional code into cuda code")
        if run_state_dict.task_info["cuda_code"] is None:
            self.verbose_info("Translating the functional code...")
            translate_success = False
            parsed_translate_response = None
            error_msg = None
            error_summary = None
            error_restart = 0
            run_state_dict.usage_history["1"] = []
            while not translate_success:
                if error_msg is not None:
                    error_summary_prompt = PromptMaker.make_translate_error_summary_prompt(
                        run_state_dict.task_info["func_py_code"],
                        parsed_translate_response,
                        error_msg
                    )
                    error_summary, error_summary_usage = self.config.translation_llm.get_response(error_summary_prompt)
                    run_state_dict.usage_history["1"].append(error_summary_usage)
                cuda_code_prompt = PromptMaker.make_translate_prompt(
                    run_state_dict.task_info["func_py_code"],
                    parsed_translate_response,
                    error_msg,
                    error_summary)
                translate_response, translate_usage = self.config.translation_llm.get_response(cuda_code_prompt)
                run_state_dict.usage_history["1"].append(translate_usage)
                parsed_translate_response = ResponseParser.parse_translate_response(translate_response)
                evaluate_cuda_dict = self.config.evaluator.compare_func_cuda_sandbox(
                    run_state_dict.task_info["func_py_code"], parsed_translate_response
                )
                translate_success, error_msg = evaluate_cuda_dict["correctness"], evaluate_cuda_dict["error_msg"]

                if not translate_success:
                    error_restart += 1
                    self.verbose_info("Translation failed, retrying...")

                if error_restart % 3 == 0:
                    error_msg = None

                if error_restart > 10:
                    run_state_dict.run_stage = "4"
                    run_state_dict.is_done = True
                    self._save_run_state(run_state_dict)
                    return
            self.verbose_info("Translation successful!")
            run_state_dict.task_info["cuda_code"] = parsed_translate_response
            run_state_dict.run_stage = "2"
            self._save_run_state(run_state_dict)
        else:
            self.verbose_info("Translation successful! (cuda_file already converted)")
            run_state_dict.run_stage = "2"
            self._save_run_state(run_state_dict)
            return

    def evolution(self, run_state_dict: AiCudaEngineerRunStateDict):
        self.verbose_stage("Stage 3: Evolving the cuda code")
        # make necessary task info, assuming that func and cuda code is right
        run_state_dict.task_info["func_runtime"] = run_state_dict.task_info.get(
            "func_runtime", self.config.evaluator.get_py_runtime_sandbox(run_state_dict.task_info["func_py_code"])["runtime"]
        )
        if "cuda_info" not in run_state_dict.task_info:
            cuda_info_dict = self.config.evaluator.get_cuda_runtime_sandbox(
                run_state_dict.task_info["func_py_code"],
                run_state_dict.task_info["cuda_code"]
            )
            run_state_dict.task_info["cuda_info"] = {
                "name": "baseline",
                "thought": "baseline",
                "code": run_state_dict.task_info["cuda_code"],
                "temp_str": cuda_info_dict["temp_str"],
                "runtime": cuda_info_dict["runtime"],
                "prof_string": cuda_info_dict["prof_string"],
                "compilation_error": False,
                "comparison_error": False
            }
            self._save_run_state(run_state_dict)

        self.verbose_info(
            f"Torch runtime: {run_state_dict.task_info['func_runtime']:.4f} ms, "
            f"CUDA runtime: {run_state_dict.task_info['cuda_info']['runtime']:.4f} ms"
        )

        if len(run_state_dict.optimization_history) == 0:
            run_state_dict.optimization_history.append(run_state_dict.task_info["cuda_info"])
            self._save_run_state(run_state_dict)
        if "2" not in run_state_dict.usage_history:
            run_state_dict.usage_history["2"] = []

        for gen_i in range(run_state_dict.evo_gen_i, 10):
            self.verbose_gen(f"Gen {gen_i + 1}")
            top_5_kernel = self._get_valid_top_5_from_slow_to_fast(run_state_dict.optimization_history)
            best_kernel = self._get_best_valid_kernel(run_state_dict.optimization_history)
            
            # Use best kernel or baseline for optimization
            cuda_individual = best_kernel if best_kernel else run_state_dict.task_info["cuda_info"]

            # Parallel processing with evo_llm_list
            llm_list_len = len(self.config.evo_llm_list)
            
            with futures.ThreadPoolExecutor(max_workers=llm_list_len) as executor:
                # Submit tasks to each LLM - each thread handles both propose AND evaluate
                future_to_index = {}
                for i in range(llm_list_len):
                    future = executor.submit(self._process_llm_proposal_and_evaluate, i, run_state_dict, top_5_kernel, cuda_individual)
                    future_to_index[future] = i
                
                # Collect completed proposal+evaluation results
                for future in futures.as_completed(future_to_index):
                    i = future_to_index[future]
                    try:
                        new_entry, usage = future.result()
                        run_state_dict.usage_history["2"].append(usage)
                        if new_entry is not None:
                            run_state_dict.optimization_history.append(new_entry)
                            runtime_str = f"{new_entry['runtime']:.4f} ms" if new_entry['runtime'] is not None else "Failed"
                            self.verbose_info(f"LLM {i}: {new_entry['name']}, runtime: {runtime_str}, temp_str: {new_entry['temp_str']}")
                        else:
                            self.verbose_info(f"LLM {i}: Failed to generate valid proposal")

                    except Exception as e:
                        self.verbose_info(f"LLM {i}: Error processing proposal - {str(e)}")

            # Update generation info
            run_state_dict.evo_gen_i = gen_i + 1
            
            self.verbose_info(f"Generation {gen_i + 1} completed. Total entries: {len(run_state_dict.optimization_history)}")
            
            self._save_run_state(run_state_dict)

        run_state_dict.run_stage = "3"
        self._save_run_state(run_state_dict)

    def rag_evolution(self, run_state_dict: AiCudaEngineerRunStateDict, hist_best_kernel_list):
        self.verbose_stage("Stage 4: RAG Evolving the cuda code")
        embedding_llm = self.config.embedding_llm
        if hist_best_kernel_list:
            embedding_database_list = [embedding_llm.get_embedding(term["task_info"]["func_py_code"]) for term in
                                       hist_best_kernel_list]
            current_embedding = embedding_llm.get_embedding(
                run_state_dict.task_info["func_py_code"]
            )

            embedding_database_list = np.array(embedding_database_list)  # N x 1536
            current_embedding = np.array(current_embedding)[np.newaxis, :]  # 1 x 1536

            # Calculate cosine similarity and get top-k similar codes
            similarities = [
                np.dot(current_embedding, db_emb.T) / (np.linalg.norm(current_embedding) * np.linalg.norm(db_emb))
                for db_emb in embedding_database_list]
            top_k_indices = np.argsort(similarities)[-5:][::-1].reshape(-1)  # Get top 5 similar codes
            similar_codes = [hist_best_kernel_list[i] for i in top_k_indices]
        else:
            similar_codes = []

        # Use similar pattern as evolution method but with RAG prompt
        best_kernel = self._get_best_valid_kernel(run_state_dict.optimization_history)
        cuda_individual = best_kernel if best_kernel else run_state_dict.task_info["cuda_info"]
        
        if "3" not in run_state_dict.usage_history:
            run_state_dict.usage_history["3"] = []

        RAG_TIMES = 5
        # Parallel processing with rag_llm for RAG evolution
        
        with futures.ThreadPoolExecutor(max_workers=RAG_TIMES) as executor:
            # Submit RAG_TIMES tasks - each thread handles both RAG propose AND evaluate
            future_to_index = {}
            for i in range(RAG_TIMES):
                future = executor.submit(self._process_rag_proposal_and_evaluate, i, run_state_dict, similar_codes, cuda_individual)
                future_to_index[future] = i
            
            # Collect completed RAG proposal+evaluation results
            for future in futures.as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    new_entry, usage = future.result()
                    run_state_dict.usage_history["3"].append(usage)
                    if new_entry is not None:
                        run_state_dict.optimization_history.append(new_entry)
                        runtime_str = f"{new_entry['runtime']:.4f} ms" if new_entry['runtime'] is not None else "Failed"
                        self.verbose_info(f"RAG {i}: {new_entry['name']}, runtime: {runtime_str}, temp_str: {new_entry['temp_str']}")
                    else:
                        self.verbose_info(f"RAG {i}: Failed to generate valid proposal")
                except Exception as e:
                    self.verbose_info(f"RAG {i}: Error processing proposal - {str(e)}")
        
        run_state_dict.run_stage = "4"
        run_state_dict.is_done = True
        self._save_run_state(run_state_dict)
    
    def _process_rag_proposal_and_evaluate(self, task_index, run_state_dict, similar_codes, cuda_individual):
        """Process a single RAG LLM proposal AND evaluate it completely in one thread."""
        # Use RAG prompt with similar codes as optimization history
        prompt = PromptMaker.make_rag_prompt(
            gpu_type=run_state_dict.task_info['gpu_type'],
            cuda_version=run_state_dict.task_info['cuda_version'], 
            optimization_history=similar_codes,
            func_runtime=run_state_dict.task_info["func_runtime"],
            cuda_indiv=cuda_individual
        )
        return self._process_proposal_and_evaluate_common(task_index, run_state_dict, prompt, "RAG", use_rag_llm=True)
    
    def _process_proposal_and_evaluate_common(self, llm_index, run_state_dict, prompt, prompt_type, use_rag_llm=False):
        """Common function for processing LLM proposals and evaluating them."""
        try:
            # Step 1: Get response from LLM
            if use_rag_llm:
                response, usage = self.config.rag_llm.get_response(prompt)
            else:
                response, usage = self.config.evo_llm_list[llm_index].get_response(prompt)
            
            # Step 2: Parse the response
            parsed_response = ResponseParser.parse_evo_response(response)
            
            # Step 3: Create new entry structure
            new_entry = {
                "name": parsed_response["name"],
                "thought": parsed_response["thought"],
                "code": parsed_response["code"],
                "temp_str": None,
                "runtime": None,
                "prof_string": None,
                "compilation_error": False,
                "comparison_error": False
            }
            
            # Step 4: Evaluate CUDA code correctness
            cuda_comparison_result = self.config.evaluator.compare_func_cuda_sandbox(
                run_state_dict.task_info["func_py_code"],
                parsed_response["code"]
            )
            new_entry["temp_str"] = cuda_comparison_result.get("temp_str")

            # Step 5: If correct, measure runtime performance
            if cuda_comparison_result["correctness"]:
                cuda_runtime_result = self.config.evaluator.get_cuda_runtime_sandbox(
                    run_state_dict.task_info["func_py_code"],
                    parsed_response["code"],
                    cuda_comparison_result.get("temp_str")
                )
                new_entry["runtime"] = cuda_runtime_result["runtime"]
                new_entry["prof_string"] = cuda_runtime_result["prof_string"]
            else:
                new_entry["comparison_error"] = True
                new_entry["compilation_error"] = cuda_comparison_result["compilation_error"]
            
            return new_entry, usage
            
        except Exception as e:
            self.verbose_info(f"{prompt_type} LLM {llm_index}: Exception in proposal generation and evaluation - {str(e)}")
            return {
                "name": "failed_proposal",
                "thought": f"Failed due to exception: {str(e)}",
                "code": "",
                "temp_str": None,
                "runtime": None,
                "prof_string": None,
                "compilation_error": True,
                "comparison_error": True
            }, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _save_run_state(self, run_state_dict: AiCudaEngineerRunStateDict):
        run_state_dict.to_json_file(os.path.join(self.config.output_path, "run_state.json"))

    def _get_valid_top_5_from_slow_to_fast(self, optimization_history):
        valid_individuals = []
        for individual in optimization_history:
            if individual.get("runtime") is not None and individual["runtime"] != float('inf'):
                valid_individuals.append(individual)
        
        # Step 2: Sort from quickest to slowest and take top 5
        valid_individuals.sort(key=lambda x: x["runtime"])
        top_5_fastest = valid_individuals[:5]
        
        # Step 3: Reverse to get from slow to fast (slowest of the top 5 first)
        top_5_slow_to_fast = top_5_fastest[::-1]
        
        return top_5_slow_to_fast

    def _get_best_valid_kernel(self, optimization_history):
        """Get the best performing valid kernel from optimization history."""
        valid_kernels = []
        for kernel in optimization_history:
            if kernel.get("runtime") is not None and kernel["runtime"] != float('inf'):
                valid_kernels.append(kernel)
        
        if not valid_kernels:
            return None
        
        # Return the kernel with minimum runtime
        best_kernel = min(valid_kernels, key=lambda x: x["runtime"])
        return best_kernel
    
    def _process_llm_proposal_and_evaluate(self, llm_index, run_state_dict, top_5_kernel, cuda_individual):
        """Process a single LLM proposal AND evaluate it completely in one thread."""
        # Use evolution prompt with top 5 kernels as optimization history
        prompt = PromptMaker.make_evo_prompt(
            run_state_dict.task_info['gpu_type'],
            run_state_dict.task_info['cuda_version'],
            top_5_kernel,
            run_state_dict.task_info["func_runtime"],
            cuda_individual
        )
        return self._process_proposal_and_evaluate_common(llm_index, run_state_dict, prompt, "EVO")