from ..base_run_state_dict import BaseRunStateDict
from evotool.task.base_task import Solution

class EohRunStateDict(BaseRunStateDict):
    def __init__(
            self,
            task_info: dict,
            generation: int = 0,
            tot_sample_nums: int = 0,
            sol_history: list[Solution] = None,
            population: list[Solution] = None,
            is_done: bool = False
    ):
        super().__init__(task_info)
        
        self.generation = generation
        self.tot_sample_nums = tot_sample_nums
        self.is_done = is_done
        self.sol_history = sol_history or []  # Complete history of all solutions
        self.population = population or []     # Current generation population
        self.usage_history = {}
        
    def to_json(self) -> dict:
        """Convert the run state to JSON-serializable dictionary"""
        # Convert Solution objects to dictionaries
        sol_history_json = []
        for sol in self.sol_history:
            sol_dict = {
                'sol_string': sol.sol_string,
                'other_info': sol.other_info,
                'evaluation_res': None
            }
            if sol.evaluation_res:
                sol_dict['evaluation_res'] = {
                    'valid': sol.evaluation_res.valid,
                    'score': sol.evaluation_res.score,
                    'additional_info': sol.evaluation_res.additional_info
                }
            sol_history_json.append(sol_dict)
        
        population_json = []
        for sol in self.population:
            sol_dict = {
                'sol_string': sol.sol_string,
                'other_info': sol.other_info,
                'evaluation_res': None
            }
            if sol.evaluation_res:
                sol_dict['evaluation_res'] = {
                    'valid': sol.evaluation_res.valid,
                    'score': sol.evaluation_res.score,
                    'additional_info': sol.evaluation_res.additional_info
                }
            population_json.append(sol_dict)
            
        return {
            'task_info': self.task_info,
            'generation': self.generation,
            'tot_sample_nums': self.tot_sample_nums,
            'sol_history': sol_history_json,
            'population': population_json,
            'is_done': self.is_done,
            'usage_history': self.usage_history
        }
        
    @classmethod
    def from_json(cls, data: dict) -> 'EohRunStateDict':
        """Create instance from JSON data"""
        from evotool.task.base_task import Solution, EvaluationResult
        
        # Convert sol_history from dictionaries back to Solution objects
        sol_history = []
        for sol_dict in data.get('sol_history', []):
            evaluation_res = None
            if sol_dict.get('evaluation_res'):
                eval_data = sol_dict['evaluation_res']
                evaluation_res = EvaluationResult(
                    valid=eval_data['valid'],
                    score=eval_data['score'],
                    additional_info=eval_data['additional_info']
                )
            
            solution = Solution(
                sol_string=sol_dict['sol_string'],
                other_info=sol_dict.get('other_info'),
                evaluation_res=evaluation_res
            )
            sol_history.append(solution)
        
        # Convert population from dictionaries back to Solution objects
        population = []
        for sol_dict in data.get('population', []):
            evaluation_res = None
            if sol_dict.get('evaluation_res'):
                eval_data = sol_dict['evaluation_res']
                evaluation_res = EvaluationResult(
                    valid=eval_data['valid'],
                    score=eval_data['score'],
                    additional_info=eval_data['additional_info']
                )
            
            solution = Solution(
                sol_string=sol_dict['sol_string'],
                other_info=sol_dict.get('other_info'),
                evaluation_res=evaluation_res
            )
            population.append(solution)
        
        instance = cls(
            task_info=data['task_info'],
            generation=data.get('generation', 0),
            tot_sample_nums=data.get('tot_sample_nums', 0),
            sol_history=sol_history,
            population=population,
            is_done=data.get('is_done', False),
        )
        instance.usage_history = data.get('usage_history', {})
        return instance