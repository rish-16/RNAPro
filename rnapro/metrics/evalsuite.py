import torch

class EvalSuite:
    def __init__(self, metrics):
        """
        Initialize the EvalSuite with a list of metric functions.

        Args:
            metrics (list): A list of metric functions to be evaluated.
        """
        self.metrics = metrics 
        self.func_mapper = {
            'sctm': self.compute_sctm,
            'pdbtm': self.compute_pdbtm,
            'scrmsd': self.compute_scrmsd,
        }

    def evaluate(self, predictions, targets):
        """
        Evaluate the predictions against the targets using the provided metrics.

        Args:
            predictions (torch.Tensor): The predicted values.
            targets (torch.Tensor): The ground truth values.

        Returns:
            dict: A dictionary with metric names as keys and their computed values.
        """
        results = {}
        for metric in self.metrics:
            metric_func = self.func_mapper[metric]
            results[metric] = metric_func(predictions, targets)
        return results
    
    def compute_sctm(self, predictions, targets):
        pass