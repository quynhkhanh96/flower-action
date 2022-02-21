from include import * 
from config import * 
from fed_ops import ServerUpdate, choose_model 
from test import get_eval_fn 

from typing import Callable, Dict, List, Optional, Tuple

import flwr 
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

from flwr.common import FitIns, FitRes, Parameters, EvaluateIns, EvaluateRes, Weights, Scalar, parameter 
from flwr.common import parameters_to_weights, weights_to_parameters

import argparse
from datetime import datetime

from logging import WARNING

DEPRECATION_WARNING = """
DEPRECATION WARNING: deprecated `eval_fn` return format

    loss, accuracy

move to

    loss, {"accuracy": accuracy}

instead. Note that compatibility with the deprecated return format will be
removed in a future release.
"""

DEPRECATION_WARNING_INITIAL_PARAMETERS = """
DEPRECATION WARNING: deprecated initial parameter type

    flwr.common.Weights (i.e., List[np.ndarray])

will be removed in a future update, move to

    flwr.common.Parameters

instead. Use

    parameters = flwr.common.weights_to_parameters(weights)

to easily transform `Weights` to `Parameters`.
"""
import os 
round_weights_dir = 'round_weights'
os.makedirs(round_weights_dir, exist_ok=True)

class TernaryFedAvg(Strategy):
    def __init__(self,
            fraction_fit: float = 0.1, 
            min_fit_clients: int = 2,
            min_available_clients: int = 2,    
            accept_failures: bool = True,
            eval_fn: Optional[
                Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
            ] = None,  
            initial_parameters: Optional[Parameters] = None,
        ):
        super().__init__()
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.accept_failures = accept_failures
        self.eval_fn = eval_fn
        self.initial_parameters = initial_parameters

    def __repr__(self) -> str:
        rep = f"T-FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients):
        """Return the sample size and the required number of available
        clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_eval)
        return max(num_clients, self.min_eval_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        if isinstance(initial_parameters, list):
            log(WARNING, DEPRECATION_WARNING_INITIAL_PARAMETERS)
            initial_parameters = weights_to_parameters(weights=initial_parameters)
        return initial_parameters

    def evaluate(
        self, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
            Evaluate current model parameters (i.e. the new global model obtained 
            from aggregation) using an evaluation function.
        """
        if self.eval_fn is None:
            # No evaluation function provided
            return None
        weights = parameters_to_weights(parameters)
        for i, w in enumerate(weights):
            try:
                _ = len(w)
            except:
                weights[i] = np.array([0])
        model = Fed_Model()
        model.set_weights(weights)
        eval_res = self.eval_fn(model=model, val_iterator=test_iter, args=Args)
        if eval_res is None:
            return None
        loss, acc, acc_top5 = eval_res
        metrics = {'accuracy': acc, 'accuracy_top5': acc_top5} 
        print(loss, metrics)
        return loss, metrics

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Convert results
        # weights_results = [
        #     (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
        #     for client, fit_res in results
        # ]
        # return weights_to_parameters(aggregate(weights_results)), {}
        w_locals = [parameters_to_weights(fit_res.parameters) for client, fit_res in results]
        local_models = []
        for weights in w_locals:
            model = Fed_Model()
            model.set_weights(weights)
            local_models.append(model.state_dict())
        num_samples = [fit_res.num_examples for client, fit_res in results]
        w_glob, ter_glob = ServerUpdate(local_models, num_samples)
        new_w_glob, tmp_flag = choose_model(w_glob, ter_glob)
        new_weights = [val.cpu().numpy() for _, val in new_w_glob.items()]
        for i, w in enumerate(new_weights):
            try:
                _ = len(w)
            except:
                new_weights[i] = np.array([0])
        np.savez(f'{round_weights_dir}/round_{rnd}-agg_params.npz', new_weights)

        return weights_to_parameters(new_weights), {}

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        loss_aggregated = weighted_loss_avg(
            [
                (
                    evaluate_res.num_examples,
                    evaluate_res.loss,
                    evaluate_res.accuracy,
                )
                for _, evaluate_res in results
            ]
        )
        
        return loss_aggregated, {}

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        # if self.on_fit_config_fn is not None:
        #     # Custom fit config function provided
        #     config = self.on_fit_config_fn(rnd)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""

        return []

if __name__ == '__main__':

    """Start server and train for some rounds."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    server_args = parser.parse_args()

    start_time = datetime.now().strftime("%H:%M:%S")

    # Create strategy 
    tfedavg_strategy = TernaryFedAvg(
        fraction_fit=Args.frac,
        min_fit_clients=Args.min_sample_size,
        min_available_clients=Args.min_num_clients,
        eval_fn=get_eval_fn(G_loss_fun)
    )

    # Configure logger and start server
    flwr.common.logger.configure("server", host=server_args.log_host)
    flwr.server.start_server(
        server_args.server_address,
        config={"num_rounds": Args.epochs}, 
        strategy=tfedavg_strategy,
    )

    end_time = datetime.now().strftime("%H:%M:%S")
    print(f'Start at {start_time}, done at {end_time}.')
