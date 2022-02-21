from flwr.common.typing import FitRes, Parameters
from include import * 
from config import * 
from fed_ops import LocalUpdate
from test import get_eval_fn 

import flwr 
from flwr.common import FitIns, FitRes, ParametersRes, EvaluateIns, EvaluateRes, Weights
from flwr.common import ndarray_to_bytes

import timeit
import argparse

class TFedClient(flwr.client.Client):
    def __init__(self, client_id, dataset_train, dataset_test, net, eval_fn, args):
        self.client_id = client_id
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.net = net 
        self.eval_fn = eval_fn
        self.args = args 

        self.wp_lists = []
        self.local = LocalUpdate(client_id, dataset_train, dataset_test, self.wp_lists, args)

    def get_parameters(self) -> ParametersRes:

        weights: Weights = self.net.get_weights()
        parameters = flwr.common.weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

    def fit(self, ins: FitIns) -> FitRes:
        # set local model weights with that of the new global model
        weights: Weights = flwr.common.parameters_to_weights(ins.parameters)
        for i, w in enumerate(weights):
            try:
                _ = len(w)
            except:
                weights[i] = np.array([0])
        self.net.set_weights(weights)
        fit_begin = timeit.default_timer()

        # Train model 
        net_dict, wp_lists = self.local.TFed_train(net=self.net)
        self.wp_lists = wp_lists 

        # Return the refined weights and the number of examples used for training
        # weights_prime: Weights = net_dict.get_weights()
        weights_prime: Weights = [val.cpu().numpy() for _, val in net_dict.items()]
        for i, w in enumerate(weights_prime):
            try:
                _ = len(w)
            except:
                weights_prime[i] = np.array([0])
        params_prime = flwr.common.weights_to_parameters(weights_prime)
        num_examples_train = len(self.dataset_train.dataset)
        fit_duration = timeit.default_timer() - fit_begin
        return FitRes(
            parameters=params_prime,
            num_examples=num_examples_train,
            num_examples_ceil=num_examples_train,
            fit_duration=fit_duration,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Compute loss of the new global model on this client's data"""

        weights = flwr.common.parameters_to_weights(ins.parameters)
        for i, w in enumerate(weights):
            try:
                _ = len(w)
            except:
                weights[i] = np.array([0])
        # Use provided weights to update the local model
        self.net.set_weights(weights)

        # Evaluate the updated model on the local dataset
        g_loss, g_acc, g_acc5 = self.eval_fn(self.net, self.dataset_test, self.Args)

        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            loss=g_loss, num_examples=len(self.dataset_test.dataset), accuracy=g_acc
        )


if __name__ == '__main__':

    """Load data, create and start MNISTClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--cid", type=str, required=True, help="Client CID (no default)"
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    client_args = parser.parse_args()
    # Configure logger
    flwr.common.logger.configure(f"client_{client_args.cid}", host=client_args.log_host)

    # Model
    net = Fed_Model()
    net.to(Args.device)
    # G_loss_fun = torch.nn.CrossEntropyLoss()

    # Start client
    tfed_client = TFedClient(client_id=int(client_args.cid),
                            dataset_train=C_iter[int(client_args.cid)],
                            dataset_test=test_iter,
                            net=net, 
                            eval_fn=get_eval_fn(G_loss_fun),
                            args=Args
                        )
    flwr.client.start_client(client_args.server_address, tfed_client)