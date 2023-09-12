import torch
from functools import reduce
from flwr.common import bytes_to_ndarray
from qsgd_video_server import QSGDVideoServer
from ..utils import qsgd

class TopkQSGDVideoServer(QSGDVideoServer):
    def _decode_fit_results(self, results):
        grads_results = []
        s = self.coder.s
        lnames = [lname for lname, _ in self.dW.items()]
        n_layers = len(lnames)

        for _, fit_res in results:
            grads = {}
            grad_prime = fit_res.parameters.tensors
            i = 0 
            while i < n_layers:
                lname = lnames[i]
                lgrad = self.dW[lname]
                if self._keep_layer_full_precision(lname):
                    dec_lgrad = bytes_to_ndarray(grad_prime[i])
                    try:
                        _ = len(dec_lgrad)
                        dec_lgrad = torch.Tensor(dec_lgrad)
                    except: # for edge case: torch.tensor(0)
                        dec_lgrad = torch.tensor(dec_lgrad)
                    grads[lname] = dec_lgrad
                    i += 1
                elif 'conv' in lname and 'bn' not in lname:
                    s_top, s_rest = grad_prime[i], grad_prime[i+1]

                    self.coder.s = s 
                    grad_top = self.coder.decode(
                        s_top, reduce(lambda x, y: x*y, lgrad.shape)
                    )
                    grad_top = torch.Tensor(grad_top).view(lgrad.shape).to(self.device)
                    
                    self.coder.s = 2 ** self.lower_bit
                    grad_rest = self.coder.decode(
                        s_rest, reduce(lambda x, y: x*y, lgrad.shape)
                    )
                    grad_rest = torch.Tensor(grad_rest).view(lgrad.shape).to(self.device)
                    with torch.no_grad():
                        grads[lname] = grad_top + grad_rest
                    i += 2
                else:
                    self.coder.s = 2 ** self.lower_bit
                    dec_lgrad = self.coder.decode(grad_prime[i],
                                    reduce(lambda x, y: x*y, lgrad.shape))
                    grads[lname] = torch.Tensor(dec_lgrad).view(lgrad.shape).to(self.device)

        grads_results.append((grads, fit_res.num_examples))
        self.coder.s = s

        return grads_results