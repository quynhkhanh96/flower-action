import torch
import numpy as np
from flwr.common import ndarray_to_bytes
from qsgd_video_client import QSGDVideoClient

class TopkQSGDVideoClient(QSGDVideoClient):

    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k
    
    def compress_weight_update_up(self):
        params_prime = []
        s = self.quantizer.s
        for lname, lgrad in self.dW.items():
            if self._keep_layer_full_precision(lname):
                params_prime.append(
                    ndarray_to_bytes(lgrad.cpu().numpy())
                )
                continue

            if torch.count_nonzero(lgrad) == 0: 
                params_prime.append(bytes(1 & 0xff))
                continue

            if 'conv' in lname and 'bn' not in lname:
                self.quantizer.s = s
                n_elts = lgrad.numel()
                n_top = int(np.ceil(n_elts * self.k))
                with torch.no_grad():
                    inds = torch.argsort(torch.abs(lgrad).flatten(), descending=True)
                    
                    # compress and encode the top-k gradients with higher #bits
                    mask_top = torch.full((lgrad.numel(),), 0.).to(lgrad.device).scatter_(0,
                                        inds[:n_top], 1).view(*tuple(lgrad.shape))
                    signature_top = self.quantizer.quantize(lgrad * mask_top)

                    # and the rest with less #bits
                    self.quantizer.s = 2 ** self.lower_bit
                    mask_rest = torch.full((lgrad.numel(),), 0.).to(lgrad.device).scatter_(0,
                                        inds[n_top:], 1).view(*tuple(lgrad.shape))
                    signature_rest = self.quantizer.quantize(lgrad * mask_rest)

                params_prime.extend(
                    [self._encode_signature(signature_top),
                    self._encode_signature(signature_rest)]
                )

            else:
                self.quantizer.s = 2 ** self.lower_bit
                signature = self.quantizer.quantize(lgrad)
                params_prime.append(
                    self._encode_signature(signature)
                )

        self.quantizer.s = s
        
        return params_prime

