
import torch
from typing import Tuple, Dict
from pymlrf.types import CriterionProtocol
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)
from pymlrf.types import (
    GenericDataLoaderProtocol
    )
from sklearn.metrics import f1_score

class ValidateSingleEpoch:
    
    def __init__(
        self, 
        half_precision:bool=False,
        cache_preds:bool=True
        ) -> None:
        """Class which runs a single epoch of validation.

        Args:
            half_precision (bool, optional): Boolean defining whether to run in 
            half-precision. Defaults to False.
            cache_preds (bool, optional): Boolean defining whether to save the 
            prediction outputs. Defaults to True.
        """
        self.half_precision = half_precision
        self.cache_preds = cache_preds

    def __call__(
        self,
        model:torch.nn.Module,
        data_loader:GenericDataLoaderProtocol,
        gpu:bool,
        criterion:CriterionProtocol
        )->Tuple[torch.Tensor, Dict[str,torch.Tensor]]:
        """ Call function which runs a single epoch of validation
        Args:
            model (BaseModel): Torch model of type BaseModel i.e., it should
            subclass the BaseModel class
            data_loader (DataLoader): Torch data loader object
            gpu (bool): Boolean defining whether to use a GPU if available
            criterion (CriterionProtocol): Criterian to use for training or 
            for training and validation if val_criterion is not specified. I.e., 
            this could be nn.MSELoss() 
            
        Returns:
            Tuple[torch.Tensor, Dict[str,torch.Tensor]]: Tuple defining the 
            final loss for the epoch and a dictionary of predictions. The keys 
            will be the same keys required by the criterion. 
        """

        losses = torch.tensor(0.0)
        denom = torch.tensor(0)
        mae = torch.tensor(0.0)
        correct = torch.tensor(0.0)
        total = torch.tensor(0)
        if gpu:
            _device = "cuda"
        else:
            _device = "cpu"
            
        if self.half_precision:
            losses = losses.half()
            denom = denom.half()
        model.eval()
        preds = []
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for i, vals in enumerate(data_loader):

                # Prepare input data
                input_vals = vals.input
                output_vals = vals.output
                if gpu:
                    input_vals = {
                        val:input_vals[val].cuda() for val in input_vals
                        }
                    output_vals = {
                        val:output_vals[val].cuda() for val in output_vals
                        }
                else:
                    input_vals = {
                        val:Variable(input_vals[val]) for val in input_vals
                        }
                    output_vals = {
                        val:Variable(output_vals[val]) for val in output_vals
                        }

                # Compute output
                if self.half_precision:
                    with torch.autocast(device_type=_device):
                        output = model(**input_vals)
                else:
                    output = model(**input_vals)

                # Logs
                val_loss = criterion(output, output_vals)
                losses += val_loss.detach().cpu()
                denom += 1

                mae += torch.sum(torch.abs(output["pos"] - output_vals["pos"]).cpu())
                _, predicted = torch.max(output["grp"], 1)
                correct += (predicted == torch.argmax(output_vals["grp"], dim=1)).sum().cpu()
                total += output_vals["grp"].size(0)
                all_preds.append(predicted.cpu())
                all_labels.append(torch.argmax(output_vals["grp"], dim=1).cpu())
                
                if self.cache_preds:
                    preds.append({k:output[k].detach().cpu() for k in output.keys()})
        _prd_lst = {}
        if self.cache_preds:
            for k in preds[0].keys():
                _prd_lst[k] = torch.concat([t[k] for t in preds],dim=0)
        losses = losses/denom
        mae = mae / total
        accuracy = correct / total
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='macro')
        return {"loss": losses, "mae": mae, "accuracy": accuracy, "f1": torch.tensor(f1)}, _prd_lst
