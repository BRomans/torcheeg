from typing import Dict, List, Union

import numpy as np

from ..base_transform import EEGTransform


class RearrangeElectrode(EEGTransform):
    r'''
    Select parts of electrode signals based on a given electrode index list.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.RearrangeElectrode(
            source=['FP1', 'F3', 'F7'],
            target=['F3', 'F7', 'FP1', 'AF2'],
            missing='mean'
        )
        t(eeg=np.random.randn(3, 128))['eeg'].shape
        >>> (4, 128)

    Args:
        source (list): The list of electrode names to be rearranged.
        target (list): The list of electrode names to be rearranged to.
        missing (str): The method to deal with missing electrodes. (default: :obj:`'random'`)

    .. automethod:: __call__
    '''
    def __init__(self,
                 source: List[str],
                 target: List[str],
                 missing: str = 'mean',
                 apply_to_baseline: bool = False):
        super(RearrangeElectrode,
              self).__init__(apply_to_baseline=apply_to_baseline)
        self.source = source
        self.target = target

        assert missing in [
            'random', 'zero', 'mean', 'approximate_mean'
        ], f'Invalid missing method {missing}, should be one of [random, zero, mean].'

        self.missing = missing

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.

        Returns:
            np.ndarray: The output signals with the shape of [number of picked electrodes, number of data points].
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].

        Returns:
            np.ndarray: The output signals with the shape of [number of rearranged electrodes, number of data points].
        '''
        output = np.zeros((len(self.target), eeg.shape[1]))
        for i, target in enumerate(self.target):
            if target in self.source:
                output[i] = eeg[self.source.index(target)]
            else:
                if self.missing == 'random':
                    output[i] = np.random.randn(eeg.shape[1])
                elif self.missing == 'zero':
                    output[i] = np.zeros(eeg.shape[1])
                elif self.missing == 'mean':
                    output[i] = np.mean(eeg, axis=0)
        return output

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'source': self.source,
                'target': self.target,
                'missing': self.missing
            })



class ImprovedRearrangeElectrode(RearrangeElectrode):
    def __init__(self,
                 source: List[str],
                 target: List[str],
                 missing: str = 'approximate_mean',
                 neighbor_map: Dict[str, List[str]] = None,
                 neighbor_weights: Dict[str, List[float]] = None,
                 apply_to_baseline: bool = False):
        super(ImprovedRearrangeElectrode, self).__init__(source, target, missing='mean', apply_to_baseline=apply_to_baseline)
        
        if missing not in ['random', 'zero', 'mean', 'approximate_mean']:
            raise ValueError(f"Invalid missing method {missing}, should be one of ['random', 'zero', 'mean', 'approximate_mean']")

        # Override missing method
        self.missing = missing
        
        # Dictionary of closest neighbors for missing electrodes by default (example for your channel sets)
        self.neighbor_map = neighbor_map or {
            'FZ': ['F3', 'F4'],
            'C3': ['FC5', 'F3'],
            'CZ': ['FC5', 'FC6'],
            'C4': ['FC6', 'F4'],
            'PZ': ['P7', 'P8'],
            'PO7': ['P7', 'O1'],
            'OZ': ['O1', 'O2'],
            'PO8': ['P8', 'O2']
        }
        # Optional weights for neighbors (must sum to 1), default uniform
        self.neighbor_weights = neighbor_weights or {key: [1/len(val)]*len(val) for key, val in self.neighbor_map.items()}

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        output = np.zeros((len(self.target), eeg.shape[1]))
        for i, target in enumerate(self.target):
            if target in self.source:
                output[i] = eeg[self.source.index(target)]
            else:
                if self.missing == 'random':
                    output[i] = np.random.randn(eeg.shape[1])
                elif self.missing == 'zero':
                    output[i] = np.zeros(eeg.shape[1])
                elif self.missing == 'mean':
                    output[i] = np.mean(eeg, axis=0)
                elif self.missing == 'approximate_mean':
                    neighbors = self.neighbor_map.get(target, None)
                    weights = self.neighbor_weights.get(target, None)
                    if neighbors is not None and weights is not None:
                        valid_neighbors = [n for n in neighbors if n in self.source]
                        valid_weights = [weights[j] for j, n in enumerate(neighbors) if n in self.source]
                        if valid_neighbors:
                            weighted_sum = np.zeros(eeg.shape[1])
                            total_weight = sum(valid_weights)
                            for n, w in zip(valid_neighbors, valid_weights):
                                weighted_sum += w / total_weight * eeg[self.source.index(n)]
                            output[i] = weighted_sum
                        else:
                            # fallback if no known neighbors
                            output[i] = np.mean(eeg, axis=0)
                    else:
                        output[i] = np.mean(eeg, axis=0)
        return output
    
    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'neighbor_map': self.neighbor_map,
                'neighbor_weights': self.neighbor_weights
            })