import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    r'''
    A compact convolutional neural network (EEGNet). For more details, please refer to the following information.

    - Paper: Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    - URL: https://arxiv.org/abs/1611.08024
    - Related Project: https://github.com/braindecode/braindecode/tree/master/braindecode

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torcheeg.models import EEGNet
        from torch.utils.data import DataLoader

        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor(),
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]))

        model = EEGNet(chunk_size=128,
                       num_electrodes=32,
                       dropout=0.5,
                       kernel_1=64,
                       kernel_2=16,
                       F1=8,
                       F2=16,
                       D=2,
                       num_classes=2)

        x, y = next(iter(DataLoader(dataset, batch_size=64)))
        model(x)

    Args:
        chunk_size (int): Number of data points included in each EEG chunk, i.e., :math:`T` in the paper. (default: :obj:`151`)
        num_electrodes (int): The number of electrodes, i.e., :math:`C` in the paper. (default: :obj:`60`)
        F1 (int): The filter number of block 1, i.e., :math:`F_1` in the paper. (default: :obj:`8`)
        F2 (int): The filter number of block 2, i.e., :math:`F_2` in the paper. (default: :obj:`16`)
        D (int): The depth multiplier (number of spatial filters), i.e., :math:`D` in the paper. (default: :obj:`2`)
        num_classes (int): The number of classes to predict, i.e., :math:`N` in the paper. (default: :obj:`2`)
        kernel_1 (int): The filter size of block 1. (default: :obj:`64`)
        kernel_2 (int): The filter size of block 2. (default: :obj:`64`)
        dropout (float): Probability of an element to be zeroed in the dropout layers. (default: :obj:`0.25`)
    '''
    def __init__(self,
                 chunk_size: int = 151,
                 num_electrodes: int = 60,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 num_classes: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.num_electrodes = num_electrodes
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout

        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1,
                                 self.F1 * self.D, (self.num_electrodes, 1),
                                 max_norm=1,
                                 stride=1,
                                 padding=(0, 0),
                                 groups=self.F1,
                                 bias=False), nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      padding=(0, self.kernel_2 // 2),
                      bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout))

        self.lin = nn.Linear(self.feature_dim(), num_classes, bias=False)

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)

            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)

        return self.F2 * mock_eeg.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 60, 151]`. Here, :obj:`n` corresponds to the batch size, :obj:`60` corresponds to :obj:`num_electrodes`, and :obj:`151` corresponds to :obj:`chunk_size`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        x = self.lin(x)

        return x
    
    
class BandEEGNet(nn.Module):
    """
    EEGNet variant that processes frequency-domain input features
    (e.g., PSD or band power per electrode x frequency band).

    Input shape: [n, 1, num_electrodes, num_bands]
    Example: (batch, 1, 32 electrodes, 5 frequency bands)
    """

    def __init__(self, num_electrodes=32, num_bands=5, F1=8, D=2, F2=16,
                 dropout=0.5, num_classes=2, kernel_1=3, kernel_2=3):
        super().__init__()
        self.num_electrodes = num_electrodes
        self.num_bands = num_bands
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropout = dropout
        self.num_classes = num_classes

        # Spectral processing along frequency dimension
        self.block_spectral = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_1), padding=(0, kernel_1 // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(p=dropout)
        )

        # Spatial processing across electrodes
        self.block_spatial = nn.Sequential(
            Conv2dWithConstraint(F1, F1 * D, (num_electrodes, 1), max_norm=1, groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.Dropout(p=dropout)
        )

        # Feature mixing across frequency bands
        self.block_feature = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, kernel_2), padding=(0, kernel_2 // 2), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(p=dropout)
        )

        self.classifier = nn.Linear(self._feature_dim(), num_classes)

    def _feature_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.num_electrodes, self.num_bands)
            x = self.block_spectral(x)
            x = self.block_spatial(x)
            x = self.block_feature(x)
            return x.flatten(start_dim=1).shape[1]

    def forward(self, x):
        x = self.block_spectral(x)
        x = self.block_spatial(x)
        x = self.block_feature(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x
    



class PACNet(nn.Module):
    """
    PACNet: Multi-branch EEGNet for power/frequency bands.
    Each branch learns spectral-spatial features from a single band.
    - Paper: Jiajun Li, Yu Qi, Yu Qi1, Gang Pan, Gang Pan. Phase-amplitude coupling-based adaptive filters for neural signal decoding.
    - URL: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1153568/full    
    """

    def __init__(self, num_electrodes=32, num_classes=2, bands=('delta', 'theta', 'alpha', 'beta'), 
                 F1=8, D=2, F2=16, dropout=0.5, kernel_1=64, kernel_2=16):
        super().__init__()
        self.bands = bands
        self.num_bands = len(bands)
        self.branches = nn.ModuleList()

        for _ in bands:
            branch = nn.Sequential(
                nn.Conv2d(1, F1, (1, kernel_1), padding=(0, kernel_1 // 2), bias=False),
                nn.BatchNorm2d(F1),
                Conv2dWithConstraint(F1, F1 * D, (num_electrodes, 1), max_norm=1, groups=F1, bias=False),
                nn.BatchNorm2d(F1 * D),
                nn.ELU(),
                nn.AvgPool2d((1, 4)),
                nn.Dropout(p=dropout),
                nn.Conv2d(F1 * D, F2, (1, kernel_2), padding=(0, kernel_2 // 2), bias=False),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AvgPool2d((1, 8)),
                nn.Dropout(p=dropout)
            )
            self.branches.append(branch)

        # After concatenating features across bands
        self.fc = nn.Linear(self._feature_dim(num_electrodes), num_classes)

    def _feature_dim(self, num_electrodes):
        with torch.no_grad():
            x = torch.zeros(1, 1, num_electrodes, 128)
            feats = []
            for branch in self.branches:
                feats.append(branch(x))
            x = torch.cat(feats, dim=1)
            return x.flatten(start_dim=1).shape[1]

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [N, B, C, T], where
                N: batch size,
                B: number of frequency bands,
                C: number of electrodes,
                T: number of time points.
        Returns:
            torch.Tensor: Output tensor of shape [N, num_classes].
        """
        # x shape: [N, B, C, T]
        outs = []
        for i in range(self.num_bands):
            out_i = self.branches[i](x[:, i:i+1, :, :])
            outs.append(out_i)
        x = torch.cat(outs, dim=1)
        x = x.flatten(start_dim=1)
        return self.fc(x)
