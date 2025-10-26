import os
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
from scipy.io import loadmat

from ....utils import get_random_dir_path
from ..base_dataset import BaseDataset


class STEWDataset(BaseDataset):
    r'''
    The STEW (Simultaneous Task EEG Workload) dataset for mental workload classification.
    This class generates training samples and test samples according to the given parameters,
    and caches the generated results in a unified input and output format (IO).

    - Author: Wei Lun Lim, Olga Sourina, Lipo Wang
    - Year: 2018
    - Download URL: https://www.kaggle.com/datasets/mitulahirwal/mental-cognitive-workload-eeg-data-stew-dataset
    - Reference: Wei Lun Lim, Olga Sourina, Lipo Wang, "STEW: Simultaneous Task EEG Workload Dataset", IEEE Dataport, 2018.
    - Stimulus: SIMKAP multitasking test (arithmetic calculations)
    - Signals: Electroencephalogram (14 channels at 128Hz, 2.5 minutes recording)
    - Rating: Mental workload rating (scale from 1 to 9)
    - Subjects: 45 subjects (originally 48, but 3 subjects lack ratings)

    The dataset folder should contain:
        dataset.mat          - EEG data (14 channels x 19200 timepoints x 45 subjects)
        rating.mat           - Workload ratings (45 x 1)
        class_012.mat        - Three-class labels (45 x 1)
        three_class_one_hot.mat - One-hot encoded three-class labels (45 x 3)

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import STEWDataset
        from torcheeg import transforms

        dataset = STEWDataset(
            root_path='./stew/mental-cognitive-workload-eeg-data-stew-dataset',
            chunk_size=128,
            overlap=0,
            num_classes=3,
            online_transform=transforms.ToTensor(),
            label_transform=transforms.Compose([
                transforms.Select('label')
            ])
        )
        print(dataset[0])
        # EEG signal (torch.Tensor[1, 14, 128]),
        # label (int)

    Args:
        root_path (str): Path to the folder containing the STEW .mat files 
            (default: :obj:`'./stew/mental-cognitive-workload-eeg-data-stew-dataset'`)
        chunk_size (int): Number of data points included in each EEG chunk. 
            If set to -1, the entire trial is used as one chunk. (default: :obj:`128`)
        overlap (int): Number of overlapping data points between different chunks. (default: :obj:`0`)
        num_channel (int): Number of EEG channels to use (max 14). (default: :obj:`14`)
        num_classes (int): Number of classes for classification. Options: 2 or 3.
            - 2 classes: Normal (4-6) vs High (7-9) workload
            - 3 classes: Normal (4-5), Moderate (6-7), High (8-9) workload
            (default: :obj:`3`)
        online_transform (Callable, optional): Transformation applied to EEG signals during loading.
        offline_transform (Callable, optional): Transformation applied before caching to disk.
        label_transform (Callable, optional): Transformation applied to labels.
        before_trial (Callable, optional): Hook applied to trial before offline transform.
        after_trial (Callable, optional): Hook applied after offline transform.
        after_session (Callable, optional): Hook applied after processing all trials in a session.
        after_subject (Callable, optional): Hook applied after processing all data for a subject.
        io_path (str): Path for caching intermediate results. (default: :obj:`None`)
        io_size (int): Maximum database size for memory mapping. (default: :obj:`1048576`)
        io_mode (str): Storage mode: 'lmdb', 'pickle', or 'memory'. (default: :obj:`'lmdb'`)
        num_worker (int): Number of worker processes for parallel loading. (default: :obj:`0`)
        verbose (bool): Whether to display progress information. (default: :obj:`True`)
    '''

    def __init__(self,
                 root_path: str = './stew/mental-cognitive-workload-eeg-data-stew-dataset',
                 chunk_size: int = 128,
                 overlap: int = 0,
                 num_channel: int = 14,
                 num_classes: int = 3,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 after_session: Union[Callable, None] = None,
                 after_subject: Union[Callable, None] = None,
                 io_path: Union[None, str] = None,
                 io_size: int = 1048576,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True):
        
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')

        assert num_classes in [2, 3], "num_classes must be 2 or 3"
        assert num_channel <= 14, "num_channel must be <= 14"

        # Store all parameters
        params = {
            'root_path': root_path,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'num_channel': num_channel,
            'num_classes': num_classes,
            'online_transform': online_transform,
            'offline_transform': offline_transform,
            'label_transform': label_transform,
            'before_trial': before_trial,
            'after_trial': after_trial,
            'after_session': after_session,
            'after_subject': after_subject,
            'io_path': io_path,
            'io_size': io_size,
            'io_mode': io_mode,
            'num_worker': num_worker,
            'verbose': verbose
        }
        super().__init__(**params)
        self.__dict__.update(params)

    def set_records(self, root_path: str = './stew/mental-cognitive-workload-eeg-data-stew-dataset', **kwargs):
        '''
        Set up the records to process. Each record represents one subject.
        
        Returns:
            list: List of subject IDs (0 to 44)
        '''
        assert os.path.exists(root_path), \
            f'root_path ({root_path}) does not exist. Please download the dataset and set the correct path.'
        
        # Check that required files exist
        required_files = ['dataset.mat', 'rating.mat']
        for file in required_files:
            file_path = os.path.join(root_path, file)
            assert os.path.exists(file_path), \
                f'Required file {file} not found in {root_path}'
        
        # Return list of subject indices (0-44, representing 45 subjects)
        return list(range(45))

    @staticmethod
    def read_record(record: int, 
                   root_path: str = './stew/mental-cognitive-workload-eeg-data-stew-dataset',
                   num_classes: int = 3,
                   **kwargs) -> Dict:
        '''
        Read the EEG data and labels for a specific subject.
        
        Args:
            record (int): Subject index (0-44)
            root_path (str): Path to dataset folder
            num_classes (int): Number of classes (2 or 3)
            
        Returns:
            dict: Dictionary containing 'sample' (EEG data) and 'label' (workload class)
        '''
        # Load the .mat files
        dataset_path = os.path.join(root_path, 'dataset.mat')
        rating_path = os.path.join(root_path, 'rating.mat')
        
        # Load data: shape is (14, 19200, 45)
        mat_data = loadmat(dataset_path)
        eeg_data = mat_data['dataset']  # (14 channels, 19200 timepoints, 45 subjects)
        
        # Load ratings: shape is (45, 1)
        mat_ratings = loadmat(rating_path)
        ratings = mat_ratings['rating'].flatten()  # (45,)
        
        # Extract data for this specific subject
        subject_eeg = eeg_data[:, :, record]  # (14, 19200)
        subject_rating = ratings[record]  # scalar value (1-9)
        
        # Convert rating to class label based on num_classes
        if num_classes == 3:
            # Three class: Normal (4-5), Moderate (6-7), High (8-9)
            if subject_rating <= 5:
                label = 0  # Normal
            elif subject_rating <= 7:
                label = 1  # Moderate
            else:
                label = 2  # High
        else:  # num_classes == 2
            # Two class: Normal (4-6), High (7-9)
            if subject_rating <= 6:
                label = 0  # Normal
            else:
                label = 1  # High
        
        return {
            'sample': subject_eeg,
            'rating': subject_rating,
            'label': label
        }

    @staticmethod
    def process_record(record: int,
                      sample: np.ndarray,
                      rating: float,
                      label: int,
                      chunk_size: int = 128,
                      overlap: int = 0,
                      num_channel: int = 14,
                      before_trial: Union[None, Callable] = None,
                      offline_transform: Union[None, Callable] = None,
                      **kwargs):
        '''
        Process the EEG data for one subject, splitting it into chunks.
        
        Args:
            record (int): Subject ID
            sample (np.ndarray): EEG data (num_channel, timepoints)
            rating (float): Mental workload rating (1-9)
            label (int): Class label
            chunk_size (int): Size of each chunk
            overlap (int): Overlap between chunks
            num_channel (int): Number of channels to use
            before_trial (Callable): Preprocessing hook
            offline_transform (Callable): Offline transformation
        '''
        subject_id = f's{record:02d}'
        
        # Use only the specified number of channels
        trial_sample = sample[:num_channel, :]  # (num_channel, 19200)
        
        if before_trial:
            trial_sample = before_trial(trial_sample)
        
        # Determine chunk size
        if chunk_size <= 0:
            dynamic_chunk_size = trial_sample.shape[1]
        else:
            dynamic_chunk_size = chunk_size
        
        # Calculate step size
        step = dynamic_chunk_size - overlap
        
        # Chunk the signal
        start_at = 0
        end_at = dynamic_chunk_size
        write_pointer = 0
        
        while end_at <= trial_sample.shape[1]:
            clip_sample = trial_sample[:, start_at:end_at]
            
            t_eeg = clip_sample
            
            if offline_transform is not None:
                t = offline_transform(eeg=clip_sample)
                t_eeg = t['eeg']
            
            clip_id = f'{subject_id}_{write_pointer}'
            write_pointer += 1
            
            # Create metadata for this chunk
            record_info = {
                'subject_id': subject_id,
                'start_at': start_at,
                'end_at': end_at,
                'clip_id': clip_id,
                'rating': rating,
                'label': label
            }
            
            yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}
            
            start_at += step
            end_at = start_at + dynamic_chunk_size

    def __getitem__(self, index: int) -> Tuple:
        '''
        Get a sample from the dataset.
        
        Args:
            index (int): Index of the sample
            
        Returns:
            tuple: (signal, label) where signal is the EEG data and label is the class
        '''
        info = self.read_info(index)
        eeg_index = str(info['clip_id'])
        eeg_record = str(info['_record_id'])
        eeg = self.read_eeg(eeg_record, eeg_index)

        signal = eeg
        label = info

        if self.online_transform:
            signal = self.online_transform(eeg=eeg)['eeg']

        if self.label_transform:
            label = self.label_transform(y=info)['y']

        return signal, label

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'root_path': self.root_path,
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'num_channel': self.num_channel,
                'num_classes': self.num_classes,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size
            })