import numpy as np
import pandas as pd
import jax.numpy as jnp
from torch.utils.data import DataLoader, Dataset
from pathlib import Path 
from os.path import isfile
import os
import pdb

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
new_HRRR_vars = ['DPT_1000mb', 'DPT_2m_above_ground',
                 'DPT_850mb',
                 'DPT_925mb',
                 'HPBL_surface',
                 'POT_2m_above_ground',
                 'PRES_surface',
                 'RH_2m_above_ground',
                 'RHPW_entire_atmosphere',
                 'TMP_1000mb',
                 'TMP_2m_above_ground',
                 'TMP_500mb',
                 'TMP_700mb',
                 'TMP_850mb',
                 'TMP_925mb',
                 'TMP_surface',
                #  'VIS_surface',
                 'VUCSH_0_1000m_above_ground',
                 'VVCSH_0_1000m_above_ground']
AOD_vars = ['AOD_047', 'AOD_055']

class Merged_Data_Dataset(Dataset):
    def __init__(self, file_dir, transform=None, target_transform=None):
        file_dir = Path(file_dir)
        self.data = self.load_file(file_dir)
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx][new_HRRR_vars + AOD_vars].values
        label = self.data.iloc[idx]['PA_calibrated']
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return sample.astype(float), label.astype(float)
    
    def load_file(self, file_dir):
        file_list = sorted(file_dir.glob('*.csv'))

        data = pd.DataFrame()
        for file_path in file_list:
            df = pd.read_csv(file_path, index_col = 0)
            df.index = pd.to_datetime(df.index)
            df = df.reset_index()
            df.rename(columns = {'idnex':'datetime'}, inplace = True)
            
            data = pd.concat([data, df])

        data.reset_index(drop= True,inplace = True)
        data = data[~data['AOD_047'].isna()]
        return data

       

class NumpyLoader(DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))
  