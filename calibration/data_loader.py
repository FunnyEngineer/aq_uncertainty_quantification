import numpy as np
import pandas as pd
import jax.numpy as jnp
from torch.utils.data import DataLoader, Dataset
from pathlib import Path 
from os.path import isfile

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

class LCSFEM_Bias_Dataset(Dataset):
    def __init__(self, pair_file, pa_dir, an_dir, transform=None, target_transform=None, ln_scale = False):
        an_dir = Path(an_dir)
        pa_dir = Path(pa_dir)
        self.data = self.load_file(pair_file, pa_dir, an_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.ln_scale = ln_scale
        
        self.an_dir = Path(an_dir)
        self.pa_dir = Path(pa_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx][-4:-1].values
        label = self.data.iloc[idx][-1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.ln_scale:
            label = jnp.log(label)
        return sample.astype(float), label.astype(float)
    
    def load_file(self, pair_file, pa_dir, an_dir, pair_dis = 1):
        train_data = pd.DataFrame()
        min_df = pd.read_csv(pair_file, index_col = 0)
        subtracted = min_df[min_df['1']< pair_dis]
        for i, row in subtracted.iterrows():
            pa_name = i
            if (isfile(pa_dir/ pa_name)):
                pa_data = pd.read_csv(pa_dir/ pa_name, index_col=0)
                pa_data['datetime'] = pd.to_datetime(pa_data['datetime'])
                pa_data.index = pa_data['datetime']
                pa_data.drop(['label'], axis=1, inplace=True) # remove label column
                pa_data = pa_data.resample('H').mean().dropna()
                pa_data['PM25'] = pa_data.loc[:, ['pm25_A', 'pm25_B']].mean(axis=1)
                pa_data = pa_data[pa_data['PM25'] <= 200]

                an_data = pd.read_csv(an_dir / row["0"], index_col = 0)
                an_data.index = pd.to_datetime(an_data['datetime'])
                an_data = an_data[an_data['concentration'] > 0]
                
                sub_data = pa_data[pa_data.index.isin(an_data.index)].copy()
                sub_data['an'] = an_data[an_data.index.isin(pa_data.index)]['concentration'].values
                
                train_data = pd.concat([train_data, sub_data])

        return train_data
       

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
  