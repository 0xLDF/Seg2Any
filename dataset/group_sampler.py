from torch.utils.data import Sampler
import numpy as np

class GroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)  
        # The flag is determined during dataset initialization.

        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0 
        
        nonzero_indices = np.nonzero(self.group_sizes)[0]
        nonzero_values = self.group_sizes[nonzero_indices]
        self.group_sizes = list(zip(nonzero_indices, nonzero_values))
        
        num_groups = len(nonzero_values)
        
        for i, size in self.group_sizes:
            self.num_samples += int(np.ceil(size / self.samples_per_gpu)) * self.samples_per_gpu
        # group_size may not be divisible by samples_per_gpu, so we ceil it
        # Example: group0 has 100 samples, group1 has 200, samples_per_gpu=29
        # Then num_samples = ceil(100/29)*29 + ceil(200/29)*29 = 116 + 203 = 319
        
        print("GroupSampler.num_samples:  ",self.num_samples)
        print("GroupSampler.num_groups:  ",num_groups)

    def __iter__(self): 
        indices = []
        for i, size in self.group_sizes:
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0] # get indices in same group 
            assert len(indice) == size
            np.random.shuffle(indice) # 打乱
            num_extra = int(np.ceil(size / self.samples_per_gpu)) * self.samples_per_gpu - len(indice)
            indice = np.concatenate([indice, np.random.choice(indice, num_extra)])
            indices.append(indice)
            # Using the example: group0=100 samples, group1=200, samples_per_gpu=29
            # num_samples = 116 (ceil(100/29)*29) + 203 (ceil(200/29)*29) = 319
            # Note: 116>100 and 203>200, so we pad with extra indices
            # Final output: 319 indices (first 116=group0, last 203=group1) 
            # ensuring each GPU batch contains samples from only one group

        indices = np.concatenate(indices)
       
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        
        return iter(indices)

    def __len__(self):
        return self.num_samples
