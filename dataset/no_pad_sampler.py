import torch.distributed as dist
from torch.utils.data import Sampler

class NonPadDistributedSampler(Sampler):
    # Sampler for evaluation. NonPadDistributedSampler return samples without padding, ensuring no duplicates across ranks.
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_initialized():
                raise RuntimeError("Requires distributed package to be initialized")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_initialized():
                raise RuntimeError("Requires distributed package to be initialized")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.total_size = len(dataset)
        self.num_samples = self.total_size // self.num_replicas
        self.remainder = self.total_size % self.num_replicas

        # get the start and end indices for each process.
        if self.rank < self.remainder:
            self.num_samples += 1
            self.start = self.rank * self.num_samples
        else:
            self.start = self.remainder * (self.num_samples + 1) + (self.rank - self.remainder) * self.num_samples
        self.end = self.start + self.num_samples

    def __iter__(self):
        indices = list(range(self.total_size))
        assert self.num_samples == self.end - self.start


        # Returns the index range handled by the current process.
        return iter(indices[self.start:self.end])

    def __len__(self):
        return self.num_samples