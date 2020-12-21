from mytorch import tensor
import numpy as np

class PackedSequence:
    
    '''
    Encapsulates a list of tensors in a packed seequence form which can
    be input to RNN and GRU when working with variable length samples
    
    ATTENTION: The "argument batch_size" in this function should not be confused with the number of samples in the batch for which the PackedSequence is being constructed. PLEASE read the description carefully to avoid confusion. The choice of naming convention is to align it to what you will find in PyTorch. 

    Args:
        data (Tensor):( total number of timesteps (sum) across all samples in the batch, # features ) 
        sorted_indices (ndarray): (number of samples in the batch for which PackedSequence is being constructed,)
        - Contains indices in descending order based on number of timesteps in each sample
        batch_sizes (ndarray): (Max number of timesteps amongst all the sample in the batch,) - ith element of this ndarray represents no.of samples which have timesteps > i
    '''
    def __init__(self,data,sorted_indices,batch_sizes):
        
        # Packed Tensor
        self.data = data # Actual tensor data

        # Contains indices in descending order based on no.of timesteps in each sample
        self.sorted_indices = sorted_indices # Sorted Indices
        
        # batch_size[i] = no.of samples which have timesteps > i
        self.batch_sizes = batch_sizes # Batch sizes
    
    def __iter__(self):
        yield from [self.data,self.sorted_indices,self.batch_sizes]
    
    def __str__(self,):
        return 'PackedSequece(data=tensor({}),sorted_indices={},batch_sizes={})'.format(str(self.data),str(self.sorted_indices),str(self.batch_sizes))


def pack_sequence(sequence): 
    '''
    Constructs a packed sequence from an input sequence of tensors.
    By default assumes enforce_sorted ( compared to PyTorch ) is False
    i.e the length of tensors in the sequence need not be sorted (desc).

    Args:
        sequence (list of Tensor): ith tensor in the list is of shape (Ti,K)
        where Ti is the number of time steps in sample i and K is the # features
    Returns:
        PackedSequence: data attribute of the result is of shape ( total number of timesteps (sum) across all samples in the batch, # features )
    '''
    
    # TODO: INSTRUCTIONS
    # Find the sorted indices based on number of time steps in each sample
    sample_time = [s.shape[0] for s in sequence]
    sorted_idx = np.argsort(sample_time, 0)[::-1]

    # Extract slices from each sample and properly order them for the construction of the packed tensor.
    # __getitem__ you defined for Tensor class will come in handy
    time_list = []
    batch_sizes = []
    for time in range(max(sample_time)):
        size = 0
        for idx in sorted_idx:
            if time >= sample_time[idx]:
                continue
            else:
                time_list.append(sequence[idx][time:time+1])
                size += 1
        batch_sizes.append(size)

    assert len(batch_sizes) == max(sample_time)
    data = tensor.cat(time_list, dim=0) # concat along the time axis


    # Use the tensor.cat function to create a single tensor from the re-ordered segements

    # Finally construct the PackedSequence object
    # REMEMBER: All operations here should be able to construct a valid autograd graph.

    # construct the PackedSequence object
    return PackedSequence(data=data, sorted_indices=sorted_idx, batch_sizes=np.array(batch_sizes))
    # raise NotImplementedError('Implement pack_Sequence!')

def unpack_sequence(ps):
    '''
    Given a PackedSequence, this unpacks this into the original list of tensors.
    
    NOTE: Attempt this only after you have completed pack_sequence and understand how it works.

    Args:
        ps (PackedSequence)
    Returns:
        list of Tensors
    '''
    
    # TODO: INSTRUCTIONS
    # This operation is just the reverse operation of pack_sequences
    # Use the ps.batch_size to determine number of time steps in each tensor of the original list
    # (assuming the tensors were sorted in a descending fashion based on number of timesteps)

    steps = []
    for i, number in enumerate(ps.batch_sizes[:-1]):
        if number - ps.batch_sizes[i+1] > 0:
            steps.extend([i+1] * (number - ps.batch_sizes[i+1]))
    steps += ps.batch_sizes[-1] * [len(ps.batch_sizes)]
    assert len(steps) == len(ps.sorted_indices)

    # Construct these individual tensors using tensor.cat
    unpack_tensor = []
    cumsum_time = np.cumsum(ps.batch_sizes)

    for idx, step in enumerate(steps[::-1]):

        slice = [idx] + list(cumsum_time[:step-1] + idx)
        unpack_tensor.append(ps.data[slice])
        # ps.batch_sizes[:step]


    # Re-arrange this list of tensor based on ps.sorted_indices
    orig_tensor = [0] * len(ps.sorted_indices)
    for loc, tens in zip(ps.sorted_indices, unpack_tensor):
        orig_tensor[loc] = tens


    return orig_tensor
    # raise NotImplementError('Implement unpack_sequence')

