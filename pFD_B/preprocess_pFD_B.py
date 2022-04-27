import torch
import numpy as np
train_dict = torch.load('train.pt')
val_dict = torch.load('val.pt')
test_dict = torch.load('test.pt')
print(train_dict['samples'].shape)
np.save('train_x.npy', train_dict['samples'].transpose(1,2))
np.save('test_x.npy', test_dict['samples'].transpose(1,2))
np.save('val_x.npy', val_dict['samples'].transpose(1,2))

def scatter_numpy(self, dim, index, src):
    """
    Writes all values from the Tensor src into self at the indices specified in the index Tensor.

    :param dim: The axis along which to index
    :param index: The indices of elements to scatter
    :param src: The source element(s) to scatter
    :return: self
    """
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    if self.ndim != index.ndim:
        raise ValueError("Index should have the same number of dimensions as output")
    if dim >= self.ndim or dim < -self.ndim:
        raise IndexError("dim is out of range")
    if dim < 0:
        # Not sure why scatter should accept dim < 0, but that is the behavior in PyTorch's scatter
        dim = self.ndim + dim
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = self.shape[:dim] + self.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and output should be the same size")
    if (index >= self.shape[dim]).any() or (index < 0).any():
        raise IndexError("The values of index must be between 0 and (self.shape[dim] -1)")

    def make_slice(arr, dim, i):
        slc = [slice(None)] * arr.ndim
        slc[dim] = i
        return slc

    # We use index and dim parameters to create idx
    # idx is in a form that can be used as a NumPy advanced index for scattering of src param. in self
    idx = [[*np.indices(idx_xsection_shape).reshape(index.ndim - 1, -1),
            index[make_slice(index, dim, i)].reshape(1, -1)[0]] for i in range(index.shape[dim])]
    idx = list(np.concatenate(idx, axis=1))
    idx.insert(dim, idx.pop())

    if not np.isscalar(src):
        if index.shape[dim] > src.shape[dim]:
            raise IndexError("Dimension " + str(dim) + "of index can not be bigger than that of src ")
        src_xsection_shape = src.shape[:dim] + src.shape[dim + 1:]
        if idx_xsection_shape != src_xsection_shape:
            raise ValueError("Except for dimension " +
                             str(dim) + ", all dimensions of index and src should be the same size")
        # src_idx is a NumPy advanced index for indexing of elements in the src
        src_idx = list(idx)
        src_idx.pop(dim)
        src_idx.insert(dim, np.repeat(np.arange(index.shape[dim]), np.prod(idx_xsection_shape)))
        self[idx] = src[src_idx]

    else:
        self[idx] = src

    return self

n_classes = 3
train_y = np.zeros((len(train_dict['labels']), n_classes))
val_y = np.zeros((len(val_dict['labels']), n_classes))
test_y = np.zeros((len(test_dict['labels']), n_classes))
scatter_numpy(train_y, 1, np.expand_dims(train_dict['labels'].numpy().astype(int), axis=1), 1)
scatter_numpy(val_y, 1, np.expand_dims(val_dict['labels'].numpy().astype(int), axis=1), 1)
scatter_numpy(test_y, 1, np.expand_dims(test_dict['labels'].numpy().astype(int), axis=1), 1)
np.save('train_y.npy', train_y)
np.save('test_y.npy', test_y)
np.save('val_y.npy', val_y)