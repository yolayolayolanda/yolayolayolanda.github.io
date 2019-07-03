---
title: pytorch_functions
date: 2019-07-02 16:12:04
categories: Deep Learning
toc: true
marked: true
---

## Commonly Used PyTorch Functions

<!-- toc -->

#### Saving and Loading a model (Serialization)

- ##### torch.save(*obj, f, pickle_module, pickle_protocol*)

<!-- more -->

> - **obj** – saved object
> - **f** – a file-like object (has to implement write and flush) or a string containing a file name
> - **pickle_module** – module used for pickling metadata and objects
> - **pickle_protocol** – can be specified to override the default protocol

- ##### torch.load(*f, map_location=None, pickle_module=<module 'pickle' from '/opt/conda/lib/python3.6/pickle.py'>, **pickle_load_args*)

  > - **f** – a file-like object (has to implement read, readline, tell, and seek), or a string containing a file name
  > - **map_location** – a function, torch.device, string or a dict specifying how to remap storage locations
  > - **pickle_module** – module used for unpickling metadata and objects (has to match the pickle_module used to serialize file)
  > - **pickle_load_args** – optional keyword arguments passed over to `pickle_module.load`and `pickle_module.Unpickler`, e.g., `encoding=...`.

  ```python
  # Example of whole model
  torch.save(model_object, 'model.pkl')
  model = torch.load('model.pkl')
  # Example of parameters in the model (Recommended)
  torch.save(model_object.state_dict(), 'params.pkl')
  model_object.load_state_dict(torch.load('params.pkl'))
  ```

#### Data Processing

- ##### torch.numel(*input Tensor*) $\rightarrow$ int

  Returns the total number of elements in the `input` tensor.

  ```python
  >>> a = torch.randn(1, 2, 3, 4, 5)
  >>> torch.numel(a)
  120
  >>> a = torch.zeros(4,4)
  >>> torch.numel(a)
  16
  ```

- ##### torch.squeeze(*input Tensor, dim=None, out=None*) $\rightarrow$ Tensor

  Returns a tensor with all the dimensions of `input` of size 1 removed.

  > - **dim** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – if given, the input will be squeezed only in this dimension
  > - **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – the output tensor

```python
>>> x = torch.zeros(2, 1, 2, 1, 2)
>>> x.size()
torch.Size([2, 1, 2, 1, 2])
>>> y = torch.squeeze(x)
>>> y.size()
torch.Size([2, 2, 2])
>>> y = torch.squeeze(x, 0)
>>> y.size()
torch.Size([2, 1, 2, 1, 2])
>>> y = torch.squeeze(x, 1)
>>> y.size()
torch.Size([2, 2, 1, 2])
```

- ##### torch.sort(*input Tensor, (dim), (descending),(out)*)

  Sorts the elements of the `input` tensor along a given dimension in ascending order by value.

  If `dim` is not given, the last dimension of the input is chosen.

  ```python
  >>> x = torch.randn(3, 4)
  >>> sorted, indices = torch.sort(x)
  >>> sorted
  tensor([[-0.2162,  0.0608,  0.6719,  2.3332],
          [-0.5793,  0.0061,  0.6058,  0.9497],
          [-0.5071,  0.3343,  0.9553,  1.0960]])
  >>> indices
  tensor([[ 1,  0,  2,  3],
          [ 3,  1,  0,  2],
          [ 0,  3,  1,  2]])
  
  >>> sorted, indices = torch.sort(x, 0)
  >>> sorted
  tensor([[-0.5071, -0.2162,  0.6719, -0.5793],
          [ 0.0608,  0.0061,  0.9497,  0.3343],
          [ 0.6058,  0.9553,  1.0960,  2.3332]])
  >>> indices
  tensor([[ 2,  0,  0,  1],
          [ 0,  1,  1,  2],
          [ 1,  2,  2,  0]])
  
  ```

- ##### torch.argsort(*input*, *dim=-1*, *descending=False*, *out=None*) → LongTensor

  Returns the **indices** that sort a tensor along a given dimension in ascpending order by value.

- ##### Multiply

  - ###### torch.mul(a, b) - tensor multiplication

    - `torch.mul(a, b)`是矩阵a和b**对应位相乘：**

      **有三种方法：**

      （1）一个tensor乘一个常数

      （2）两个shape完全一致的tensor

      （3）如果形状维度不一致，可以触发boradcast机制的方法：

      **Boradcat总结起来三句话**：

      1. 如果两个数组在维度的数量上有差异，那么维度较少的数组的形状就会被用1填充在它的前导(左)边。

      2. 如果两个数组的形状在任何维度上都不匹配，但等于1，那么在这个维度中，形状为1的数组将被拉伸以匹配另一个形状。

      3. 如果在任何维度上，大小都不一致，且两者都不等于1，就会出现错误.

  - ###### torch.mm - matrixes multiplication

  - ###### torch.prod()

    - `torch.prod`(*input*, *dtype=None*) → Tensor

      Returns the product of all elements in the `input` tensor.

    - `torch.prod`(*input*, *dim*, *keepdim=False*, *dtype=None*) → Tensor

      Returns the product of each row of the `input` tensor in the given dimension `dim`.

      If `keepdim` is `True`, the output tensor is of the same size as `input` except in the dimension `dim` where it is of size 1. Otherwise, `dim` is squeezed (see [`torch.squeeze()`](https://pytorch.org/docs/stable/torch.html#torch.squeeze)), resulting in the output tensor having 1 fewer dimension than `input`.

    - ###### torch.cartesian_prod(**tensors*)

      Do cartesian product of the given sequence of tensors. The behavior is similar to python’s itertools.product.

      ```python
      >>> a = [1, 2, 3]
      >>> b = [4, 5]
      >>> list(itertools.product(a, b))
      [(1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)]
      >>> tensor_a = torch.tensor(a)
      >>> tensor_b = torch.tensor(b)
      >>> torch.cartesian_prod(tensor_a, tensor_b)
      tensor([[1, 4],
              [1, 5],
              [2, 4],
              [2, 5],
              [3, 4],
              [3, 5]])
      ```

      

- ##### torch.clamp(*min, max*)

- ##### torch.gather(*input*, *dim*, *index*, *out=None*, *sparse_grad=False*) → Tensor

  ```python
  b = torch.Tensor([[1,2,3],[4,5,6]])
  print b
  index_1 = torch.LongTensor([[0,1],[2,0]])
  index_2 = torch.LongTensor([[0,1,1],[0,0,0]])
  print torch.gather(b, dim=1, index=index_1)
  print torch.gather(b, dim=0, index=index_2)
  
  ----------------------
   1  2  3
   4  5  6
  [torch.FloatTensor of size 2x3]
  
  
   1  2
   6  4
  [torch.FloatTensor of size 2x2]
  
  
   1  5  6
   1  2  3
  [torch.FloatTensor of size 2x3]
  
  ```

- ##### torch.stack(*seq, dim=1, out=None*) $\rightarrow$ Tensor

  Concatenates sequence of tensors along a new dimension.

  > - **seq** (*sequence of Tensors*) – sequence of tensors to concatenate
  > - **dim** ([*int*](https://docs.python.org/3/library/functions.html#int)) – dimension to insert. Has to be between 0 and the number of dimensions of concatenated tensors (inclusive)
  > - **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – the output tensor

```python
a=torch.rand((1,2))
b=torch.rand((1,2))

c=torch.stack((a,b),0)

c.size()
-----------------------
torch.Size([2, 1, 2])
```

- ##### torch.cat(*inputs, dim*)

```python
>>> x = torch.randn(2, 3)
>>> x
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 0)
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
```

#### Evaluating

- ##### torch.eq(*input*, *other*, *out=None*) → Tensor

  Computes element-wise equality, return 0 at the different place and 1 at the same place.

  The second argument can be a number or a tensor whose shape is [broadcastable](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics) with the first argument.

- ##### torch.bincount(*self*, *weights=None*, *minlength=0*) → Tensor

  Count the frequency of each value in an array of non-negative ints.

- ##### torch.diag(*input*, *diagonal=0*, *out=None*) → Tensor

  - If the input is a vector:

    ```python
    >>> a = torch.randn(3)
    >>> a
    tensor([ 0.5950,-0.0872, 2.3298])
    >>> torch.diag(a)
    tensor([[ 0.5950, 0.0000, 0.0000],
            [ 0.0000,-0.0872, 0.0000],
            [ 0.0000, 0.0000, 2.3298]])
    >>> torch.diag(a, 1)
    tensor([[ 0.0000, 0.5950, 0.0000, 0.0000],
            [ 0.0000, 0.0000,-0.0872, 0.0000],
            [ 0.0000, 0.0000, 0.0000, 2.3298],
            [ 0.0000, 0.0000, 0.0000, 0.0000]])
    ```

    

  - If the input is a matrix:

  ```python
  >>> a = torch.randn(3, 3)
  >>> a
  tensor([[-0.4264, 0.0255,-0.1064],
          [ 0.8795,-0.2429, 0.1374],
          [ 0.1029,-0.6482,-1.6300]])
  >>> torch.diag(a, 0)
  tensor([-0.4264,-0.2429,-1.6300])
  >>> torch.diag(a, 1)
  tensor([ 0.0255, 0.1374])
  ```

  

- ##### torch.histc(*input*, *bins=100*, *min=0*, *max=0*, *out=None*) → Tensor

  ```python
  >>> torch.histc(torch.tensor([1., 2, 1]), bins=4, min=0, max=3)
  tensor([ 0.,  2.,  1.,  0.])
  ```

- ##### torch.trace(*input*) $\rightarrow$ Tensor

  Returns the sum of the elements of the diagonal of the input 2-D matrix.

  ```python
  >>> x = torch.arange(1., 10.).view(3, 3)
  >>> x
  tensor([[ 1.,  2.,  3.],
          [ 4.,  5.,  6.],
          [ 7.,  8.,  9.]])
  >>> torch.trace(x)
  tensor(15.)
  ```

- ##### torch.trill(*input*, *diagonal=0*, *out=None*) → Tensor

  Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices `input`, the other elements of the result tensor `out` are set to 0.

  > - **input** (Tensor) – the input tensor
  > - **diagonal** (int, optional) – the diagonal to consider
  > - **out** (Tensor, optional) – the output tensor

  ```python
  >>> a = torch.randn(3, 3)
  >>> a
  tensor([[-1.0813, -0.8619,  0.7105],
          [ 0.0935,  0.1380,  2.2112],
          [-0.3409, -0.9828,  0.0289]])
  >>> torch.tril(a)
  tensor([[-1.0813,  0.0000,  0.0000],
          [ 0.0935,  0.1380,  0.0000],
          [-0.3409, -0.9828,  0.0289]])
  
  >>> b = torch.randn(4, 6)
  >>> b
  tensor([[ 1.2219,  0.5653, -0.2521, -0.2345,  1.2544,  0.3461],
          [ 0.4785, -0.4477,  0.6049,  0.6368,  0.8775,  0.7145],
          [ 1.1502,  3.2716, -1.1243, -0.5413,  0.3615,  0.6864],
          [-0.0614, -0.7344, -1.3164, -0.7648, -1.4024,  0.0978]])
  # 1 means that the upper triangle is equilateral
  >>> torch.tril(b, diagonal=1)
  tensor([[ 1.2219,  0.5653,  0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.4785, -0.4477,  0.6049,  0.0000,  0.0000,  0.0000],
          [ 1.1502,  3.2716, -1.1243, -0.5413,  0.0000,  0.0000],
          [-0.0614, -0.7344, -1.3164, -0.7648, -1.4024,  0.0000]])
  # -1 means that the lower triangle is equilateral
  >>> torch.tril(b, diagonal=-1)
  tensor([[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.4785,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
          [ 1.1502,  3.2716,  0.0000,  0.0000,  0.0000,  0.0000],
          [-0.0614, -0.7344, -1.3164,  0.0000,  0.0000,  0.0000]])
  ```

  