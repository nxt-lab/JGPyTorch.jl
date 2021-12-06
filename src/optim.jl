# Optimization algorithms (most are from Torch)
module optim

using PyCall

import ..JGPyTorch: TorchObject, Tensor, py_torch, to_torch, from_torch

abstract type TorchOptimizer <: TorchObject end

# Common methods for torch.Optimizer
step(opt::TorchOptimizer, args...) = get_object(opt).step(args...)
zero_grad(opt::TorchOptimizer, set_to_none=false) = get_object(opt).zero_grad(set_to_none)

# Specific algorithms

# Adam
mutable struct Adam <: TorchOptimizer
    obj::PyObject
    Adam(params; kwargs...) = new(py_torch.optim.Adam(params; kwargs...))
end

# Adagrad
mutable struct Adagrad <: TorchOptimizer
    obj::PyObject
    Adagrad(params; kwargs...) = new(py_torch.optim.Adagrad(params; kwargs...))
end

# LBFGS
mutable struct LBFGS <: TorchOptimizer
    obj::PyObject
    LBFGS(params; kwargs...) = new(py_torch.optim.LBFGS(params; kwargs...))
end

# LBFGS
# learning rate (lr argument) is required
mutable struct SGD <: TorchOptimizer
    obj::PyObject
    SGD(params, lr; kwargs...) = new(py_torch.optim.SGD(params, lr; kwargs...))
end

end