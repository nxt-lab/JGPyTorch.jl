# Kernel functions defined by GPyTorch
module kernels

using PyCall

import ..JGPyTorch: Tensor, GPyTorchModule, py_gpytorch, py_torch, get_object, to_torch, from_torch

abstract type AbstractKernel <: GPyTorchModule end

# Common kernel methods

# Compute the covariance between x1 and x2
function forward(k::AbstractKernel, x1, x2, diag::Bool=false)
    return from_torch(get_object(k).forward(to_torch(x1), to_torch(x2), diag), Tensor, typeof(x1), typeof(x2))
end

# Callable kernel functions
function (k::AbstractKernel)(x1, x2, diag::Bool=false)
    return from_torch(get_object(k)(to_torch(x1), to_torch(x2), diag), Tensor, typeof(x1), typeof(x2))
end

# RBF Kernel
mutable struct RBFKernel <: AbstractKernel
    obj::PyObject
    RBFKernel(args...; kwargs...) = new(py_gpytorch.kernels.RBFKernel(args...; kwargs...))
end

# Scale Kernel
mutable struct ScaleKernel <: AbstractKernel
    kern::AbstractKernel
    obj::PyObject
end

ScaleKernel(k::AbstractKernel, args...; kwargs...) = 
    ScaleKernel(k, py_gpytorch.kernels.ScaleKernel(get_object(k), args...; kwargs...))

# RBF Kernel with gradients
mutable struct RBFKernelGrad <: AbstractKernel
    obj::PyObject
    RBFKernelGrad(args...; kwargs...) = new(py_gpytorch.kernels.RBFKernelGrad(args...; kwargs...))
end

end  # module