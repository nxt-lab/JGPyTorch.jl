# Mean functions defined by GPyTorch
module means

using PyCall

import ..JGPyTorch: Tensor, GPyTorchModule, py_gpytorch, py_torch, get_object, to_torch, from_torch

abstract type AbstractMean <: GPyTorchModule end

# Callable mean functions
function (m::AbstractMean)(x)
    return from_torch(get_object(m)(to_torch(x)), Tensor, typeof(x))
end

# Constant mean
mutable struct ConstantMean <: AbstractMean
    obj::PyObject
    ConstantMean(args...; kwargs...) = new(py_gpytorch.means.ConstantMean(args...; kwargs...))
end

# Constant mean with gradients
mutable struct ConstantMeanGrad <: AbstractMean
    obj::PyObject
    ConstantMeanGrad(args...; kwargs...) = new(py_gpytorch.means.ConstantMeanGrad(args...; kwargs...))
end

end  # module