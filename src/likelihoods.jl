# Likelihood functions defined by GPyTorch
module likelihoods

using PyCall

import ..JGPyTorch: GPyTorchModule, py_gpytorch, py_torch, get_object, to_torch
import ..distributions: wrap_distribution

abstract type AbstractLikelihood <: GPyTorchModule end

# TODO: expose some methods for all likelihoods https://docs.gpytorch.ai/en/latest/likelihoods.html#likelihood

# Callable likelihood functions
function (m::AbstractLikelihood)(x::PyObject)
    return get_object(m)(x)
end

function (m::AbstractLikelihood)(x)
    return wrap_distribution(m(to_torch(x)))
end

# Gaussian Likelihood (for scalar output)
mutable struct GaussianLikelihood <: AbstractLikelihood
    obj::PyObject
    GaussianLikelihood(args...; kwargs...) = new(py_gpytorch.likelihoods.GaussianLikelihood(args...; kwargs...))
end


# Multitask Gaussian Likelihood (with full cross-task covariance structure for the noise)
mutable struct MultitaskGaussianLikelihood <: AbstractLikelihood
    obj::PyObject
    MultitaskGaussianLikelihood(num_tasks, args...; kwargs...) = 
        new(py_gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks, args...; kwargs...))
end

end