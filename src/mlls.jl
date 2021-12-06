# Marginal log likelihoods
module mlls

using PyCall

import ..JGPyTorch: Tensor, GPyTorchModule, py_gpytorch, py_torch, get_object, to_torch, from_torch
import ..distributions: AbstractMultivariateNormal
import ..likelihoods: AbstractLikelihood
import ..models: AbstractGPModel

abstract type MarginalLogLikelihood <: GPyTorchModule end

# Common methods

# Computes the MLL given p(f) and y
function forward(mlls::MarginalLogLikelihood, function_dist::AbstractMultivariateNormal, target, args...)
    # GPyTorch's forward method returns a torch.Tensor
    Tensor(get_object(mlls).forward(get_object(function_dist), to_torch(target), args...))
end

# Callable
function (mlls::MarginalLogLikelihood)(function_dist::AbstractMultivariateNormal, target)
    return Tensor(get_object(mlls)(get_object(function_dist), to_torch(target)))
end


# ExactMarginalLogLikelihood
mutable struct ExactMarginalLogLikelihood <: MarginalLogLikelihood
    obj::PyObject
end

function ExactMarginalLogLikelihood(likelihood::AbstractLikelihood, model::AbstractGPModel)
    ExactMarginalLogLikelihood(py_gpytorch.mlls.ExactMarginalLogLikelihood(get_object(likelihood), get_object(model)))
end

end