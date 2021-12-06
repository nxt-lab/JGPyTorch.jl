# Distributions defined by GPyTorch
module distributions

using PyCall

import ..JGPyTorch: Tensor, AbstractTorchDistribution, TorchDistribution, py_gpytorch, py_torch, get_object, to_array, from_array

export wrap_distribution

# Wrap a distribution object from Python with an appropriate type
function wrap_distribution(d::PyObject)
    if pyisinstance(d, py_gpytorch.distributions.multivariate_normal.MultivariateNormal)
        MultivariateNormal(d)
    elseif pyisinstance(d, py_gpytorch.distributions.multitask_multivariate_normal.MultitaskMultivariateNormal)
        MultitaskMultivariateNormal(d)
    else
        TorchDistribution(d)
    end
end

# TODO: expose some methods for distributions

abstract type AbstractMultivariateNormal <: AbstractTorchDistribution end
    
# MultivariateNormal
mutable struct MultivariateNormal <: AbstractMultivariateNormal
    obj::PyObject
end
MultivariateNormal(mean, covariance_matrix, args...) = 
    MultivariateNormal(py_gpytorch.distributions.MultivariateNormal(get_object(Tensor(mean)), get_object(Tensor(covariance_matrix)), args...))

# MultitaskMultivariateNormal
mutable struct MultitaskMultivariateNormal <: AbstractMultivariateNormal
    obj::PyObject
end
MultitaskMultivariateNormal(mean, covariance_matrix, args...) = 
    MultitaskMultivariateNormal(py_gpytorch.distributions.MultitaskMultivariateNormal(get_object(Tensor(mean)), get_object(Tensor(covariance_matrix)), args...))

end