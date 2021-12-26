# Models defined by GPyTorch
module models

using PyCall

import ..JGPyTorch: GPyTorchModule, py_gpytorch, py_torch, to_torch, get_object

import ..distributions: wrap_distribution

import ..means: AbstractMean

import ..kernels: AbstractKernel

import ..likelihoods: AbstractLikelihood

import ..optim: TorchOptimizer

abstract type AbstractGPModel <: GPyTorchModule end

# Common methods for models

# Get the parameters of a model (will return a PyObject ready to be used with Python functions)
parameters(model::AbstractGPModel) = PyObject(get_object(model).parameters())

# Callable model functions
function (m::AbstractGPModel)(x::PyObject)
    return get_object(m)(x)
end

function (m::AbstractGPModel)(x)
    return wrap_distribution(m(to_torch(x)))
end

# Convenient train function for models
"""Train a GP model.

Arguments:
    model: the GP model to be trained
    train_x, train_y: training data
    mll: the loss function (marginal likelihood)
    optimizer: the Torch optimizer to be used
    num_iters: the number of iterations (epochs)

Keyword arguments:
    iter_func (function or nothing): a function that is called after each iteration
            iter_func(iter, loss_value)
        The function can be a closure which has direct access to the model, kernel, likelihood, etc.
    test_x, test_y: validation data, used for early stopping
    patience (int): how long to wait after last time validation loss improved, used for early stopping
    delta (real): Minimum change in the monitored quantity to qualify as an improvement, used for early stopping

Early stopping: on if patience > 0; when it's on, the loss
(on the test data if test_x and test_y are specified, otherwise on the training data) is used to determine whether to stop early.
Reference: https://github.com/Bjarten/early-stopping-pytorch
"""
function train!(model::AbstractGPModel, train_x, train_y, mll,
                optimizer::TorchOptimizer, num_iters;
                iter_func::Union{Function, Nothing} = nothing,
                test_x = nothing, test_y = nothing,
                patience::Integer = -1, delta = 0.0)
    # Get the raw Python objects
    py_model = get_object(model)
    py_train_x = to_torch(train_x)
    py_train_y = to_torch(train_y)
    py_mll = get_object(mll)
    py_opt = get_object(optimizer)

    # Early stopping
    early_stopping = (patience > 0)
    if early_stopping
        counter = 0
        best_score = typemin(Float64)
    end

    has_test = !isnothing(test_x) && !isnothing(test_y)
    if has_test
        py_test_x = to_torch(test_x)
        py_test_y = to_torch(test_y)
    end

    # Change model and likelihood to train mode
    py_model.train()
    py_model.likelihood.train()

    # The training loop
    for i = 1:num_iters
        # Zero gradients from previous iteration
        py_opt.zero_grad()
        # Output from model
        output = PyObject(py_model(py_train_x))
        # Calc loss and backprop gradients
        loss = -py_mll(output, py_train_y)
        loss.backward()

        if !isnothing(iter_func)
            iter_func(i, loss.item())
        end

        # Early stopping
        if early_stopping
            # Calculate the loss value
            if has_test
                # Change model and likelihood to eval mode
                py_model.eval()
                py_model.likelihood.eval()

                # Calculate validation loss
                test_output = PyObject(py_model(py_test_x))
                val_loss = -py_mll(test_output, py_test_y).item()

                # Switch back to train mode
                py_model.train()
                py_model.likelihood.train()
            else
                val_loss = loss.item()
            end

            score = -val_loss
            if score - delta < best_score
                counter += 1
                if counter >= patience
                    # Early stop
                    break
                end
            else
                best_score = score
                counter = 0
            end
        end

        optimizer.step()
    end
end


# Each GP type class requires a Python class inheriting from a standard GP class
# However, defining a Python class in a Julia module using @pydef will cause an error "ref of NULL PyObject"
# because the part of @pydef during pre-compilation refers to a Python parent class which was not loaded at that time (hence NULL).
# The work-around is to define the class in Python in the __init__ function, using py"code" and pycall() later on.
function __init__()
    # The model class in Python for ExactGPModel, which will be wrapped by Julia
    #
    # The model class in Python for ExactGPWithDerivativesModel, which will be wrapped by Julia
    # This is very similar to PyExactGP, only the forward() method returns a MultitaskMultivariateNormal
    # However, the mean and covariance functions must be of the appropriate types (*Grad)

    py"""
    import gpytorch
    class PyExactGP(gpytorch.models.ExactGP):
        def __init__(self, meanf, kernf, train_x, train_y, likelihood):
            super(PyExactGP, self).__init__(train_x, train_y, likelihood)
            self.mean_module = meanf
            self.covar_module = kernf
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    class PyExactGPWithDerivatives(PyExactGP):
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    """
end

## Abstract ExactGP model
abstract type AbstractExactGPModel <: AbstractGPModel end

## Common methods for AbstractExactGPModel
function set_train_data(model::AbstractExactGPModel; inputs=nothing, targets=nothing, strict=true)
    if isnothing(inputs) && isnothing(targets)
        return model
    end

    get_object(model).set_train_data(
        isnothing(inputs) ? inputs : to_torch(inputs),
        isnothing(targets) ? targets : to_torch(targets),
        strict)

    return model
end

## ExactGPModel (wrapper)
mutable struct ExactGPModel <: AbstractExactGPModel
    obj::PyObject
end

function ExactGPModel(meanf::AbstractMean, kernf::AbstractKernel, train_x, train_y, likelihood::AbstractLikelihood)
    ExactGPModel(pycall(py"PyExactGP", PyObject,
                        to_torch(meanf),
                        to_torch(kernf),
                        to_torch(train_x),
                        to_torch(train_y),
                        to_torch(likelihood)))
end


## ExactGPWithDerivativesModel (wrapper)
mutable struct ExactGPWithDerivativesModel <: AbstractExactGPModel
    obj::PyObject
end

function ExactGPWithDerivativesModel(meanf::AbstractMean, kernf::AbstractKernel, train_x, train_y, likelihood::AbstractLikelihood)
    ExactGPWithDerivativesModel(pycall(py"PyExactGPWithDerivatives", PyObject,
                                to_torch(meanf),
                                to_torch(kernf),
                                to_torch(train_x),
                                to_torch(train_y),
                                to_torch(likelihood)))
end

end