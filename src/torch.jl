import Base: getproperty, setproperty!

abstract type TorchObject end

get_object(x::TorchObject) = getfield(x, :obj)

function Base.getproperty(x::TorchObject, sym::Symbol)
    # Check if property is an attribute of the torch object to return it
    obj = get_object(x)
    if sym in keys(obj)
        ls = getproperty(get_object(x), sym)
        (isa(ls, PyObject) && pyisinstance(ls, py_torch.Tensor)) ? to_array(Tensor(ls)) : ls
    else
        # Fall back to Julia object's field
        getfield(x, sym)
    end
end

function Base.setproperty!(x::TorchObject, sym::Symbol, value::PyObject)
    setproperty!(get_object(x), sym, value)
end

function Base.setproperty!(x::TorchObject, sym::Symbol, value::Array)
    setproperty!(get_object(x), sym, get_object(Tensor(value)))
end

function Base.setproperty!(x::TorchObject, sym::Symbol, value)
    setproperty!(get_object(x), sym, get_object(value))
end


# In general, the construction of a tensor will copy data from the original object, unless it's already a tensor
mutable struct Tensor <: TorchObject
    obj::PyObject
    function Tensor(x)
        if isa(x, PyObject)
            if pyisinstance(x, py_torch.Tensor) || pyisinstance(x, py_gpytorch.lazy.LazyTensor)
                new(x)
            else
                new(py_torch.Tensor(x))
            end
        else
            Tensor(PyObject(x))
        end
    end
    Tensor(x::Tensor) = x
end

backward(t::Tensor) = get_object(t).backward()

# Create a tensor that shares the data with the numpy array created from a Julia array
from_array(x::Array) = Tensor(py_torch.from_numpy(x))

function to_array(x::Tensor)
    obj = get_object(x)
    if pyisinstance(obj, py_torch.Tensor)
        return obj.detach().numpy()
    else
        return obj.evaluate().detach().numpy()
    end
end

# Convert an object from Julia to PyTorch
# Rule: PyObject -> PyObject, Array -> Tensor -> PyObject, Number -> Array -> Tensor -> PyObject, otherwise just get_object
to_torch(x::PyObject) = x
to_torch(x::Number) = to_torch([x])
to_torch(x) = get_object(x)
to_torch(x::AbstractArray) = to_torch(collect(x))

to_torch(x::Vector) = py_torch.Tensor(x)  # py_torch.from_numpy(x)

# Convert an array from Julia to Torch, optionally guaranteeing row major (default)
# The row major guarantee can be critical for some PyTorch functions
# PyCall doesn't guarantee row major, especially in its automatic type conversion between Python and Julia,
# and often result in column-major.
function to_torch(x::Matrix, row_major=true)
    xt = py_torch.Tensor(PyObject(x))  # Standard conversion to Tensor
    if !row_major || xt.stride()[1] == size(x)[end]
        return xt
    end

    # At this point, we need to fix the row major
    # There are several ways:
    # 1. PyReverseDims(collect(x')) will create a numpy array with row major, which can be converted into Tensor
    # 2. PyObject(collect(x')') is similar, the performance is slightly better than (1)
    return py_torch.Tensor(PyObject(collect(x')'))
end

# Convert an object from PyTorch to Julia, with one type argument
# from_torch(x, Type{T}, Type{A}) converts a PyTorch object x to Julia based on the type A
# Rule:
#   A = PyObject -> as is
#   A = Array -> to_array(T(x)), T is often a Tensor
#   otherwise -> T(x)
from_torch(x, ::Type{T}, ::Type{PyObject}) where {T} = x
from_torch(x, ::Type{T}, ::Type{A}) where {T, A <: AbstractArray} = to_array(T(x))
from_torch(x, ::Type{T}, ::Type{A}) where {T, A} = T(x)

# Convert an object from PyTorch to Julia, with two type arguments
# from_torch(x, Type{T}, Type{A}, Type{B}) converts a PyTorch object x to Julia based on the types A and B
# Rule:
#   A = Array and B = Array -> to_array(T(x)), T is often a Tensor
#   A = PyObject or B = PyObject -> as is
#   otherwise -> T(x)
from_torch(x, ::Type{T}, ::Type{A}, ::Type{PyObject}) where {T,A} = x
from_torch(x, ::Type{T}, ::Type{PyObject}, ::Type{B}) where {T,B} = x
from_torch(x, ::Type{T}, ::Type{A}, ::Type{B}) where {T, A <: AbstractArray, B <: AbstractArray} = to_array(T(x))
from_torch(x, ::Type{T}, ::Type{A}, ::Type{B}) where {T,A,B} = T(x)


# Distributions
abstract type AbstractTorchDistribution <: TorchObject end

mutable struct TorchDistribution <: AbstractTorchDistribution
    obj::PyObject

    function TorchDistribution(x)
        if isa(x, PyObject) && pyisinstance(x, py_torch.distributions.distribution.Distribution)
            new(x)
        else
            error("Only a torch.distributions.distribution.Distribution can be wrapped.")
        end
    end
    
    TorchDistribution(x::TorchDistribution) = x
end