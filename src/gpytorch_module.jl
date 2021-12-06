# GPyTorch's Module: the parent of many classes (kernels, means, etc.)
# All wrapper types should contain a field named "obj" that is the PyObject object holding the original object in Python

import Base: getproperty, setproperty!

using PyCall

import ..JGPyTorch: py_gpytorch, py_torch

abstract type GPyTorchModule end

get_object(x::GPyTorchModule) = getfield(x, :obj)

function Base.getproperty(x::GPyTorchModule, sym::Symbol)
    # Check if property is an attribute of the GPyTorchModule object to return it
    obj = get_object(x)
    if sym in keys(obj)
        ls = getproperty(get_object(x), sym)
        (isa(ls, PyObject) && pyisinstance(ls, py_torch.Tensor)) ? to_array(Tensor(ls)) : ls
    else
        # Fall back to Julia object's field
        getfield(x, sym)
    end
end

function Base.setproperty!(x::GPyTorchModule, sym::Symbol, value)
    setproperty!(get_object(x), sym, get_object(Tensor(value)))
end

train_mode(x::GPyTorchModule) = get_object(x).train()

eval_mode(x::GPyTorchModule) = get_object(x).eval()