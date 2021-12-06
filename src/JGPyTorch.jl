module JGPyTorch

using PyCall
import Base: convert

export Tensor
export get_object, from_array, to_array, train_mode, eval_mode

const py_torch = PyNULL()
const py_gpytorch = PyNULL()

function __init__()
    copy!(py_torch, pyimport("torch"))
    copy!(py_gpytorch, pyimport("gpytorch"))
end

# Basic PyTorch wrappers
include("torch.jl")

# GPyTorch's GPyTorchModule: the parent of many classes (kernels, means, etc.)
# See the file for details
include("gpytorch_module.jl")

include("optim.jl")

include("distributions.jl")

include("means.jl")

include("kernels.jl")

include("likelihoods.jl")

include("models.jl")

include("mlls.jl")

end # module
