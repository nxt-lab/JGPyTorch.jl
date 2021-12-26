# JGPyTorch.jl

A simple wrapper of GPyTorch (Gaussian process library for Python using PyTorch) in Julia.  This is not a complete wrapper.  It is created for and used in our research, so only the features needed by our research will be wrapped.

## Requirements
- `PyCall.jl`
- In the Python environment used by PyCall, install `GPyTorch` and of course `PyTorch`. Note: while PyTorch can be installed via Conda, in my experience, it may not work (well) with PyCall, therefore you are recommended to install PyTorch via Pip instead.
