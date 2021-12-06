# Demonstrate how to use GPyTorch from Julia with JGPyTorch for GP regression with derivative information.
# It is based on the GP regression with derivative observations in GPyTorch at https://docs.gpytorch.ai/en/stable/examples/08_Advanced_Usage/Simple_GP_Regression_Derivative_Information_2d.html.

using JGPyTorch
using LinearAlgebra
using Statistics
using Printf

using PyCall

# The ground truth function
function franke(X, Y)
    term1 = @. 0.75exp(-((9*X - 2)^2 + (9*Y - 2)^2)/4)
    term2 = @. 0.75exp(-((9*X + 1)^2)/49 - (9*Y + 1)/10)
    term3 = @. 0.5exp(-((9*X - 7)^2 + (9*Y - 3)^2)/4)
    term4 = @. 0.2exp(-(9*X - 4)^2 - (9*Y - 7)^2)

    f = term1 .+ term2 .+ term3 .- term4
    dfx = @. -2*(9*X - 2)*9/4 * term1 - 2*(9*X + 1)*9/49 * term2 +
          -2*(9*X - 7)*9/4 * term3 + 2*(9*X - 4)*9 * term4
    dfy = @. -2*(9*Y - 2)*9/4 * term1 - 9/10 * term2 +
          -2*(9*Y - 3)*9/4 * term3 + 2*(9*Y - 7)*9 * term4

    return f, dfx, dfy
end

# Create a mesh grid for inputs
train_x = vcat([[x, y]' for x in LinRange(0, 1, 5), y in LinRange(0, 1, 5)]...)
f, dfx, dfy = franke(train_x[:, 1], train_x[:, 2])
train_y = [f dfx dfy]
train_y .+= 0.05 * randn(size(train_y))

# mean and kernel
mean_func = JGPyTorch.means.ConstantMeanGrad()
base_kernel = JGPyTorch.kernels.RBFKernelGrad(ard_num_dims=2)
kern_func = JGPyTorch.kernels.ScaleKernel(base_kernel)

# initialize likelihood and model
likelihood = JGPyTorch.likelihoods.MultitaskGaussianLikelihood(3)  # Value + x-derivative + y-derivative
model = JGPyTorch.models.ExactGPWithDerivativesModel(mean_func, kern_func, train_x, train_y, likelihood)

# Find optimal model hyperparameters
# First, switch to train mode
train_mode(model)
train_mode(likelihood)

# Use the adam optimizer
optimizer = JGPyTorch.optim.Adam(JGPyTorch.models.parameters(model), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood    
mll = JGPyTorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

function myprint(i, loss)
    # ls = model.covar_module.base_kernel.lengthscale.squeeze().detach().numpy()
    ls = base_kernel.lengthscale
    @printf("Iter %d - Loss: %.3f   lengthscale: %.3f,%.3f   noise: %.3f\n", i, loss, ls[1], ls[2], model.likelihood.noise.item())
end

JGPyTorch.models.train!(model, train_x, train_y, mll, optimizer, 500; iter_func=myprint, patience=7)

eval_mode(model)
eval_mode(likelihood)

## Let's consider a standard GP without derivatives

# mean and kernel
std_mean_func = JGPyTorch.means.ConstantMean()
std_base_kernel = JGPyTorch.kernels.RBFKernel()
std_kern_func = JGPyTorch.kernels.ScaleKernel(std_base_kernel)

# initialize likelihood and model
std_likelihood = JGPyTorch.likelihoods.GaussianLikelihood()
std_model = JGPyTorch.models.ExactGPModel(std_mean_func, std_kern_func, train_x, train_y[:,1], std_likelihood)

# Find optimal model hyperparameters
train_mode(std_model)
train_mode(std_likelihood)

# Use the adam optimizer
std_optimizer = JGPyTorch.optim.Adam(JGPyTorch.models.parameters(std_model), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood    
std_mll = JGPyTorch.mlls.ExactMarginalLogLikelihood(std_likelihood, std_model)

function std_myprint(i, loss)
    # ls = model.covar_module.base_kernel.lengthscale.squeeze().detach().numpy()
    ls = std_base_kernel.lengthscale
    @printf("Iter %d - Loss: %.3f   lengthscale: %.3f,%.3f   noise: %.3f\n", i, loss, ls[1], ls[2], std_model.likelihood.noise.item())
end

JGPyTorch.models.train!(std_model, train_x, train_y[:,1], std_mll, std_optimizer, 500; iter_func=myprint, patience=7)

eval_mode(std_model)
eval_mode(std_likelihood)


## Test points
n1, n2 = 50, 50
xv = vec(LinRange(0, 1, n1)' .* ones(n2))
yv = vec(ones(1, n1) .* LinRange(0, 1, n2))

fv, dfxv, dfyv = franke(xv, yv)

test_x = [xv yv]
# pred_mean = nothing
# @pywith JGPyTorch.py_torch.no_grad() begin
#     @pywith JGPyTorch.py_gpytorch.settings.fast_computations(log_prob=false, covar_root_decomposition=false) begin
#         global pred_mean
        predictions = likelihood(model(test_x))
        pred_mean = predictions.mean
#     end
# end

r²_score(pred, target) = 1 - sum((pred .- target).^2) / sum((target .- mean(target)).^2)

@printf("The r² score is %.4f\n", r²_score(pred_mean[:,1], fv))

# std_pred_mean = nothing
# @pywith JGPyTorch.py_torch.no_grad() begin
#     @pywith JGPyTorch.py_gpytorch.settings.fast_computations(log_prob=false, covar_root_decomposition=false) begin
#         global std_pred_mean
        predictions = std_likelihood(std_model(test_x))
        std_pred_mean = predictions.mean
#     end
# end

@printf("The r² score is %.4f\n", r²_score(std_pred_mean[:,1], fv))