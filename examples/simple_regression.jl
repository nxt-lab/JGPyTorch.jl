# Demonstrate how to use GPyTorch from Julia with JGPyTorch.
# It is based on the basic GP regression tutorial in GPyTorch at https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html.
using JGPyTorch
using LinearAlgebra
using Statistics
using Printf

# Training data is 100 points in [0,1] inclusive regularly spaced
Npoints = 100
train_x = LinRange(0, 1, Npoints)
# True function is sin(2*pi*x) with Gaussian noise
train_y = sin.(2π .* train_x) .+ randn(size(train_x)) * √0.04

# mean and kernel
mean_func = JGPyTorch.means.ConstantMean()
kern_func = JGPyTorch.kernels.ScaleKernel(JGPyTorch.kernels.RBFKernel())

# initialize likelihood and model
likelihood = JGPyTorch.likelihoods.GaussianLikelihood()
model = JGPyTorch.models.ExactGPModel(mean_func, kern_func, train_x, train_y, likelihood)

# Find optimal model hyperparameters
# First, switch to train mode
train_mode(model)
train_mode(likelihood)

# Use the adam optimizer
optimizer = JGPyTorch.optim.Adam(JGPyTorch.models.parameters(model), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood    
mll = JGPyTorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

myprint(i, loss) =
    @printf("Iter %d - Loss: %.3f   lengthscale: %.3f   noise: %.3f\n", i, loss, kern_func.base_kernel.lengthscale.item(), model.likelihood.noise.item())

JGPyTorch.models.train!(model, train_x, train_y, mll, optimizer, 50; iter_func=myprint, patience=7)

eval_mode(model)
eval_mode(likelihood)


# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
test_x = LinRange(0, 1, 51)
observed_pred = likelihood(model(test_x))

fv = sin.(2π .* test_x)

f_preds = model(test_x)

f_mean = f_preds.mean
f_var = f_preds.variance
f_covar = f_preds.covariance_matrix

r²_score(pred, target) = 1 - sum((pred .- target).^2) / sum((target .- mean(target)).^2)

@printf("The r² score is %.4f\n", r²_score(f_mean, fv))