
# BLaS Simulation and Estimation in R
# Dependencies: rstan or cmdstanr
install.packages("devtools")
devtools::install_github("stan-dev/cmdstanr", dependencies = TRUE)

library(cmdstanr)
install_cmdstan()
cmdstanr::cmdstan_path()


library(posterior)
library(bayesplot)

model <- cmdstan_model("blas_simmodel.stan")

# Step 1: Simulate data
set.seed(123)
N <- 2000
K <- 4

X <- matrix(runif(N * K, -1, 1), nrow = N)
beta_mu <- c(0.5, -0.3, 0, 0)
beta_sigma <- c(0.2, 0, 0.4, 0)
beta_skew <- c(0, 0.6, 0, -0.2)
beta_kurt <- c(0.3, 0, 0, 0.3)

mu <- X %*% beta_mu
sigma <- exp(X %*% beta_sigma)
skew <- X %*% beta_skew
kurt <- exp(X %*% beta_kurt)

# Approximate sinh-arcsinh latent Y*
z <- rnorm(N)
w <- sinh((asinh(z) - skew) / kurt)
y_star <- mu + sigma * w
y <- as.integer(y_star > 0)

# Step 2: Prepare data for Stan
stan_data <- list(
  N = N,
  y = y,
  K = K,
  X_mu = X,
  X_sigma = X,
  X_skew = X,
  X_kurt = X
)

# Step 3: Compile and fit the model
model <- cmdstan_model("blas_simmodel.stan")
fit <- model$sample(data = stan_data, chains = 4, parallel_chains = 4, iter_warmup = 1000, iter_sampling = 1000, init = 0.01)

# Step 4: Diagnostics
print(fit)
mcmc_trace(fit$draws(), pars = c("beta_mu[1]", "beta_sigma[1]", "beta_skew[1]", "beta_kurt[1]"))

# Save draws or summary
posterior_summary <- as_draws_df(fit$draws())
write.csv(posterior_summary, "blas_sim_posterior_summary.csv", row.names = FALSE)

