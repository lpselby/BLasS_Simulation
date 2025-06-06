
// Stan model for the BLaS framework
// Models a binary outcome Y via a latent variable Y* ~ sinh-arcsinh(mu, sigma, skew, kurtosis)
// with thresholding at c = 0

functions {
  real sas_logpdf(real y, real mu, real sigma, real epsilon, real delta) {
    // log PDF of the sinh-arcsinh distribution
    real z = (y - mu) / sigma;
    real w = asinh(z);
    real g = sinh((w - epsilon) / delta);
    real log_jacobian = log1p(square(z)) - log(delta) - log(sigma);
    return normal_lpdf(g | 0, 1) + log_jacobian;
  }

  real sas_cdf(real y, real mu, real sigma, real epsilon, real delta) {
    // approximate CDF of the sinh-arcsinh
    real z = (y - mu) / sigma;
    real w = asinh(z);
    real g = sinh((w - epsilon) / delta);
    return Phi(g);  // standard normal CDF
  }
}

data {
  int<lower=1> N;               // number of observations
  //int<lower=0,upper=1> y[N];    // binary outcome
  int<lower=1> K;               // number of predictors
  array[N] int<lower=0, upper=1> y;    // binary outcome
  matrix[N, K] X_mu;            // predictors for mean
  matrix[N, K] X_sigma;         // predictors for scale
  matrix[N, K] X_skew;          // predictors for skewness
  matrix[N, K] X_kurt;          // predictors for kurtosis
}

parameters {
  vector[K] beta_mu;
  vector[K] beta_sigma;
  vector[K] beta_skew;
  vector[K] beta_kurt;
}

model {
  vector[N] mu = X_mu * beta_mu;
  vector[N] sigma = exp(X_sigma * beta_sigma);  // ensure positive
  vector[N] skew = X_skew * beta_skew;
  vector[N] kurt = exp(X_kurt * beta_kurt);     // ensure positive

  // priors
  beta_mu ~ normal(0, 2);
  beta_sigma ~ normal(0, 2);
  beta_skew ~ normal(0, 2);
  beta_kurt ~ normal(0, 2);

  for (n in 1:N) {
    real p = 1 - sas_cdf(0 | mu[n], sigma[n], skew[n], kurt[n]);  // P(Y* > 0)
    y[n] ~ bernoulli(p);
  }
}
