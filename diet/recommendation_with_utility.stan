functions {
  real utility(vector Q, vector RI, int r) {

    // TODO: decide per intake Q_i if we prefer lower or higher levels than RI

    // utility aims to minimize the sum of squares between personal recommendations (Q) and the generally recommended intake (RI)
    
    // - positive differences means that intake (Q) is above the general recommendation (RI)
    vector[r] diffs = RI-Q;
    real sum_of_residuals = sum(fabs(diffs));
    real mean_of_residuals = mean(fabs(diffs));
    real invsum_of_residuals = inv(sum_of_residuals);
    real invmean_of_residuals = inv(mean_of_residuals);
    
    // print("Qs: ", Q);
    // print("RIs: ", RI);
    // print("diffs: ", diffs);
    // print("sum_of_residuals: ", sum_of_residuals);
    // print("invsum_of_residuals: ", invsum_of_residuals);
    // print("loginvsum_of_residuals: ", log(invsum_of_residuals));
    
    return -invmean_of_residuals;
    
    // return (Q[2]);
  }
  
  real concentration_distribution_with_soft_limit(real mu, real alpha, real lowerlimit, real upperlimit, real linear_transformation) {
    
    real lp = 0;
    real beta;
    real log_diff;
    real diff;
    real u;
    real c;
    
    real trans_lowerlimit = lowerlimit + linear_transformation;
    real trans_upperlimit = upperlimit + linear_transformation;

    print("mu, limits: ", mu, ", ", trans_lowerlimit, ", ", trans_upperlimit);
    
    beta = alpha / mu;
    
    log_diff = log_diff_exp(gamma_lcdf(trans_upperlimit | alpha, beta), gamma_lcdf(trans_lowerlimit | alpha, beta));
    u = gamma_cdf(trans_upperlimit, alpha, beta) - gamma_cdf(trans_lowerlimit, alpha, beta);
    lp = -log_diff;
    
    print("u: ", u);
    print("lp: ", lp);
    
    return lp;
  }
  
}

data {
    int<lower=1> responses;    // number of responses
    int<lower=1> p;            // number of predictors
    int<lower=0> r;            // number of conditioned predictors
    real proposal_lowerlimits[r]; // limits of proposals   
    real proposal_upperlimits[r]; // limits of proposals   
    vector[r] general_RI;        // intake statistics for calculating the utility
    vector[r] personal_CI;       // intake statistics for calculating the utility
    real Y_lower_limits[responses];        // lower limits of concentrations
    real Y_upper_limits[responses];        // upper limits of concentrations      

    int posterior_samples;     // number of samples to draw from predicted concentration distrubution

    // point estimates from parameter posteriors (CI5%, CI95%, median..)
    real intercept_point[responses];     
    real alpha_point[responses];         
    vector[p] X_beta_point[responses];   
    vector[r] Q_beta_point[responses];
    
    // sufficient statistics of nutrient variables X
    matrix[p,2] X_evidence;
    real X_sd_coef;
    vector[p] X_evidence_point;
  
    real linear_transformation;
    int repeat_only;
    vector[r] Q_index;
}

transformed data {

  // Estimated value of concentration without the queried predictors (mu_q0)
  real mu_q0[responses];

  real Y_lower_trans[responses];
  real Y_upper_trans[responses];      

  for (m in 1:responses) {
    mu_q0[m] = intercept_point[m] + dot_product(X_evidence_point, X_beta_point[m]);
    Y_lower_trans[m] = Y_lower_limits[m] + linear_transformation;
    Y_upper_trans[m] = Y_upper_limits[m] + linear_transformation;
  } 
}

parameters {
 
 // These limits should match with the priors for sampling to work
 
  real<lower=Y_lower_trans[1], upper=Y_upper_trans[1]> pk; 
  real<lower=Y_lower_trans[2], upper=Y_upper_trans[2]> fppi; 
  real<lower=Y_lower_trans[3], upper=Y_upper_trans[3]> palb;

  //real pk; 
  //real fppi; 
  //real palb;

  vector[r] Q_trans;
}

transformed parameters {

  real<lower=linear_transformation> pk_mu;
  real<lower=linear_transformation> fppi_mu;
  real<lower=linear_transformation> palb_mu;

  vector[r] Q;
  
  Q = Q_trans * 0.05;

  if (repeat_only != 1) {
    pk_mu = mu_q0[1] + dot_product(Q, Q_beta_point[1]) + linear_transformation;
    fppi_mu = mu_q0[2] + dot_product(Q, Q_beta_point[2]) + linear_transformation;
    palb_mu = mu_q0[3] + dot_product(Q, Q_beta_point[3]) + linear_transformation;
  }
  else
  {
    // in repeat, mu_q0 already has all the parameters and Q is empty
    pk_mu = mu_q0[1] + linear_transformation;
    fppi_mu = mu_q0[2] + linear_transformation;
    palb_mu = mu_q0[3] + linear_transformation;
  }
}

model {
  
  // priors
  
  for (i in 1:r) {
      Q[i] ~ uniform(proposal_lowerlimits[i],proposal_upperlimits[i]);
  }
  
  // TODO: nämä rajat pitäisi olla laajemmat, niin että 90% jakaumista on suositusten sisällä

  pk ~ uniform(Y_lower_trans[1],Y_upper_trans[1]);
  fppi ~ uniform(Y_lower_trans[2],Y_upper_trans[2]);
  palb ~ uniform(Y_lower_trans[3],Y_upper_trans[3]);

  //print("after priors:", target());

  // likelihood of Q:s connection to concentration distributions 

  target += gamma_lpdf(pk | alpha_point[1], alpha_point[1] / pk_mu);
  target += gamma_lpdf(fppi | alpha_point[2], alpha_point[2] / fppi_mu);
  target += gamma_lpdf(palb | alpha_point[3], alpha_point[3] / palb_mu);

  //print("after gamma_lpdf:", target());
  
  //print("after limits:", target());
  
  target += utility(Q, general_RI, r);

  //print("after utility:", target());
  
}

generated quantities {
  
  real concentration[posterior_samples, responses];
  real mu_q0_pred[responses];
  real mu_pred[responses];

  {
  vector[p] X;  // posteriors for unmodified nutrients

  // sample X from evidence
  for (i in 1:p)
  {
    // - gaussian or factor?
    if (X_evidence[i,2] != -1) {

      // X_sd_multiplier allows simulating smaller uncertainty of unmodified diet
      if (X_sd_coef > 0)
      {
        X[i] = normal_rng(X_evidence[i,1], X_evidence[i,2] * X_sd_coef);
      }
      else
      {
        X[i] = X_evidence_point[i];
      }
    } else {
      X[i] = X_evidence_point[i];
    }
  }

  for (m in 1:responses) {
    
    mu_q0_pred[m] = intercept_point[m] + dot_product(X_evidence_point, X_beta_point[m]);

    mu_pred[m] = mu_q0_pred[m];
    
    if (repeat_only != 1) {
      mu_pred[m] += dot_product(Q, Q_beta_point[m]);
    }

    for (po in 1:posterior_samples)
    {
      concentration[po, m] = gamma_rng(alpha_point[m], alpha_point[m] / (mu_pred[m] + linear_transformation)) - linear_transformation;
    }

  } // responses
  }
  
}

