functions {
  
  real utility(vector Q, vector RI, vector Q_upperlimit, int r) {

    // utility aims to minimize the sum of squares between personal recommendations (Q) and the generally recommended intake (RI)
    
    // - positive differences means that intake (Q) is above the general recommendation (RI)
    vector[r] diffs = RI-Q;
    real sum_of_residuals = sum(fabs(diffs));
    real mean_of_residuals = mean(fabs(diffs));
    real invsum_of_residuals = inv(sum_of_residuals);
    real invmean_of_residuals = inv(mean_of_residuals);
    
    return -log(sum_of_residuals);
    
    // print("Qs: ", Q);
    // print("RIs: ", RI);
    // print("diffs: ", diffs);
    // print("sum_of_residuals: ", sum_of_residuals);
    // print("invsum_of_residuals: ", invsum_of_residuals);
    // print("loginvsum_of_residuals: ", log(invsum_of_residuals));
  }
  
}

data {
    int<lower=1> responses;    // number of responses
    int<lower=1> p;            // number of predictors
    int<lower=0> r;            // number of conditioned predictors
    vector[r] proposal_lowerlimits; // limits of proposals   
    vector[r] proposal_upperlimits; // limits of proposals   
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
 
  vector<lower=proposal_lowerlimits, upper=proposal_upperlimits>[r] Q;
  
}

transformed parameters {

  real preference;

  real pk_mu; 
  real fppi_mu;
  real palb_mu;

  pk_mu = mu_q0[1] + dot_product(Q, Q_beta_point[1]) + linear_transformation;
  fppi_mu = mu_q0[2] + dot_product(Q, Q_beta_point[2]) + linear_transformation;
  palb_mu = mu_q0[3] + dot_product(Q, Q_beta_point[3]) + linear_transformation;
  
  preference = utility(Q, general_RI, proposal_upperlimits, r);
}

model {
  
  // priors for nutrients
  
  for (i in 1:r) {
       Q[i] ~ uniform(proposal_lowerlimits[i],proposal_upperlimits[i]);
  }
  
  // soft limits for concentrations 
  
  target += gamma_lpdf(pk_mu | alpha_point[1],  alpha_point[1] / Y_lower_trans[1]);
  target += gamma_lpdf(pk_mu | alpha_point[1],  alpha_point[1] / Y_upper_trans[1]);

  target += gamma_lpdf(fppi_mu | alpha_point[2],  alpha_point[2] / Y_lower_trans[2]);
  target += gamma_lpdf(fppi_mu | alpha_point[2],  alpha_point[2] / Y_upper_trans[2]);

  target += gamma_lpdf(palb_mu | alpha_point[3],  alpha_point[3] / Y_lower_trans[3]);
  target += gamma_lpdf(palb_mu | alpha_point[3],  alpha_point[3] / Y_upper_trans[3]);

  // select preferred diet from all the equal options

  target += preference;

  //print("target: ", target());

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

    mu_pred[m] = mu_q0_pred[m] + dot_product(Q, Q_beta_point[m]);
    
    for (po in 1:posterior_samples)
    {
      concentration[po, m] = gamma_rng(alpha_point[m], alpha_point[m] / (mu_pred[m] + linear_transformation)) - linear_transformation;
    }

  } // responses
  }
  
}