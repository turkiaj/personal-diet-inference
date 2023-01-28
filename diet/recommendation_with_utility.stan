functions {
  
  real diet_preference(vector Q, vector RI, int r, real preference_strength) {

    // this preference function aims to minimize the sum of squares between personal recommendations (Q) and current levels of Q
    
    vector[r] diffs = Q - RI;
    real sum_of_errors = sum(fabs(diffs));
    
    // - exponential function integrates to 1 when integrated from 0 to infinity like probability distributions
    // - negating the sum of errors makes function to increase as error decreases
    
    return exp(-preference_strength * sum_of_errors);
  }
  
}

data {
    int<lower=1> responses;    // number of responses
    int<lower=1> p;            // number of predictors
    int<lower=0> r;            // number of conditioned predictors
    vector[r] proposal_lowerlimits; // limits of proposals   
    vector[r] proposal_upperlimits; // limits of proposals   
    vector[r] general_RI;        // general intake recommendation for calculating the utility
    vector[r] current_Q;       // current personal intake level for calculating the utility 
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
    vector[p] X_evidence_point;
    
    real preference_strength;
    real linear_transformation;
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
  
  preference = log(diet_preference(Q, current_Q, r, preference_strength));
}

model {
  
  // priors for nutrients
  
  for (i in 1:r) {
    target += uniform_lpdf(Q[i] | proposal_lowerlimits[i], proposal_upperlimits[i]);
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
  real mu_pred[responses];
  real mu_q0_pred[responses];
  
  for (m in 1:responses) {
    
    mu_q0_pred[m] = intercept_point[m] + dot_product(X_evidence_point, X_beta_point[m]);

    mu_pred[m] = mu_q0_pred[m] + dot_product(Q, Q_beta_point[m]);
    
    for (po in 1:posterior_samples)
    {
      concentration[po, m] = gamma_rng(alpha_point[m], alpha_point[m] / (mu_pred[m] + linear_transformation)) - linear_transformation;
    }
  }

}