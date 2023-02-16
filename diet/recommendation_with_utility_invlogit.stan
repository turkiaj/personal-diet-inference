functions {
  
  // This function defines a custom probability distribution that receives 
  // its maximum value when the current diet proposal is closest to most preferred diet
  real diet_preference(vector Q, vector RI, int r, real preference_strength) {

    // This function prefers such a personal diet proposal that is closest to person's current diet
    // and it is achieven by minimizing the absolute sum of squares between personal recommendations (Q) and current personal levels of RI
    
    vector[r] diffs = Q - RI;
    real sum_of_errors = sum(fabs(diffs));
    
    return exponential_lpdf(sum_of_errors | preference_strength);
  }
  
}

data {
    int<lower=1> responses;        // number of responses
    int<lower=1> p;                // number of predictors
    int<lower=0> r;                // number of conditioned predictors
    vector[r] proposal_lowerlimits; // limits of proposals   
    vector[r] proposal_upperlimits; // limits of proposals   
    vector[r] general_RI;        // general intake recommendation for calculating the preference function
    vector[r] current_Q;         // current personal intake level for calculating the preference function 
    real Y_lower_limits[responses];        // lower limits of concentrations
    real Y_upper_limits[responses];        // upper limits of concentrations      

    int posterior_samples;      // number of samples to draw from predicted concentration distrubution

    // point estimates from parameter posteriors (CI5%, CI95%, median..)
    // - these point estimates are used instead of full posterior distributions for 
    // making the proposed intake distributions Q more concise
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
  // This part of the expected value (mu) does not change during the sampling
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
 
  // Resulting parameters of this inference are these personal recommendations for intake proposals Q
  // Each nutrient Q is given lower and upperlimits for its recommendation
  vector<lower=proposal_lowerlimits, upper=proposal_upperlimits>[r] Q;

}

transformed parameters {

  real softbound_sum;
  real in_concentration_range;
  real preference;

  real pk_mu;
  real fppi_mu;
  real palb_mu;

  pk_mu = mu_q0[1] + dot_product(Q, Q_beta_point[1]) + linear_transformation;
  fppi_mu = mu_q0[2] + dot_product(Q, Q_beta_point[2]) + linear_transformation;
  palb_mu = mu_q0[3] + dot_product(Q, Q_beta_point[3]) + linear_transformation;
  
  // Probability of reaching the concentrations limits
  {
    real steepness = 100;

    softbound_sum = inv_logit((pk_mu - Y_lower_trans[1]) * steepness);
    softbound_sum += inv_logit((Y_upper_trans[1] - pk_mu) * steepness);
    
    softbound_sum += inv_logit((fppi_mu - Y_lower_trans[2]) * steepness);
    softbound_sum += inv_logit((Y_upper_trans[2] - fppi_mu) * steepness);
    
    softbound_sum += inv_logit((palb_mu - Y_lower_trans[3]) * steepness);
    softbound_sum += inv_logit((Y_upper_trans[3] - palb_mu) * steepness);
  
    in_concentration_range = exponential_lpdf(2*responses - softbound_sum | 100);
  }
  
  // Preference is published as parameter for visualization
  // - diet_preference returns lpdf_exponential()
  preference = diet_preference(Q, current_Q, r, preference_strength);
}

model {

  for (i in 1:r) {
    target += uniform_lpdf(Q[i] | proposal_lowerlimits[i], proposal_upperlimits[i]);
    //target += normal_lpdf(Q[i] | general_RI[i], 10*(proposal_upperlimits[i] - proposal_lowerlimits[i]));
  }
  
  {
    //real pk_center = (Y_upper_trans[1] + Y_lower_trans[1]) / 2;
    //real fppi_center = (Y_upper_trans[2] + Y_lower_trans[2]) / 2;
    //real palb_center = (Y_upper_trans[3] + Y_lower_trans[3]) / 2;
    //real sigma = 10;

    //target += normal_lpdf(pk_mu | pk_center, (Y_upper_trans[1]-Y_lower_trans[1]));
    //target += normal_lpdf(fppi_mu | fppi_center, (Y_upper_trans[2]-Y_lower_trans[2]));
    //target += normal_lpdf(palb_mu | palb_center, (Y_upper_trans[3]-Y_lower_trans[3]));

    // probability of being inside the concentration normal ranges

    target += in_concentration_range;
    
    // diet preference conditionally on being inside the concentration limits

    target += preference;
  }
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