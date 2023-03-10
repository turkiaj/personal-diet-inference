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
    
    real bound_steepness;     // steepness of single soft bound
    real bound_requirement;   // steepness of meeting all the bounds
}

transformed data {

  // Estimated value of concentration without the queried predictors (mu_q0)
  // This part of the expected value (mu) does not change during the sampling
  real mu_q0[responses];

  //real Y_lower_trans[responses];
  //real Y_upper_trans[responses];      

  for (m in 1:responses) {
    mu_q0[m] = intercept_point[m] + dot_product(X_evidence_point, X_beta_point[m]);
  } 
}

parameters {
 
  // Resulting parameters of this inference are these personal recommendations for intake proposals Q
  // Each nutrient Q is given lower and upperlimits for its recommendation
  vector<lower=proposal_lowerlimits, upper=proposal_upperlimits>[r] Q;
}

transformed parameters {

  real in_concentration_range;
  real preference;
  real Y_mu[responses];
  real Y_bound[responses*2];
  real softbound_sum = 0;

  // Probability of reaching the concentrations limits
  for (m in 1:responses) {
    Y_mu[m] = mu_q0[m] + dot_product(Q, Q_beta_point[m]);
  
    // lower bound sigmoid    
    Y_bound[2*m-1] = inv_logit((Y_mu[m] - Y_lower_limits[m]) * bound_steepness);
    
    // upper bound sigmoid
    Y_bound[2*m] = inv_logit((Y_upper_limits[m] - Y_mu[m]) * bound_steepness);
    
    softbound_sum += Y_bound[2*m-1] + Y_bound[2*m];
  }

  // bound_requirement is the lambda of exponential distribution
  in_concentration_range = exponential_lpdf(2*responses - softbound_sum | bound_requirement);

  // Preference is published as parameter for visualization
  // - diet_preference returns lpdf_exponential()
  preference = diet_preference(Q, current_Q, r, preference_strength);
}

model {
  
  real Y_range;
  
  // PRIORS
  
  // nutrient proposals are taken from uniform distribution
  for (i in 1:r) {
    target += uniform_lpdf(Q[i] | proposal_lowerlimits[i], proposal_upperlimits[i]);
  }

  // center of concentration normal ranges is slightly preferred
  for (m in 1:responses) {
     Y_range = Y_upper_limits[m]-Y_lower_limits[m];
     target += normal_lpdf(Y_mu[m] | Y_range/2, Y_range*5);
  //   
  //   //target += uniform_lpdf(Y_mu[m] | Y_lower_limits[m], Y_upper_limits[m]);
  }
  
  // POSTERIOR
  
  // probability of being inside the concentration normal ranges
  target += in_concentration_range;
    
  // diet preference conditionally on being inside the concentration limits
  target += preference;
}

generated quantities {
  
  // real concentration[posterior_samples, responses];
  // real mu_pred[responses];
  // real mu_q0_pred[responses];
  // 
  // for (m in 1:responses) {
  // 
  //   mu_q0_pred[m] = intercept_point[m] + dot_product(X_evidence_point, X_beta_point[m]);
  // 
  //   mu_pred[m] = mu_q0_pred[m] + dot_product(Q, Q_beta_point[m]);
  // 
  //   for (po in 1:posterior_samples)
  //   {
  //     concentration[po, m] = gamma_rng(alpha_point[m], alpha_point[m] / (mu_pred[m] + linear_transformation)) - linear_transformation;
  //   }
  // }

}