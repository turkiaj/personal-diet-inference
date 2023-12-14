functions {
  
  // This function defines a custom probability distribution that receives 
  // its maximum value when the current diet proposal is closest to most preferred diet
  real preference_error(vector Q, vector RI, vector beta, int r) {

    // This function prefers such a personal diet proposal that is closest to person's current diet
    // and it is achieved by minimizing the absolute sum of squares between personal recommendations (Q) and current personal levels of RI
    
    vector[r] diffs = Q - RI;
    
    // Multiply difference elementwise with inverse of personal effect. This prefers changes in nutrients having stronger effect. 
    // So, nutrients with stronger effects are allowed to be changed more.
    vector[r] weighted_diffs = (1 - fabs(beta)) .* diffs;
    
    //real sum_of_errors = sum(fabs(diffs));
    real sum_of_errors = sum(fabs(weighted_diffs));

    // scale the error sum with number of predictors 
    return sum_of_errors / r;
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
    vector[p] X_beta_point[responses];   //  strength coefficients of personal effects for fixed level nutrients 
    vector[r] Q_beta_point[responses];   //  strength coefficients of personal effects for inferred nutrient levels
    
    // sufficient statistics of nutrient variables X
    vector[p] X_evidence_point;
    
    real linear_transformation;

    real bound_steepness;     // l1: steepness of single soft bound
    real bound_requirement;   // l2: steepness of meeting all the bounds
    real preference_strength; // l3: preference requirement
    real mix; // mixing proportion of preference into in_concentration_range  
}

transformed data {

  // Estimated value of concentration without the queried predictors (mu_q0)
  // This part of the expected value (mu) does not change during the sampling
  real mu_q0[responses];

  for (m in 1:responses) {
    mu_q0[m] = intercept_point[m] + dot_product(X_evidence_point, X_beta_point[m]);
  } 
}

parameters {
 
  // Resulting parameters of this inference are these personal recommendations for intake proposals Q
  // Each nutrient Q is given lower and upperlimits for its recommendation
  vector<lower=proposal_lowerlimits, upper=proposal_upperlimits>[r] Q;
  
  //real<lower=0> l3;
}

transformed parameters {

  real in_concentration_range;
  real in_range;
  real preference;
  real Y_mu[responses];
  real Y_bound[responses*2];
  real softbound_sum = 0;
  real preference_error_sum = 0;

  // Probability of reaching the concentrations limits
  for (m in 1:responses) {
    Y_mu[m] = mu_q0[m] + dot_product(Q, Q_beta_point[m]);
  
    // lower bound sigmoid    
    Y_bound[2*m-1] = inv_logit((Y_mu[m] - Y_lower_limits[m]) * bound_steepness);
    
    // upper bound sigmoid
    Y_bound[2*m] = inv_logit((Y_upper_limits[m] - Y_mu[m]) * bound_steepness);
    
    softbound_sum += Y_bound[2*m-1] + Y_bound[2*m];
    preference_error_sum += preference_error(Q, current_Q, Q_beta_point[m], r);
  }
  
  // make preference_error_sum invariant to number of concentrations (responses)
  preference_error_sum = preference_error_sum / responses;
  
  //in_concentration_range = exponential_lpdf(2*responses - softbound_sum | bound_requirement);
  in_concentration_range = normal_lpdf(softbound_sum | 2*responses, 1/bound_requirement);
  
  // Pitääkö olla -1 tai +1? Pitääkö arvon mennä yli nollan että ässä taipuu?
  in_range = inv_logit((softbound_sum - 2*responses - 1) * 20);
  
  //in_concentration_range_cdf = exponential_cdf(2*responses - softbound_sum, 1/bound_requirement);
  
  //preference = exponential_lpdf(preference_error_sum | preference_strength);
  preference = normal_lpdf(preference_error_sum | 0, 1/preference_strength);
}

model {
  
  //real Y_range;
  
  // PRIORS
  
  // nutrient proposals are taken from uniform distribution
  for (i in 1:r) {
    target += uniform_lpdf(Q[i] | proposal_lowerlimits[i], proposal_upperlimits[i]);
  }
  
  //target += normal_lpdf(l3 | 5, 10);

  // center of concentration normal ranges can be slightly preferred
  //for (m in 1:responses) {
  //   Y_range = Y_upper_limits[m]-Y_lower_limits[m];
  //   target += normal_lpdf(Y_mu[m] | Y_lower_limits[m] + Y_range/2, Y_range*2);
  //}
  
  // POSTERIOR
  // Diet preference and concentration requirements are assumed to be independent.
  // This allows us to sum their log densities without considering their joint density.
  
  target += in_concentration_range;
  target += in_range * preference;

  //target += log_mix(in_range, preference, in_concentration_range);

}

