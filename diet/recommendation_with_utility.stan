functions {
  
  // This function defines a custom probability distribution that receives 
  // its maximum value when the current diet proposal is closest to most preferred diet
  real preference_error(vector Q_diffs, vector RI, vector beta, int r) {
    
    // This function prefers such a personal diet proposal that is closest to person's current diet
    // and it is achieved by minimizing the absolute sum of squares between personal recommendations (Q) and current personal levels of RI
    
    // Multiply difference elementwise with inverse of personal effect. This prefers changes in nutrients having stronger effect. 
    // So, nutrients with stronger effects are allowed to change more

    vector[r] inverse_beta = 1 ./ (1+fabs(beta));
    
    // Q_diffs = fabs(Q - current_Q) is pre-calculated as a model parameter for analysis
    real weighted_diffs = dot_product(inverse_beta, Q_diffs);

    // resulting error is scaled with the number of predictors
    return weighted_diffs / r;
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

    real bound_steepness;      // l1: steepness of single soft bound
    real bound_requirement;    // l2: steepness of meeting all the bounds
    real preference_strength;  // l3: preference requirement (estimated, remvoe?)
    real transition_steepness; // l4: transition steepness between mixture components (is in_range necessary, remove?)
    real penalty_rate;         // l5: Exponential penalty_rate of preference error
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
  
  //real<lower=0> personal_preference_strength;
}

transformed parameters {

  real in_concentration_range_lpdf;
  real preference_lpdf;
  real in_range;
  real Y_mu[responses];
  real Y_bound[responses*2];
  real softbound_sum = 0;
  real preference_error_sum = 0;
  vector[r] Q_effects[responses];
  vector[r] Q_diffs; 
  
  // Probability of reaching the concentrations limits
  for (m in 1:responses) {
    Y_mu[m] = mu_q0[m] + dot_product(Q, Q_beta_point[m]);
  
    // lower bound sigmoid    
    Y_bound[2*m-1] = inv_logit((Y_mu[m] - Y_lower_limits[m]) * bound_steepness);
    
    // upper bound sigmoid
    Y_bound[2*m] = inv_logit((Y_upper_limits[m] - Y_mu[m]) * bound_steepness);
    
    softbound_sum += Y_bound[2*m-1] + Y_bound[2*m];
    
    //Q_diffs = fabs(Q - current_Q);
    
    for (i in 1:r) {
        // Calculate relative differences
        if (current_Q[i] != 0) {
            Q_diffs[i] = fabs(Q[i] - current_Q[i]) / fabs(current_Q[i]);
        } else {
            // One approach is to default to the absolute difference or use a small non-zero value
            Q_diffs[i] = fabs(Q[i] - current_Q[i]);
        }
    }    
    
    preference_error_sum += preference_error(Q_diffs, current_Q, Q_beta_point[m], r);
    
    Q_effects[m] = Q_beta_point[m] .* Q;
  }

  // transition coefficient (from 0 to 1) when all the concentration bounds are met
  // - epsilon 0.1 defines an allowed gap to maximum boundsum so that sigmoid is pushed to 1
  in_range = inv_logit((softbound_sum - (2*responses - 0.1)) * transition_steepness);
  
  // normal distribution is used for concentration bounds as we require that all bounds are fulfilled
  in_concentration_range_lpdf = normal_lpdf(softbound_sum | 2*responses, 1/bound_requirement);

  // make preference_error_sum invariant to number of concentrations (responses)
  // also penalize bigger preference errors with power growth
  preference_error_sum = pow(preference_error_sum / responses, penalty_rate);

  // exponential distribution is used for preferences as the strenght is estimated personally and can vary
  // - square and exponential growth of error penalizes more large errors
  preference_lpdf = exponential_lpdf(preference_error_sum | preference_strength);

}

model {
  
  //real Y_range;
  
  // PRIORS
  
  // nutrient proposals are taken from uniform distribution
  for (i in 1:r) {
    target += uniform_lpdf(Q[i] | proposal_lowerlimits[i], proposal_upperlimits[i]);
  }
  
  //arget += normal_lpdf(personal_preference_strength | 5, 10);

  // center of concentration normal ranges can be slightly preferred
  //for (m in 1:responses) {
  //   Y_range = Y_upper_limits[m]-Y_lower_limits[m];
  //   target += normal_lpdf(Y_mu[m] | Y_lower_limits[m] + Y_range/2, Y_range*2);
  //}
  
  // POSTERIOR

  // in_range-coefficient provides steep but smooth transition between the mixture components
  // it also ensures that preference is not considered before all the concentration limits are satisfied

  // Random variables are dependent through in_range-coefficient and thus multiplied
  target += log((1-in_range) * exp(in_concentration_range_lpdf));
  target += log(in_range * exp(preference_lpdf));
  
}

// generated quantities {
  
// Sensitivity analysis of recommendations
// Seek how much the recommendations can be altered before the concentrations go beyond their limits
  
//   Y_mu_pred[m] = mu_q0[m] + dot_product(Q, Q_beta_point[m]);
// }



