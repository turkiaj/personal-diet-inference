functions {
  
  // This function defines a custom probability distribution that receives 
  // its maximum value when the current diet proposal is closest to most preferred diet
  real preference_error(vector Q_diffs, vector beta, int r) {
    
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
  
  // Function calculates maximum preference error given the data of proposed intake limits and personal reactions (beta) 
  real max_preference_error(vector[] Q_beta_point, vector current_Q, vector proposal_lowerlimits, vector proposal_upperlimits, int r, int responses, real penalty_rate) {

    vector[r] Q_max_diffs;
    real preference_error_sum = 0;

    for (m in 1:responses) {

      for (i in 1:r) {
        // Calculate maximum relative differences
        if (current_Q[i] != 0) {
            Q_max_diffs[i] = fmax(fabs(proposal_lowerlimits[i] - current_Q[i]), fabs(proposal_upperlimits[i] - current_Q[i])) / fabs(current_Q[i]);
        } else {
            Q_max_diffs[i] = fmax(fabs(proposal_lowerlimits[i] - current_Q[i]), fabs(proposal_upperlimits[i] - current_Q[i]));
        }
      }

      preference_error_sum += preference_error(Q_max_diffs, Q_beta_point[m], r);
    }

    preference_error_sum = pow(preference_error_sum / responses, penalty_rate);

    return preference_error_sum;
  }
  
}

data {
    int<lower=1> responses;        // number of responses
    int<lower=1> p;                // number of predictors
    int<lower=0> r;                // number of conditioned predictors
    vector[r] proposal_lowerlimits; // lower limits of proposed nutrient intake   
    vector[r] proposal_upperlimits; // upper limits of proposed nutrient intake    
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
    real preference_strength;  // l3: preference requirement 
    real transition_steepness; // l4: transition steepness between mixture components 
    real penalty_rate;         // l5: Exponential penalty_rate of preference error
    
    int verbose;
}

transformed data {

  // This is theoritical maximum error with current intake, personal effects and limits
  real max_personal_preference_error = max_preference_error(Q_beta_point, current_Q, proposal_lowerlimits, proposal_upperlimits, r, responses, penalty_rate);

  // Calculate preference strength inversely non-linearly related to max_personal_preference_error
  // Here, 0.01 is baseline constant to prevent division by zero and set a baseline strength and
  // preference_strength parameter is an adjustment factor to scale the impact of max_personal_preference_error

  real personal_preference_strength = 1 / (0.01 + preference_strength * pow(max_personal_preference_error, penalty_rate));

  if (verbose == 1) {
    print("max_personal_preference_error: ", max_personal_preference_error);
    print("personal_preference_strength: ", personal_preference_strength);
  }

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
    
    for (i in 1:r) {
        // Calculate relative differences
        if (current_Q[i] != 0) {
            Q_diffs[i] = fabs(Q[i] - current_Q[i]) / fabs(current_Q[i]);
        } else {
            Q_diffs[i] = fabs(Q[i] - current_Q[i]);
        }
    }    
    
    preference_error_sum += preference_error(Q_diffs, Q_beta_point[m], r);
    
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
  
  preference_lpdf = exponential_lpdf(preference_error_sum | personal_preference_strength);
  
  if (verbose == 1) {
    print(preference_error_sum);
  }
}

model {
  
  // PRIORS
  
  // nutrient proposals are taken from uniform distribution
  for (i in 1:r) {
    target += uniform_lpdf(Q[i] | proposal_lowerlimits[i], proposal_upperlimits[i]);
  }
  
  // POSTERIOR

  // in_range-coefficient provides steep but smooth transition between the mixture components
  // it also ensures that preference is not considered before all the concentration limits are satisfied

  // Random variables are dependent through in_range-coefficient and thus multiplied
  target += log((1-in_range) * exp(in_concentration_range_lpdf));
  target += log(in_range * exp(preference_lpdf));
}
