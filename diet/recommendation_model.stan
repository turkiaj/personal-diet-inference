functions {
  // Function calculates the error in dietary preference by minimizing the weighted differences between the proposed and a given reference diet
  // The function emphasizes changes in nutrients with stronger effects, aiming for minimal overall diet change
  real preference_error(row_vector Q, row_vector reference_Q, row_vector[] beta, int nutrients, int responses, real penalty) {

    real error_sum = 0;
    row_vector[nutrients] Q_diffs;

    for (m in 1:responses) {
      for (i in 1:nutrients) {
        
        // Calculate difference between the proposed Q and given reference level 
        if (size(reference_Q) == size(Q)) {
          Q_diffs[i] = reference_Q[i] != 0 ? fabs(Q[i] - reference_Q[i]) ./ fabs(reference_Q[i]) : fabs(Q[i] - reference_Q[i]);
        } else {
          // In calculating maximum error, we compare to both lower and upper limits. 
          real max_diff = fmax(fabs(reference_Q[i] - Q[i]), fabs(reference_Q[nutrients+i] - Q[i]));
          Q_diffs[i] = Q[i] != 0 ? max_diff / fabs(Q[i]) : max_diff;
        }
      }

      // Weights the differences using the inverse of nutrient effects to prioritize changes in nutrients with more significant impacts
      row_vector[nutrients] inverse_beta = 1 ./ (1+fabs(beta[m]));
      real weighted_diffs = dot_product(inverse_beta, Q_diffs);
      
      // Penalize bigger differences with power function
      error_sum += pow(weighted_diffs / nutrients, penalty);
    }

    // Normalize the sum of penalized preference errors by the number of responses
    return error_sum / responses;
  }
}

data {
    int<lower=1> responses;              // Total number of responses (concentrations)
    int<lower=1> p;                      // Number of predictors
    int<lower=0> r;                      // Number of nutrients considered
    row_vector[r] proposal_lowerlimits;  // Lower intake limits for each nutrien
    row_vector[r] proposal_upperlimits;  // Upper intake limits for each nutrient   
    row_vector[r] general_RI;            // General intake recommendations
    row_vector[r] current_Q;             // Current nutrient intake levels
    real Y_lower_limits[responses];      // Lower concentration limits
    real Y_upper_limits[responses];      // Upper concentration limits

    // Personal nutritional effects
    real intercept_point[responses];       // Baseline intercepts from parameter distributions
    row_vector[p] X_beta_point[responses]; // Personal effect coefficients for fixed-level nutrients
    row_vector[r] Q_beta_point[responses]; // Personal effect coefficients for variable nutrient levels
    row_vector[p] X_evidence_point;
    
    // Inference hyperparameters
    real limit_steepness;                // l1: Steepness parameter for individual soft bounds
    real limit_requirement;              // l2: Requirement for meeting all bounds
    real preference_strength;            // l3: Weighting for preference in the model
    real transition_steepness;           // l4: Steepness for the transition between model components
    real penalty_rate;                   // l5: Rate at which preference errors are penalized
    real general_recommendation_prior;   // l6: Weight for general recommendation in proposal distributions
    
    int verbose;                         // Flag for verbose output
}

transformed data {

  // Calculate personal maximum preference error by combining the current diet to lower and upper intake limits  
  real max_personal_preference_error = preference_error(current_Q, append_col(proposal_lowerlimits, proposal_upperlimits), Q_beta_point, r, responses, penalty_rate);
  
  real lambda_k_min = -lambert_w0(-machine_precision()*max_personal_preference_error) / max_personal_preference_error;

  // Scaling with max_personal_preference_error make preference_strength ignorant of nutrient and constraint number
  real personal_preference_strength = lambda_k_min + preference_strength / max_personal_preference_error;

  if (verbose > 0) {
    print("lambda_k_min: ", lambda_k_min);
    print("max_personal_preference_error: ", max_personal_preference_error);
    print("personal_preference_strength: ", personal_preference_strength);
  }

  // Calculates the expected concentration value excluding queried predictors, remaining constant during sampling
  real mu_q0[responses];
  
  for (m in 1:responses) {
    mu_q0[m] = intercept_point[m] + dot_product(X_evidence_point, X_beta_point[m]);
  }
}

parameters {
  // Resulting personalized nutrient intake recommendations
  row_vector<lower=proposal_lowerlimits, upper=proposal_upperlimits>[r] Q;
}

transformed parameters {

  real Y_mu[responses];             // Expected concentration levels for proposed diets
  real Y_mu_Q0[responses] = mu_q0;  // Concentration level when only the fixed factors are considered and all of Q are 0
  real Y_limit[responses*2];        // Sigmoid values indicating when concentration limits are met
  real softlimit_sum = 0;           // Sum of sigmoid values, with a maximum of 2*responses

  real within_limits_lpdf;          // Log probability of meeting all concentration limits
  real preference_lpdf;             // Log probability of adherence to personal dietary preferences
  real preference_error_sum = 0;    // Sum of dietary preference errors
  real in_range;                    // Indicates whether all concentration limits have been met
  
  row_vector[r] Q_contributions[responses];        // Normalized contribution of nutrient on concentrations 
  
  // Computes expected values and adherence to concentration limits for each response
  for (m in 1:responses) {
    // Expected concentrations for diet proposal Q
    Q_contributions[m] = Q_beta_point[m] .* Q;
    Y_mu[m] = mu_q0[m] + sum(Q_contributions[m]);
  
    // Sigmoid values for the expected concentrations relative to the limits
    Y_limit[2*m-1] = inv_logit((Y_mu[m] - Y_lower_limits[m]) * limit_steepness);
    Y_limit[2*m] = inv_logit((Y_upper_limits[m] - Y_mu[m]) * limit_steepness);
    softlimit_sum += Y_limit[2*m-1] + Y_limit[2*m];
  }

  // Log probabilities of mixture distribution components
  within_limits_lpdf = normal_lpdf(softlimit_sum | 2*responses, 1/limit_requirement);

  preference_error_sum = preference_error(Q, current_Q, Q_beta_point, r, responses, penalty_rate);
  preference_lpdf = exponential_lpdf(preference_error_sum | personal_preference_strength);

  // Sigmoid coefficient for transitioning between model components
  // - subtracting the constant 0.1 adjusts the sigmoid to reach maximum when all limits are met  
  in_range = inv_logit((softlimit_sum - (2*responses - 0.1)) * transition_steepness);
  
  if (verbose > 1) {
    print("preference_lpdf: ", preference_lpdf);
  }
}

model {
  
  // PRIORS: The inferred nutrients are given healthy prior limits from the general recommendations, 
  // with a mixture distribution adjusting the preference of the generally recommended intake
  
  for (i in 1:r) {
    real pref_sigma = (proposal_upperlimits[i] - proposal_lowerlimits[i]) / general_recommendation_prior;
    target += log_mix(0.5,
                    uniform_lpdf(Q[i] | proposal_lowerlimits[i], proposal_upperlimits[i]),
                    normal_lpdf(Q[i] | general_RI[i], pref_sigma));
  }
  
  // LIKEHOOD: Mixture model that first aims to reach the concentration targets and then follows the diet preference
  target += log((1-in_range) * exp(within_limits_lpdf));
  target += log(in_range * exp(preference_lpdf));
}
