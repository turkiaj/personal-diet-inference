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
    
    real soft_limit_coef;
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

  vector[r] Q;
}

transformed parameters {

  real pk_mu;
  real fppi_mu;
  real palb_mu;

  pk_mu = mu_q0[1] + dot_product(Q, Q_beta_point[1]) + linear_transformation;
  fppi_mu = mu_q0[2] + dot_product(Q, Q_beta_point[2]) + linear_transformation;
  palb_mu = mu_q0[3] + dot_product(Q, Q_beta_point[3]) + linear_transformation;
}

model {
  
  // priors
  
  for (i in 1:r) {
      Q[i] ~ uniform(proposal_lowerlimits[i],proposal_upperlimits[i]);
  }
  
  {
    //real pk; 
    //real fppi; 
    //real palb;

    // pk ~ uniform(Y_lower_trans[1],Y_upper_trans[1]);
    // fppi ~ uniform(Y_lower_trans[2],Y_upper_trans[2]);
    // palb ~ uniform(Y_lower_trans[3],Y_upper_trans[3]);

    real pk_beta = alpha_point[1] / pk_mu;
    real fppi_beta = alpha_point[2] / fppi_mu;
    real palb_beta = alpha_point[3] / palb_mu;

    //real concentration_lp = 0;
    //real exceeding = 0;
    //real scaledown = 1;
    
    //pk ~ gamma(alpha_point[1], pk_beta);
    //fppi ~ gamma(alpha_point[2], fppi_beta);
    //palb ~ gamma(alpha_point[3], palb_beta);

    target += gamma_lupdf(pk | alpha_point[1], pk_beta);
    target += gamma_lupdf(fppi | alpha_point[2], fppi_beta);
    target += gamma_lupdf(palb | alpha_point[3], palb_beta);

    //target += concentration_lp;
    target += utility(Q, general_RI, r);

    // accumulate the amount of that estimated concentrations are over their minimum or maximun limits
    // if (pk < Y_lower_trans[1]) {exceeding += Y_lower_trans[1] - pk;}
    // if (pk > Y_upper_trans[1]) {exceeding += pk - Y_upper_trans[1];}
    // if (fppi < Y_lower_trans[2]) {exceeding += Y_lower_trans[2] - fppi;}
    // if (fppi > Y_upper_trans[2]) {exceeding += fppi - Y_upper_trans[2];}
    // if (palb < Y_lower_trans[3]) {exceeding += Y_lower_trans[3] - palb;}
    // if (palb > Y_upper_trans[3]) {exceeding += palb - Y_upper_trans[3];}
    // 
    // print("exceeding: ", exceeding);
    // 
    // // scale down the log probability by the amount of (possible) limit exceeding
    // if (exceeding > 0)
    // {
    //   // add scaled down lp increment instead
    //   scaledown = inv(exceeding);
    //   
    //   //print("full concentration_lp: ", concentration_lp);
    // 
    //   concentration_lp = concentration_lp * scaledown;
    // 
    //   //print("scaled concentration_lp: ", concentration_lp);
    //   
    //   target += concentration_lp;
    //   
    //   //print("diet proposal produces invalid concentrations, scaledown: ", scaledown);
    // }
    // else {
    //   
    //   target += concentration_lp;
    // 
    //   //print("diet proposal produces valid concentrations");
    //   
    //   // select the preferred diet from the valid concentrations
    //   target += utility(Q, general_RI, r);
    // }
    
    //print("target: ", target());
    
  }
  
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

