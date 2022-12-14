
data {
    int<lower=1> responses;    // number of responses
    int<lower=1> p;            // number of predictors
    int<lower=0, upper=p> r;   // number of conditioned predictors
    real proposal_limits[r*2]; // min and max limits of proposals    real proposal_limits[r*2]; // min and max limits of proposals
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
    int condition_in_repeat;
    vector[r] Q_index;
}

transformed data {

  // Estimated value of concentration without the queried predictors (mu_q0)
  real mu_q0[responses];

  // linear transformation needs to be applied to limits also
  real Y_lower_trans[responses];        
  real Y_upper_trans[responses];
  
  // In X_evidence[p,:]
  // - factors are indicated with -1 in second column and their value 0/1 is at first column
  // - for gaussian variables colums give mean and sd

  for (m in 1:responses) {
    mu_q0[m] = intercept_point[m] + dot_product(X_evidence_point, X_beta_point[m]);

    //mu_q0[m] = intercept_point[m] + dot_product(X_evidence[1:p,1], X_beta_point[m]);
    
    // if Q is 0 outside the concentration limits then expand the sampling there

    //Y_lower_trans[m] = Y_lower_limits[m] + linear_transformation;
    //Y_upper_trans[m] = Y_upper_limits[m] + linear_transformation;
    
    // tällä systeemillä pk:lle samplataan liian pieniä arvoja kun sen alaraja on muq0
    // ehkä rajat pitäisi kuitenkin olla kuten yllä ja Q muunnettu niin että samplays pääsee alkuun
    // tarkista tuloksista että eikö pk oikeasti voi olla isompi?
    
    if (Y_lower_limits[m] < mu_q0[m]-0.1) {
      // Q = 0 point is above the lower concentration limit and it is a valid proposal
      Y_lower_trans[m] = Y_lower_limits[m] + linear_transformation;
    }
    else
    {
      // Q needs to be over 0 for reaching the concentration limit
      Y_lower_trans[m] = mu_q0[m] - 0.1 + linear_transformation;
    }

    if (Y_upper_limits[m] > mu_q0[m]+0.1) {
      // there is room between upper limit and Q = 0 point
      Y_upper_trans[m] = Y_upper_limits[m] + linear_transformation;
    }
    else
    {
      // Q should lower mu_q0 in order to reach the recommendations
      Y_upper_trans[m] = mu_q0[m] + 0.1 + linear_transformation;
    }
  }
}

parameters {
  
  real<lower=linear_transformation> pk;
  real<lower=linear_transformation> fppi;
  real<lower=linear_transformation> palb;

  vector[r] Q_trans;
}

transformed parameters {
  
  real<lower=Y_lower_trans[1],upper=Y_upper_trans[1]> pk_mu;
  real<lower=Y_lower_trans[2],upper=Y_upper_trans[2]> fppi_mu;
  real<lower=Y_lower_trans[3],upper=Y_upper_trans[3]> palb_mu;
  
  vector[r] Q;
  
  // these can be alo negative if mu_q0 is greater than the limit midpoint
  //real pk_q_add = (Y_lower_limits[1] + (Y_upper_limits[1] - Y_lower_limits[1]) / 2) - mu_q0[1]; 
  //real fppi_q_add = (Y_lower_limits[1] + (Y_upper_limits[1] - Y_lower_limits[1]) / 2) - mu_q0[1]; 
  //real palb_q_add = (Y_lower_limits[1] + (Y_upper_limits[1] - Y_lower_limits[1]) / 2) - mu_q0[1]; 
   
  // real q_add_half = ((pk_q_add + fppi_q_add + palb_q_add) / 3) / 2;
  // 
  Q = Q_trans * 0.05;
  
  if (repeat_only != 1) {
    pk_mu = mu_q0[1] + dot_product(Q, Q_beta_point[1]) + linear_transformation;
    fppi_mu = mu_q0[2] + dot_product(Q, Q_beta_point[2]) + linear_transformation;
    palb_mu = mu_q0[3] + dot_product(Q, Q_beta_point[3]) + linear_transformation;
  }
  else
  {
    pk_mu = mu_q0[1] + linear_transformation;
    fppi_mu = mu_q0[2] + linear_transformation;
    palb_mu = mu_q0[3] + linear_transformation;
  }
}

model {

  // allowed limits for recommended nutrients
  //(for normalized pot/pho these are -2.793127  2.304855 -2.482934  3.844332)
  
  Q[1] ~ uniform(proposal_limits[1],proposal_limits[2]);
  Q[2] ~ uniform(proposal_limits[3],proposal_limits[4]);

  //Q[1] ~ lognormal(-0.244136,10);
  //Q[2] ~ lognormal(0.680699,10);
  
  // Estimate the queried variables Q
  pk ~ gamma(alpha_point[1], alpha_point[1] / pk_mu);
  fppi ~ gamma(alpha_point[2], alpha_point[2] / fppi_mu);
  palb ~ gamma(alpha_point[3], alpha_point[3] / palb_mu);

  // print("mu limits");
  // print("pk:   ", Y_lower_trans[1], " ", Y_upper_trans[1]);
  // print("fppi: ", Y_lower_trans[2], " ", Y_upper_trans[2]);
  // print("palb: ", Y_lower_trans[3], " ", Y_upper_trans[3]);
  // 
  // print("estimated mu_q0");
  // print("pk:   ", mu_q0[1]);
  // print("fppi: ", mu_q0[2]);
  // print("palb: ", mu_q0[3]);
  //  
  // print("estimated mu");
  // print("pk:   ", pk_mu);
  // print("fppi: ", fppi_mu);
  // print("palb: ", palb_mu);
}

generated quantities {
  
  real concentration[posterior_samples, responses];
  real mu_q0_pred[responses];
  real mu_pred[responses];
  vector[r] Q_pred;

  {
  vector[p] X;  // posteriors for unmodified nutrients

  // sample X from evidence
  for (i in 1:p)
  {
    // - gaussian or factor?
    if (X_evidence[i,2] != -1) {

      if (condition_in_repeat == 1 && (i == Q_index[1] || i == Q_index[2])) {
        
          // This allows comparing acceptance sampling with NUTS
          if (i == Q_index[1]) {
            X[i] = uniform_rng(proposal_limits[1],proposal_limits[2]);
            Q_pred[1] = X[i];
          } else
          if (i == Q_index[2]) {
            X[i] = uniform_rng(proposal_limits[3],proposal_limits[4]);
            Q_pred[2] = X[i];
          }
      } 
      else
      {
        // in this case, Q is estimated with NUTS model
        Q_pred[1] = Q[1];
        Q_pred[2] = Q[2];
        
        // X_sd_multiplier allows simulating smaller uncertainty of unmodified diet
        if (X_sd_coef > 0)
        {
          X[i] = normal_rng(X_evidence[i,1], X_evidence[i,2] * X_sd_coef);
        }
        else
        {
          X[i] = X_evidence_point[i];
        }
      }
    } else {
      X[i] = X_evidence_point[i];
      //X[i] = X_evidence[i,1];
    }
  }

  // same sample X is used for all the responses in this proposal

  for (m in 1:responses) {
    
    mu_q0_pred[m] = intercept_point[m] + dot_product(X, X_beta_point[m]);

    mu_pred[m] = mu_q0_pred[m];
    
    if (repeat_only != 1) {
      mu_pred[m] += dot_product(Q_pred, Q_beta_point[m]);
    }

    for (po in 1:posterior_samples)
    {
      concentration[po, m] = gamma_rng(alpha_point[m], alpha_point[m] / (mu_pred[m] + linear_transformation)) - linear_transformation;
    }

  } // responses
  }
  
}
