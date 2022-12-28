functions {
   
   real utility(real a, real b) {
    
     return b*1.5 + a*0.82;
   }
}

data {
}

parameters {
  real<lower=0, upper=100> Q1;
  real<lower=0, upper=200> Q2;
}

transformed parameters {
  real pk_mu = Q1 * 1.23 + Q2 * 0.78;
}

model {

  pk_mu ~ cauchy(3.4, 1);
  pk_mu ~ cauchy(4.7, 1);

}
