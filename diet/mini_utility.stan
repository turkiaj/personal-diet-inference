functions {
   
   real utility(real a, real b) {
     return b-a;
   }
}

data{
}

parameters{
  real<lower=-20> a;
  real<upper=40> b;
  real<lower=-1000, upper=1000> pk;
  real pk_mu;
}

model{
  
  target += uniform_lpdf(pk_mu | a,b);
  target += normal_lpdf(pk | pk_mu, 1);
  target += utility(a,b);
}
