functions {

// credit http://srmart.in/informative-priors-for-correlation-matrices-an-easy-approach/
  vector lower_tri(matrix mat) {
    int d = rows(mat);
    int lower_tri_d = d * (d - 1) / 2;
    vector[lower_tri_d] lower;
    int count = 1;
    for(r in 2:d) {
      for(c in 1:(r - 1)) {
	lower[count] = mat[r,c];
	count += 1;
      }
    }
    return(lower);
 }

real lkj_corr_cholesky_point_lower_tri_lpdf(matrix cor_L, vector point_mu_lower, vector point_scale_lower) {
    real lpdf = lkj_corr_cholesky_lpdf(cor_L | 1);
    int d = rows(cor_L);
    matrix[d,d] cor = multiply_lower_tri_self_transpose(cor_L);
    lpdf += normal_lpdf(lower_tri(cor) | point_mu_lower, point_scale_lower);
    return(lpdf);
 }
}

data{
  int<lower = 1> nt;
  int<lower = 1> nObs;
  int<lower = 1> nSubjects;
  int nIIV;
  int<lower = 1> iObs[nObs];
  int<lower = 1> start[nSubjects];
  int<lower = 1> end[nSubjects];
  int<lower = 1> cmt[nt];
  int evid[nt];
  int addl[nt];
  int ss[nt];
  real amt[nt];
  real time[nt];
  real rate[nt];
  real ii[nt];
  real weight[nt];
  real age[nt];
  vector<lower = 0>[nObs] cObs;
  int<lower = 0, upper = 1> runestimation; //   a switch to evaluate the likelihood

}

transformed data{
  vector[nObs] logCObs = log(cObs);
  int nTheta = 5;
  int nCmt = 3;
  int nti[nSubjects];
  real biovar[nCmt];
  real tlag[nCmt];
  vector[6] point_mu_lower = [0.538,-0.036,0.323,-0.110,-0.199,-0.144]';
  vector[6] point_scale_lower = [0.25,0.25,0.25,0.25,0.25,0.25]';

  for(i in 1:nSubjects) nti[i] = end[i] - start[i] + 1;

  for (i in 1:nCmt) {
    biovar[i] = 1;
    tlag[i] = 0;
 }
}
parameters{
  real<lower = 0, upper = 500> CLHat;
  real<lower = 0, upper = 500> QHat;
  real<lower = 0, upper = 3500> V1Hat;
  real<lower = 0, upper = 3500> V2Hat;
  real<lower = 0, upper = 5> betaHat;
  real<lower = 0, upper = 100> TclHat;
  real<lower = 0> allo[4];
  real<lower = 0> fr[4];
  real<lower = 0> sigma;
  real<lower = 3>  nu; // normality constant

// Inter-Individual variability
  vector<lower = 0.01, upper = 2>[nIIV] omega;
  matrix[nIIV, nSubjects] etaStd;
  cholesky_factor_corr[nIIV] L;
}

transformed parameters{
  vector<lower = 0>[nIIV] thetaHat;
  matrix<lower = 0>[nSubjects, nIIV] thetaM; // variable required for Matt's trick
  real<lower = 0> theta[nTheta];
  matrix<lower = 0>[nCmt, nt] x;
  row_vector<lower = 0>[nt] cHat;
  row_vector<lower = 0>[nObs] cHatObs;

  thetaHat[1] = CLHat;
  thetaHat[2] = QHat;
  thetaHat[3] = V1Hat;
  thetaHat[4] = V2Hat;

 // Matt's trick to use unit scale 
  thetaM = (rep_matrix(thetaHat, nSubjects) .* exp(diag_pre_multiply(omega, L * etaStd)))'; 
  
  for(j in 1:nSubjects)
  {

    theta[1] = thetaM[j, 1] * fr[1] * (weight[start[j]] / 70)^allo[1] * (1-(1-betaHat)*exp(-(4*age[start[j]])*0.693/TclHat)); // CL
    theta[2] = thetaM[j, 2] * fr[2] * (weight[start[j]] / 70)^allo[2]; // Q
    theta[3] = thetaM[j, 3] * fr[3] * (weight[start[j]] / 70)^allo[3]; // V1
    theta[4] = thetaM[j, 4] * fr[4] * (weight[start[j]] / 70)^allo[4]; // V2
    theta[5] = 0; // ka

    x[,start[j]:end[j]] = pmx_solve_twocpt(time[start[j]:end[j]], 
                                       amt[start[j]:end[j]],
                                       rate[start[j]:end[j]],
                                       ii[start[j]:end[j]],
                                       evid[start[j]:end[j]],
                                       cmt[start[j]:end[j]],
                                       addl[start[j]:end[j]],
                                       ss[start[j]:end[j]],
                                       theta, biovar, tlag);
                                       
    cHat[start[j]:end[j]] = x[2,start[j]:end[j]] ./ theta[3]; // divide by V1
  }

  cHatObs  = cHat[iObs];
}

model{
  //Informative Priors
  CLHat ~ lognormal(log(14.6),0.06);
  QHat  ~ lognormal(log(157),0.111);
  V1Hat ~ lognormal(log(62.5),0.17);
  V2Hat ~ lognormal(log(119),0.087);
  betaHat ~ lognormal(log(0.262),0.274);
  TclHat ~ lognormal(log(25.7),0.294);
  
  fr ~ lognormal(log(1),0.25);

  L~lkj_corr_cholesky_point_lower_tri(point_mu_lower, point_scale_lower);

  nu ~ gamma(2,0.1);

 // Inter-individual variability (see transformed parameters block
 // for translation to PK parameters)

  to_vector(etaStd) ~ normal(0, 1);

  omega[1] ~ lognormal(log(0.351),0.25);
  omega[2] ~ lognormal(log(0.773),0.25);
  omega[3] ~ lognormal(log(0.711),0.25);
  omega[4] ~ lognormal(log(0.229),0.25); 

  allo[1] ~ normal(0.75,0.125);
  allo[2] ~ normal(0.75,0.125);
  allo[3] ~ normal(1,0.125);
  allo[4] ~ normal(1,0.125);

  sigma ~ lognormal(log(0.10), 0.25);

  if(runestimation==1){
    logCObs ~ student_t(nu,log(cHatObs), sigma);
  }
}

generated quantities{
    real cObsCond[nObs];
    real log_lik[nObs];
    row_vector[nt] cHatPred;
    real cObsPred[nObs];
    row_vector<lower = 0>[nObs] cHatObsPred;
    matrix[nCmt,nt] xPred;
    matrix[nIIV, nSubjects] etaStdPred;
    matrix<lower=0>[nSubjects, nIIV] thetaPredM;
    corr_matrix[nIIV] rho;
    real<lower = 0> thetaPred[nTheta];

    rho = L * L';

    for(i in 1:nSubjects){
      for(j in 1:nIIV){ 
        etaStdPred[j, i] = normal_rng(0, 1);
      }
    }

    thetaPredM = (rep_matrix(thetaHat, nSubjects) .* exp(diag_pre_multiply(omega, L * etaStdPred)))';

    for(j in 1:nSubjects){

      thetaPred[1] = thetaPredM[j,1]* (weight[start[j]] / 70)^allo[1] * (1-(1-betaHat)*exp(-(4*age[start[j]])*0.693/TclHat)); // CL
      thetaPred[2] = thetaPredM[j,2]* (weight[start[j]] / 70)^allo[2]; // Q 
      thetaPred[3] = thetaPredM[j,3]* (weight[start[j]] / 70)^allo[3]; // V1
      thetaPred[4] = thetaPredM[j,4]* (weight[start[j]] / 70)^allo[4]; // V2
      thetaPred[5] = 0; // ka 
    
      xPred[,start[j]:end[j]] = pmx_solve_twocpt(time[start[j]:end[j]],
                                              amt[start[j]:end[j]],
                                              rate[start[j]:end[j]],
                                              ii[start[j]:end[j]],
                                              evid[start[j]:end[j]],
                                              cmt[start[j]:end[j]],
                                              addl[start[j]:end[j]],
                                              ss[start[j]:end[j]],
                                              thetaPred, biovar, tlag);

     cHatPred[start[j]:end[j]] = xPred[2,start[j]:end[j]] ./ thetaPred[3];
  }
  
cHatObsPred = cHatPred[iObs];

  for(i in 1:nObs){
      cObsCond[i] = exp(student_t_rng(nu,log(fmax(machine_precision(),cHatObs[i])), sigma)); // individual predictions
      cObsPred[i] = exp(student_t_rng(nu,log(fmax(machine_precision(),cHatObsPred[i])), sigma)); // population predictions
      log_lik[i] = student_t_lpdf(cObs[i] | nu, log(fmax(machine_precision(),cHatObs[i])), sigma);  
  }
}
