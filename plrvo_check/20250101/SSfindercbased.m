clear all;
clc;
% Parameters
tasks = {'sst-2', 'mnli', 'qnli', 'qqp', 'e2e', 'dart', 'cifar10', 'mnist', 'p100'};
sample_rate = [0.015204383138576668, 0.0026075685696315028, 0.009776309634056691, ...
               0.0028143697763070743, 0.024345593304961843, 0.010988657216135298, ...
               0.02048, 0.017066666666666667, 0.1024];
steps = [197, 1150, 306, 1065, 410, 910, 488, 585, 97];
target_delta = [7.4240152043831385e-06, 1.2732268406403822e-06, 4.773588688504244e-06, ...
                1.3742039923374386e-06, 1.18874967309384e-05, 5.365555281316063e-06, ...
                1e-05, 8.333333333333334e-06, 5e-05];

% Initialize a directory for saving CSV files
output_dir = 'output_csv';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end




for i=1:9 

%Gaussian comprehensive
gaussian_results = [];  % Store Gaussian results

 sigmad=linspace(0.5,10,2000);



lambda_max = 200;

 T_values=steps(i)
 delta_values=target_delta(i)

q_values=sample_rate(i)
clip_values=linspace(0.1,5, 10);
for idx_T=1:length(T_values)
T=T_values(idx_T)
    for idx_q = 1:length(q_values)
     q=q_values(idx_q) 
        for idx_delta=1:length(delta_values)
    delta=delta_values(idx_delta)

    % 
            for idx_clip =  1:length(clip_values)
                 clip = clip_values(idx_clip);
         for idx_sig=1:length(sigmad)
            sigma=sigmad(idx_sig);
             eps_check_gaussian =inf;
            % Calculate sigma based on image formula
                     % Loop over lambda up to lambda_max
            for lambda = 1:lambda_max
                % Compute epsilon for Gaussian noise

                % term1 = T * lambda / (2 * (sigma)^2);
  lmbda1=lambda+1;              
             % lmbda1=lambda      
log_factorials = [0, cumsum(log(1:lmbda1))]; 

% Initialize the summation

                  
          % Initialize the summation
S = 0;
log_factorials = [0, cumsum(log(1:lmbda1))]; % Include log(0!) = 0 at the start

% Initialize the summation


% Loop through all k values
for kk = 0:lmbda1
    % Compute binomial coefficient using logarithms
    
    % Compute the term
    
    % Compute binomial coefficient using logarithms
    log_binom_coeff = log_factorials(lmbda1 + 1) - log_factorials(kk + 1) - log_factorials( lmbda1 - kk + 1);
    
    binom_coeff = exp(log_binom_coeff); % Convert back from log scale

    % Compute the term
    exp_term = exp((kk^2 - kk) / (2 * sigma^2));
    term = binom_coeff * (1 - q)^( lmbda1 - kk)* q^kk* exp_term;

    % Accumulate the result
    S = S + term;
end
S=log(S)/(lambda);
eps_check_gaussian1 = (T*S)+ log((lambda) / (lambda+1)) - ((log(delta) + log(lambda+1)) / (lambda));
                    
                % Calculate eps_check_gaussian and check if it meets eps_target
                % eps_check_gaussian = epsilon_gaussian(lambda) + (log(lambda / (1 + lambda)) - log(delta) + log(1 + lambda)) / lambda;
                % 
              if eps_check_gaussian1>0
               if eps_check_gaussian1 <= eps_check_gaussian 
                   eps_check_gaussian=eps_check_gaussian1;
                
                    % Calculate distortion for Gaussian
                  distortion_gaussian = sigma * sqrt(2 / pi)*clip;
                    % if distortion_gaussian<distmin
                        % distmin=distortion_gaussian;
                    % Store the best Gaussian config for each eps_target
                        gaussian_eps = eps_check_gaussian;
                           gaussian_dist=distortion_gaussian;
                          gaussian_clip=clip;
                          gaussian_q=q;
                          gaussian_sigma=sigma;
                   
                % end
            end
        end
            end
            gaussian_results = [gaussian_results; gaussian_eps, gaussian_q, gaussian_sigma, clip, gaussian_dist/clip];
                          
            end
            end
            
     end
    end
 end




%%SSfuinder1 (noaccount)-------------
alph=linspace(1,2,30);
lambda_max=linspace(50,200, 4);
theta_log_values = logspace(log10(0.001), log10(0.005), 100);
theta_lin_values = linspace(0.0001,0.0051,100);
theta_values_test= [theta_log_values, theta_lin_values];
k_log_values = logspace(log10(1.001), log10(50),100);
k_lin_values = linspace(40,5000,300);
% k_lin_values1 = linspace(200,2000,300);
k_values_test = [k_lin_values, k_log_values];

 C_lamb=[];
for j=1:length(lambda_max)
    lambdaa=lambda_max(j)
  
    for t=1:length(clip_values)
        c=clip_values(t);
         for idx_k =1:length(k_values_test)
            % for idx_k = 1:678
            idx_k;
            k = k_values_test(idx_k);
        maxtheta1=1/(lambdaa*c);
         % maxtheta2=(c)/(k);
         % maxtheta3=sqrt(pi/2)/(c*(k-1));
        % maxtheta=min(maxtheta1,maxtheta3);
        mintheta=(pi*k)/(2*c^3*(k-1)^2);
        if mintheta<maxtheta1
         theta_values_test= linspace(0.1*mintheta,maxtheta1,100);
        
            for idx_theta =1:length(theta_values_test)
                theta = theta_values_test(idx_theta);
               
                ff=sqrt(pi/2)/(c*theta*(k-1));
                if ff>1 
                    x=c/ff^2;
              y=gamcdf(x,k,theta);
                if y>0.4
dist=1/((k-1)*theta);
if dist<10
% DD=alph(i)*(pi/2);
% GG=nthroot(DD,3)
% C=sqrt(pi/2)/(GG*lambdaa)-0.01;
C_lamb=[C_lamb; c, lambdaa, ff, k, theta];
                end
                end
                end
            end
         end
    end
end
end

%%SSfinder2:accountpertask-comapreto gauss
%-----------------------


less_config = C_lamb;
theta_values =less_config(:,5);
k_values=less_config(:,4);
clip_values=less_config(:,1);
% k_values=40
% theta_values=0.00380952380952381
Q=[]


for idx_T=1:length(T_values)
   
T=T_values(idx_T)
for idx_q = 1:length(q_values)
     q=q_values(idx_q) 
for idx_delta=1:length(delta_values)
    delta=delta_values(idx_delta)

    Qn=[];
    
    for idx_clip =  1:length(clip_values)
       lambda_max=less_config(idx_clip,2);
        clip = clip_values(idx_clip);
        C=clip;
 % Replace with your desired tolerance
                    matches1 =gaussian_results(:,4) == clip;
                    gaussian_resultsc = gaussian_results(matches1,:);
      
      
            
            k = k_values(idx_clip);

            
                theta = theta_values(idx_clip);
                  distortion_PLRV = 1/((k-1)*theta);
                    mean_PLRV=k*theta;                % Loop over lambda up to lambda_max
                eps_check=inf;
                %accounting
                bestlambda=1;
               
                
                for lambda = 1:lambda_max
         lmbda1=lambda+1;              
            
log_factorials = [0, cumsum(log(1:lmbda1))]; 


M = 0;



    

  for kk = 0: lmbda1
      if kk<2
          rhs=1;
      end
      if kk>1
            x = kk *C;
    y= (kk-1)*C;
    A=0.5*(1-y* theta)^(-k);
    B=(2*kk)-1;
    B=((-0.5*(1+x* theta)^(-k))+(0.5*(1-y* theta)^(-k)))/B;
    Cp=0.5*(1 + x * theta)^(-k);
   rhs=A+B+Cp;
      end

    log_binom_coeff = log_factorials(lmbda1 + 1) - log_factorials(kk + 1) - log_factorials( lmbda1 - kk + 1);
    binom_coeff = exp(log_binom_coeff); 

 
    term = binom_coeff * (1 - q)^( lmbda1 - kk)* q^kk* (rhs);

        M = M + term;
   
end

M=log(M)/lambda;
      M=T * M;             
                    if isnan(M) || isinf(M) || M < 0
                        continue;
                    end
                    
                 
                    eps_check1 =M + log(lambda / (1 + lambda)) - ((log(delta) + log(1 + lambda)) / (lambda));
                    
                    
                    if eps_check1 > 20 || eps_check1 <= 0
                        continue;
                    end
                     if eps_check1<eps_check 
                       eps_check=eps_check1;
                       bestlambda=lambda;
                       kmax=kk*clip;
                    end
                   
                   
                    
                end
                if imag(eps_check) == 0 && eps_check<10
                    tolerance = max(eps_check/50,0.01);
                    matches = abs(gaussian_resultsc(:,1) - eps_check) <= tolerance;
                        
                    matching_entries = max(gaussian_resultsc(matches,5));
                   
                        if distortion_PLRV< matching_entries

                Qn = [Qn; eps_check, distortion_PLRV/clip, delta, clip, q, k, theta, T, lambda_max matching_entries/clip];
                      
                    end
                end
        end
          
          Qn=sortrows(Qn, 1); 
          Q=[Q;Qn];
       
         
    end

  end
end
%%%----Laplace
Qlap=[];
epsilonlap=linspace(0.1,10,50);
k_values_test = linspace(4000,10000,50);
clip_values=linspace(0.1,5, 10);
for idx_eps=1:50
    eps=epsilonlap(idx_eps);
    for idx_clip=1:10
        C=clip_values(idx_clip);
        for idx_k=1:50
            k=k_values_test(idx_k);
            theta= C/(eps*k);

             distortion_lap = 1/((k-1)*theta);
                                 % Loop over lambda up to lambda_max
                eps_check=inf;
                %accounting
                bestlambda=1;
                bound=floor(((eps*k)/(C^2)))-1; 
                bound=min(200,bound);
                for lambda =bound
                    lmbda1=lambda+1;              
            
log_factorials = [0, cumsum(log(1:lmbda1))]; 


M = 0;



    

  for kk = 0: lmbda1
      if kk<2
          rhs=1;
      end
      if kk>1
            x = kk *C;
    y= (kk-1)*C;
    A=0.5*(1-y* theta)^(-k);
    B=(2*kk)-1;
    B=((-0.5*(1+x* theta)^(-k))+(0.5*(1-y* theta)^(-k)))/B;
    Cp=0.5*(1 + x * theta)^(-k);
   rhs=A+B+Cp;
      end

    log_binom_coeff = log_factorials(lmbda1 + 1) - log_factorials(kk + 1) - log_factorials(lmbda1 - kk + 1);
    binom_coeff = exp(log_binom_coeff); 

 
    term = binom_coeff * (1 - q)^( lmbda1 - kk)* q^kk* (rhs);

        M = M + term;
   
end

M=log(M)/lambda;
      M=T * M;             
                    if isnan(M) || isinf(M) || M < 0
                        continue;
                    end
                    
                 
                    eps_check1 =M + log(lambda / (1 + lambda)) - ((log(delta) + log(1 + lambda)) / (lambda));
                     if imag(eps_check1) == 0 && eps_check1<10
                    
                    if eps_check1 > 20 || eps_check1 <= 0
                        continue;
                    end
                     if eps_check1<eps_check 
                       eps_check=eps_check1;
                       bestlambda=lambda;
                       kmax=kk*clip;
                    end
                   
                   
                    
                end
             
                    
                end
                if eps_check<10
                   
           

                Qlap = [Qlap; eps_check, distortion_lap/C, delta, C, q, k, theta, T, lambda_max C/eps_check];
                end   
                    end
                end
end
    % Generate filename
        % Prepare filenames for each result
    gaussian_filename = fullfile(output_dir, [tasks{i}, '_gaussian_results.csv']);
    Qn_filename = fullfile(output_dir, [tasks{i}, '_Qn.csv']);
    Qlap_filename = fullfile(output_dir, [tasks{i}, '_Qlap.csv']);
    
    % Save each result to its respective CSV file
    writematrix(gaussian_results, gaussian_filename);
    writematrix(Qn, Qn_filename);
    writematrix(Qlap, Qlap_filename);
end