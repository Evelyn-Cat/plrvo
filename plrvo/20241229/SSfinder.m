%Gaussian comprehensive
gaussian_results = [];  % Store Gaussian results

 sigmad=linspace(0.5,10,200);



lambda_max = 200;

 T_values=197
 delta_values=1/(2*67349);

q_values=1024/67349;
clip_values=linspace(1,5, 30);
for idx_T=1:length(T_values)
T=T_values(idx_T)
    for idx_q = 1:length(q_values)
     q=q_values(idx_q) 
        for idx_delta=1:length(delta_values)
    delta=delta_values(idx_delta)

    % 
            for idx_clip =  209597:length(clip_values)
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
            gaussian_results = [gaussian_results; gaussian_eps, gaussian_q, gaussian_sigma, clip, gaussian_dist];
                          
            end
            end
            
     end
    end
 end




%%SSfuinder1 (noaccount)-------------
alph=linspace(1,2,30);
lambda_max=linspace(50,150,5);
theta_log_values = logspace(log10(0.001), log10(0.005), 100);
theta_lin_values = linspace(0.0001,0.0051,100);
theta_values_test= [theta_log_values, theta_lin_values];
k_log_values = logspace(log10(1.001), log10(30),100);
k_lin_values = linspace(30,200,200);
% k_lin_values1 = linspace(200,2000,300);
k_values_test = [k_lin_values, k_log_values];

 C_lamb=[];
for j=1:length(lambda_max)
    lambdaa=lambda_max(j)
  
    for t=1:length(clip_values)
        c=clip_values(t);
         mintheta=1/(lambdaa*c);
theta_values_test= linspace(mintheta,3*mintheta,200);
         for idx_k =1:length(k_values_test)
            % for idx_k = 1:678
            idx_k;
            
            k = k_values_test(idx_k);
            for idx_theta =1:length(theta_values_test)
                theta = theta_values_test(idx_theta);
               
                ff=sqrt(pi/2)/(c*theta*(k-1));
                if ff>1 
                    x=c/ff^2;
              y=gamcdf(x,k,theta);
                if y>0.7
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


%%SSfinder2:accountpertask-comapreto gauss
%-----------------------


less_config = C_lamb;
theta_values =less_config(:,5);
k_values=less_config(:,4);
clip_values=less_config(:,1);
% k_values=40
% theta_values=0.00380952380952381
Q=[]

tolerance = 1e-6;
for idx_T=1:length(T_values)
   
T=T_values(idx_T)
for idx_q = 1:length(q_values)
     q=q_values(idx_q) 
for idx_delta=1:length(delta_values)
    delta=delta_values(idx_delta)

    Qn=[];
    
    for idx_clip = 1:length(clip_values)
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
                bound=50; 
                
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
                    
                    matches = abs(gaussian_resultsc(:,1) - eps_check) <= tolerance;
                    matching_entries = gaussian_resultsc(matches,5);
                    
                        if distortion_PLRV< matching_entries
                Qn = [Qn; eps_check, distortion_PLRV, delta, clip, q, k, theta, T, lambda_max matching_entries];
                      
                    end
                end
        end
       
         Qn=sortrows(Qn, 1); 
         Q=[Q;Qn];
       
         
    end

  end
end

  