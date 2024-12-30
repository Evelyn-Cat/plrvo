function SSS = SSS_Finder(C, array_alpha, func_G)
% SSS_FINDER: Generates superior search space for the PLRV-O mechanism.
% Inputs:
%   C: The clipping threshold (e.g., 0.1).
%   array_alpha: An array of accountant orders (e.g., 2:256).
%   func_G: function of PLRV-O accountant.
% Output:
%   SSS: A matrix where each row contains [k, theta] values satisfying the conditions.


%% S_range
S_range_k =  [logspace(log10(1.00001), log10(5), 50), linspace(5,100,200)];

SSS = [];
for k = S_range_k
    S_range_theta = linspace(1/(k-1), 2/(k-1), 200);
    
    for theta = S_range_theta
        valid = true;
        
        for eta = array_alpha
            lhs = exp(0.5 * (eta^2-eta) * C^2 * (k-1)^2 * theta^2);
            t1 = -eta*C;
            t2 = (eta-1)*C;

            if 1 - t2 * theta < 0 || 1 - t1 * theta < 0
                valid = false;
                break;
            end
            
            rhs = func_G(C, eta, k, theta);
            
            if lhs < rhs
                valid = false;
                break;
            end
        
        end
        
        if valid
            SSS = [SSS; k, theta];
        end

    end
end

end