function [bic, gamma, lambda] = bic_sel(X, P, varargin)
%
% objective: given a permutation, use BIC to select a lambda & gamma 
%           in MCP penalty. At each (gamma, lambda), use regularized 
%           regression to estimate the loss by the proximal gradient 
%           algorithm. 
%           lambda: max --> zero coef; min --> 0,1 * max 
%           gamma: {2, 10, 50, 100}
% inputs: X - design matrix
%         P - an initial order (s.t. L is lower triagular matrix)
% output: bic - minimium bic value
%         (gamma, lambda) - a pair of parameter with smallest BIC, 
%
% BIC = c ln(max(n,p))*k + 2 * neg-log-likelihood
%       where c is a constant with default value of 
%             k is the # of estimates (including variance estimate)
%
%% parameters
parser = inputParser;
parser.KeepUnmatched = true;
addOptional(parser, 'min_prop', 1e-1) 
addOptional(parser, 'lambda_num', 20) 
addOptional(parser, 'c', 1) 
% getL fn
addOptional(parser, 'beta', 0.8) 
addOptional(parser, 'TMAX', 1e2) 
addOptional(parser, 'TOL', 1e-2)

parse(parser, varargin{:});
min_prop =  parser.Results.min_prop;
lambda_num =  parser.Results.lambda_num;
c =  parser.Results.c;
% getL fn
beta =  parser.Results.beta;
TMAX =  parser.Results.TMAX;
TOL =  parser.Results.TOL;
%% 
% handle functions
% truncat_coef = @(Bsa)  Bsa .* (abs(Bsa)>  0.1) ;

[n, p] = size(X);
lambda_max = sqrt(n);
lambda_min = lambda_max * min_prop;
lambda_rec = lambda_min:(lambda_max-lambda_min)/(lambda_num-1):lambda_max;
gamma_rec = [2, 10, 50, 100];
gamma_num = length(gamma_rec);

% bic
Pmat = eye(p); Pmat = Pmat(P,:);
bic_rec = zeros(gamma_num, lambda_num);
for j = 1:gamma_num  
    gamma = gamma_rec(j);
    for i = 1:lambda_num
        lambda = lambda_rec(i);
        [ gval, ~, lres] = getLpg_Lsubmat_v2(X, Pmat, gamma, lambda, ones(p), 1:p, ...
            'beta', beta, 'TMAX', TMAX, 'TOL', TOL);
        % [B, ~] = convert_PL_to_BO(lres, P);
        % B = truncat_coef(B); 
        % bic_rec(i) = log(max(n,p)) * nnz(B) + 2 * gval;
        bic_rec(j, i) = c*log(max(n,p)) * nnz(lres) + 2 * gval;
    end
end

[bic, ~] = min(bic_rec(:));
[gamma_ind, lambda_ind] = find(bic_rec == bic);
gamma = gamma_rec(gamma_ind);
lambda = lambda_rec(lambda_ind);

