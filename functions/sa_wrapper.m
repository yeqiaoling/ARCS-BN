function [gamma, lambda, Bsa, topo_sort_pi] = sa_wrapper(X, varargin)
%
% SA: each step of SA is exact solution, which samples a permutation to
%     minimize objective function; 
% SA inputs: 
%       alpha: none zero ---> CD initialization by model selection with alpha =0.1
%              0 ---> CD initialization with the best model (lambda)
%       min_prop, lambda_num: in BIC path, inidicating solution path (min
%                             lambda and # of lambda)
%       OTHER SA PARAMETERS: MOST IMPORTANT ONE is 
%              Gini: 0 --> CD initialization; 2 --> GES initialization; 
%                    others ---> random initialization
%       SAVESHD: 1 --->  save SHD (not 100% accurate) as outputs
%                others ---> save (B, Omega) as outputs
%    
%% inputs
parser = inputParser;
parser.KeepUnmatched = true;
% initials
addOptional(parser, 'coef0', 0) 
addOptional(parser, 'Pini', 0) 
% bic
addOptional(parser, 'min_prop', 1e-1) 
addOptional(parser, 'lambda_num', 20) 
addOptional(parser, 'c', 1)
% SA parameters
addOptional(parser, 'T_max', 1);
addOptional(parser, 'TMAX', 1e2) 
addOptional(parser, 'HIGHTEMP', 0) 
addOptional(parser, 'TOL',  1e-2) 
addOptional(parser, 'N', 1e4);
addOptional(parser, 'k', 4);
% refinement
addOptional(parser, 'alpha', 1e-5);
% intervention
addOptional(parser, 'X_ind', 0)

parse(parser, varargin{:});
% initials
coef0 = parser.Results.coef0;
Pini = parser.Results.Pini;
% bic
min_prop = parser.Results.min_prop;
lambda_num = parser.Results.lambda_num;
c = parser.Results.c;
% SA parameters
N =  parser.Results.N;
T_max = parser.Results.T_max;
TMAX =  parser.Results.TMAX;
HIGHTEMP = parser.Results.HIGHTEMP;
TOL =  parser.Results.TOL;
k =  parser.Results.k;
% refinement
alpha = parser.Results.alpha;
% intervention
X_ind = parser.Results.X_ind;

%% setup & handle functions
truncat_coef = @(Bsa)  Bsa .* (abs(Bsa) >  0.1) ;
[~,p] = size(X);    
I = eye(p); 
INTV = (sum(X_ind)~=0);
%% SA
% initialization
if (sum(coef0) ~= 0) 
    % given an initial coeficient
    topo_sort = toposort(digraph(coef0)); % dfs
    inv_sort = flip(topo_sort); 
    Pini = I(inv_sort,:);
elseif (sum(Pini) == 0)
    % random initialization
    Pini = I(randperm(p),:);
    HIGHTEMP = 1;
end
[p_1, p_2] = size(Pini);
if (p_1 ~= p_2)
    Pini = I(Pini,:);
end
% select tuning parameters
if (INTV)
    [ ~, gamma, lambda] = bic_sel_intv(X, X_ind, Pini*[1:p]', ...
        'min_prop', min_prop, 'lambda_num', lambda_num, 'c', c);
    tic;
    [fval, pi_sa, Bsa]= sa_update_intv(X, X_ind, Pini*[1:p]', gamma, lambda, ...
        'TOL', TOL, 'TMAX', TMAX, 'N', N, 'HIGHTEMP', HIGHTEMP, 'k', k, 'T_max', T_max);
    isatime = toc;
else
    [~, gamma, lambda] = bic_sel(X, Pini*[1:p]', 'min_prop', min_prop, ...
        'lambda_num', lambda_num, 'c', c);
    tic;
    % run SA
    [fval, Psa, Lsa] = sa_update(X, Pini, gamma, lambda, ...
        'TOL', TOL, 'TMAX', TMAX, 'N', N, 'HIGHTEMP', HIGHTEMP, 'k', k, 'T_max', T_max);
    isatime = toc;
    pi_sa = Psa*[1:p]';
end
%% refine coeficent
if (INTV)
    [Bsa, ~] = convert_PL_to_BO(Bsa, 1:p);
else
    [Bsa, ~] = convert_PL_to_BO(Lsa, Psa *[1:p]');
end
Bsa = truncat_coef(Bsa); 
refine_coef(X, Bsa, 'X_ind', X_ind, 'alpha', alpha);
% save    
topo_sort_pi = flip(pi_sa);
filename = 'topological_sort.txt';
save(filename, 'topo_sort_pi', '-ascii')
