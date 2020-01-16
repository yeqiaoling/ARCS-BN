function [fval, pi_sa, Lsa, iniP_fval, iniP_l] = ...
    sa_update_intv(X, X_ind, ini_pi, gamma, lambda, varargin)
%
% keywords: obs/intv data, SA, proximal gradient aglorithm, columnwise
%           updates, initial topological sort
%
% object: use the simulated annealing to sample permutation for minimizing  
%           a loss function, which is negative log-likelihood + MCP 
%           penalty (with parameter gamma & lambda).
% detail: Each P propose is flipping an interval of a permutation of length 
%          'k', then calculate the loss for the proposed one and accept it
%          with the Metropolis rate. 
% note: SA update is performed on corresponding columns of coefficient
%       matrix (instead of the whole matrix), and such udpate is performed
%       by the proximal gradient algorithm with columnwise updatas (instead
%       of whole matrix update)
% penalty: MCP penalty with gamma & lambda (penalty parameters)
% 
% explicit inputs: X - data matrix, n*p
%                  X_ind - intervention indicator matrix, n*p
%                  ini_pi - initial permuation s.t. B is comptible with
%                  lambda - penalty function parameter
%                  gamma - penalty parameter 
% inexplicit inputs: SA parameters & update parameters (stoping criteria) & save
%                       indicators (SAVE, FLAG etc)
%                   SAVE: 1 ---> save 
% 

%% inputs
parser = inputParser;
parser.KeepUnmatched = true;
% SA
addOptional(parser,'T_min',1e-1);
addOptional(parser,'T_max',1);
addOptional(parser,'N',1e4);
addOptional(parser,'step',0.999)
addOptional(parser,'HIGHTEMP',0)
addOptional(parser,'k',4)
% in function of getLpg_Lsubmat
addOptional(parser,'beta',0.8) 
addOptional(parser,'TMAX',1e2) 
addOptional(parser,'TOL', 1e-2) 
% flags & save
addOptional(parser,'choose_coef_mode',0)
addOptional(parser,'Gini',1)
addOptional(parser,'FLAG',0)
addOptional(parser,'SAVE',0)

parse(parser, varargin{:});
% SA
T_min =  parser.Results.T_min;
T_max =  parser.Results.T_max;
N =  parser.Results.N;
step =  parser.Results.step;
HIGHTEMP =  parser.Results.HIGHTEMP;
k =  parser.Results.k;
% in function of getLpg_Lsubmat
beta = parser.Results.beta;
TMAX = parser.Results.TMAX;
TOL = parser.Results.TOL;
% flags 
FLAG = parser.Results.FLAG;

%% parameters
% handle functions
accProb = @(Edel, T) min(1,exp(-Edel/T));

rng(1);
[~,p] = size(X); 
if (HIGHTEMP)
    T_max = 20;
end 
if (FLAG)
    fprintf('TEMPERATURE \n ');
    fprintf('highest: %1.1e \n ', T_max);
    fprintf('lowest: %1.1e \n', T_min);
end

% initial fval
l = ones(p);
[ ~,  iniP_fval, iniP_l] = getB_givenP_col_v2(X, X_ind, ini_pi, gamma, lambda, l, 1:p, ...
    'beta', beta, 'TMAX', max(1e3, TMAX), 'TOL', min(1e-3,TOL) );
% initialization
old_cost = iniP_fval; l = iniP_l;

%% SA 
% temperature
T = T_max; Tmax = 0;
while (Tmax <= 1)
    Tmax = N/(log(T_min) - log(T_max))*log(step); % # of iters per temperature
    if (Tmax > 1)
        break
    end
    step = 0.9*step;
end

% start
iter = 0; fval = zeros(1, N); c = ini_pi; 
tic;
while T > T_min
    % fprintf('temperature: %1d\n',T);
    i = 1;
    while i <= Tmax
        start_pos = randsample((p-k+1),1); 
        swap_intv = start_pos:(start_pos+k-1);
        % proposed permutation
        t = c;  t(swap_intv) = flip(c(swap_intv));
        % update 
        [  ~, new_cost, new_l] = getB_givenP_col_v2(X, X_ind, t, gamma, ...
            lambda, l, c(swap_intv), 'beta', beta, 'TMAX', TMAX, 'TOL', TOL );
        ap = accProb(new_cost-old_cost, T);
        ac = (ap > rand(1));
        if(ac)
            c = t;
            old_cost = new_cost;
            l = new_l;
        end
        i = i+1;
        iter = iter + 1;
        fval(iter) = old_cost;
    end
    T = T*step;
    if(FLAG)
        fprintf(' temp: %d, iter: %d, accept: %d, fval: %1.4e\n', T, floor(iter/100)*100, ac, old_cost);
    end
end

SA_time = toc;
fval = nonzeros(fval);
pi_sa = c;

%fprintf('ISA: min fval: %1.1f, iter %d.\n', fval(end), length(fval));
Lsa = l;
% [  ~, ~, Lsa] = getB_givenP_col_v2(X, X_ind, Psa, gamma, lammcp, I, 1:p, 'beta', beta, 'TMAX', 1e2, 'TOL', TOL);
