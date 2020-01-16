function [fval, Psa, Lsa] = sa_update(X, p0, gamma, lambda, varargin)
%
% object: use the simulated annealing to sample permutation for minimizing  
%           a loss function, which is negative log-likelihood + MCP 
%           penalty (with parameter gamma & lambda).
% detail: Each P propose is flipping an interval of a permutation of length 
%          'k', then calculate the loss for the proposed one and accept it
%          with the Metropolis rate. 
% note: SA update is performed on corresponding columns of coefficient
%       matrix (instead of the whole matrix). Because of that, each
%       update in the SA is a submatrix of a lower triangular matrix.
% penalty: MCP penalty with gamma & lambda (penalty parameters)
% 
% explicit inputs: X - data matrix
%                  p0 - initial permuation s.t. L is lower triangular
%                  lambda - penalty function parameter
%                  gamma - penalty parameter 
%                  coef_true - the true coefficient matrix
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
% in function 'getLpg_Lsubmat_v2'
addOptional(parser,'beta',0.8) 
addOptional(parser,'TMAX',1e2) 
addOptional(parser,'TOL', 1e-2) 
% flag
addOptional(parser,'FLAG',0)

parse(parser, varargin{:});
% SA
T_min =  parser.Results.T_min;
T_max =  parser.Results.T_max;
N =  parser.Results.N;
step =  parser.Results.step;
HIGHTEMP =  parser.Results.HIGHTEMP;
k =  parser.Results.k;
% in function 'getLpg_Lsubmat_v2'
beta = parser.Results.beta;
TMAX = parser.Results.TMAX;
TOL = parser.Results.TOL;
% flag
FLAG = parser.Results.FLAG;

%% start the algorithm
rng(1);
% handle functions
accProb = @(Edel, T) min(1,exp(-Edel/T));
% set up
[~,p] = size(X);
I = eye(p);
if (HIGHTEMP)
    T_max = 20;
end 
if (FLAG)
    fprintf('TEMPERATURE \n ');
    fprintf('highest: %1.1e \n ', T_max);
    fprintf('lowest: %1.1e \n', T_min);
end

% initialization
l = ones(p);
[ ~, old_cost, l] = getLpg_Lsubmat_v2(X, p0, gamma, lambda, l, 1:p, 'beta', beta, 'TMAX', max(1e3, TMAX), 'TOL', min(1e-3,TOL) ); 
% temperature
T = T_max; Tmax = 0;
while (Tmax <= 1)
    Tmax = N/(log(T_min) - log(T_max))*log(step); % # of iters per temperature
    if (Tmax > 1)
        break
    end
    step = 0.9*step;
end

% SA
iter = 0; fval = zeros(1, N);
c = p0*[1:p].'; % start permutation

% start
tic;
while T > T_min
    %fprintf('temperature: %1d\n',T);
    i = 1;
    while i <= Tmax
        start_pos = randsample((p-k+1),1); 
        swap_intv = start_pos:(start_pos+k-1);
        % proposed permutation
        t = c;  t(swap_intv,1) = flip(c(swap_intv,1));
        csort = 1:p;  csort(swap_intv) = flip(csort(swap_intv));
        pp = I(t,:);
            
        % update 
        [  ~, new_cost, new_l] = getLpg_Lsubmat_v2(X, pp, gamma, lambda, l(csort, csort), swap_intv, 'beta', beta, 'TMAX', TMAX, 'TOL', TOL);
        ap = accProb(new_cost-old_cost, T);
        ac = (ap > rand(1));
        if(ac)
            p0 = pp; 
            c = t;
            old_cost = new_cost;
            l = new_l;
        end
        i = i+1;
        iter = iter + 1;
        fval(iter) = old_cost;
    end
    T = T*step;
end

SA_time = toc;
fval = nonzeros(fval);
Psa = p0;

if (FLAG)
    fprintf('SA: objective value: %1.1f using %d iterations and %d seconds.\n', fval(end), length(fval), SA_time);
end
Lsa = l;
