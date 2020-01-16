function [ gval, fval, lres] = getLpg_Lsubmat_v2(X, P, gamma, lambda, lfull, vec, varargin)
%
% objective: given a permutation P, use proximal gradient algorithm find 
% a minimizer L (lower triangular matrix) 
%
% loss: negative log-likelihood with MCP penalty 
%
% inputs: X - design matrix
%         P - permutation matrix (s.t. L is lower triangular)
%         gamma - MCP penalty concavity
%         lambda - MCP penalty strength
%         lfull - initial L matrix (p*p) in the order of
%                   proposed_permutation, suppose propose an interval by 
%                   flipping columns k to m, then only update k to m 
%                   columns in 'lfull'
%         vec - which colomns to updates in 'lfull', i.e. k to m in the
%               previous example
%
% outputs: fval - minimum loss 
%          gval - minimum negative log-likelihood 
%          lres - minimizer
%
%% parameters
parser = inputParser;
parser.KeepUnmatched = true;
addOptional(parser,'beta', 0.8) 
addOptional(parser,'TMAX', 1e2) 
addOptional(parser,'TOL', 1e-2) 
% testing purpose
addOptional(parser,'SAVE', 0) 

parse(parser, varargin{:});
beta =  parser.Results.beta;
TMAX =  parser.Results.TMAX;
TOL =  parser.Results.TOL;
% testing purpose
SAVE =  parser.Results.SAVE;

%%
[n, p] = size(X);
Sigh = X'*X/n;

MCP = @(x) (lambda*abs(x)- x.^2/(2*gamma)) .* (abs(x)<=lambda*gamma) + lambda^2*gamma/2 * (abs(x)>lambda*gamma); % lambda>=0, gamma>1
Sigma = P*Sigh*P'; 
g = @(L, Diagl) trace(Sigma * L * L')*0.5*n - sum(log(diag(Diagl)))*n;

l = lfull(:,vec); 
iter = 0; t = max(1, norm(l,'fro')); err = Inf;  

% record
if (SAVE) 
    t_rec = zeros(1,TMAX);
    [lnm_rec, fval_rec, err_rec, normt_rec] = deal(t_rec);
    l_rec = cell(TMAX,1);
end 

while iter < TMAX && err >= TOL;    
    iter = iter+1;
    lold = l; 
    % initialization for each prox. grad.
    T2 = 0;
    deril = dg_Lsubmat(Sigma, n, l, vec);
    lnm = norm(deril,'fro');
    % record
    if (SAVE)
        t_rec(iter) = t; lnm_rec(iter) = lnm; normt_rec(iter) = t/lnm; 
        l_rec{iter} = l;
    end
    while T2 <= TMAX % find step size
        T2 = T2+1;
        tl = l-t*deril/lnm;
        ZERO = zeros(p);
        ZERO(:,vec) = tl;
        zoff = prox_th_mcp(lambda, gamma, tril(ZERO,-1), t/lnm);
       
        z = zoff + diag(diag(ZERO));
        z = z(:,vec); 
        if (g(z, diagl_lsub(z, vec)) <= g(l, diagl_lsub(l, vec)) + reshape(deril,1,[]) * reshape((z-l),[],1) + norm(z-l, 'fro')^2*lnm/(2*t))  
            break
        else
            t = beta * t;
        end
    end % end of step size search
    l = z;
    % err = norm(l - lold,'fro')/max(1, norm(l,'fro'));
    err = max(vecnorm(l - lold)./max(1, norm(l,'fro')));
    % fprintf('ONE STEP IN PROX. GRAD.: iters: %d, step: %2.2f, error: %1.2e.\n', T2, t, err/max(1, norm(l,'fro')));
    % record
    if (SAVE) 
        lres = lfull; lres(:,vec) = l;
        fval_rec(iter) = g(lres, diag(diag(lres))) + sum(sum(MCP(tril(lres,-1))));
        err_rec(iter) = err;  
    end
end
lres = lfull;
lres(:,vec) = l;
gval = g(lres, diag(diag(lres)));
fval =  gval + sum(sum(MCP(tril(lres,-1))));

% record
if (SAVE)
    t_rec = nonzeros(t_rec);
    lnm_rec = nonzeros(lnm_rec);
    fval_rec = nonzeros(fval_rec);
    err_rec = nonzeros(err_rec);
    normt_rec = normt_rec(1:length(t_rec));
    l_rec = l_rec{1:length(t_rec)};
    save('getL_pg.mat', 't_rec', 'lnm_rec', 'fval_rec', 'err_rec', 'normt_rec', 'l_rec')
end

% fprintf('FULL PROX. GRAD. Time: %d, obj: %2.2e, step: %2.2e, error: %2.2e, \n', T, fval, t, err);
% fprintf('g(l): %d, mcp: %2.2e\n', g(l) , MCP(tril(l,-1)));



