function [ gval, fval, bres] = getB_givenP_col_v2(X, X_ind, pi, gamma, lambda, bini, vec, varargin)
%
% keywords: interventional/observation data, proximal gradiant algorithm, 
%           MCP penalty, column updates on B (original order)
%
% objective: given a permutation P, use proximal gradient algorithm to find 
%            a minimizer B (in the original order) by column updates
%
% loss: negative log-likelihood g(B) plus MCP penalty 
%       g(B) = n * tr(X'XBB') - n log(|B|)
%       with X being X_{Oj,pa_j} (dim: nj * pj)
%       where Oj is a set of observations that j is not under intervention, 
%             and pa_j is parent set of j, we have a similar form
% 
% idea: for each j (needs to be updated), take design matrix with rows that
%       j is not under intervention, then regress j on potential parents of
%       j (i.e. nodes ahead of j in permutation pi). 
% 
% inputs: X - design matrix
%         X_ind - an interventional indicator matrix (n*p)
%         pi - a permutation (s.t. B is compatible with)
%         gamma - MCP penalty concavity
%         lambda - MCP penalty strength
%         bini - initial coeficient matrix 
%         vec - which colomns to updates in 'lini'
%
% outputs: fval - minimum loss 
%          gval - minimum negative log-likelihood 
%          bres - minimizer (in the original order)
%
%% inputs
parser = inputParser;
parser.KeepUnmatched = true;
addOptional(parser,'beta', 0.8) 
addOptional(parser,'TMAX', 1e2) 
addOptional(parser,'TOL', 1e-2) 
% testing purpose
addOptional(parser,'SAVE', 0) 
addOptional(parser,'FLAG', 0) 

parse(parser, varargin{:});
beta =  parser.Results.beta;
TMAX =  parser.Results.TMAX;
TOL =  parser.Results.TOL;
% testing purpose
SAVE =  parser.Results.SAVE;
FLAG = parser.Results.FLAG;

%% functions
MCP = @(x) (lambda*abs(x)- x.^2/(2*gamma)) .* (abs(x)<=lambda*gamma) + lambda^2*gamma/2 * (abs(x)>lambda*gamma); % lambda>=0, gamma>1
% g_sel: calculate neg-log-likilihood (g) of B_j by regress j on sel_j (parents of j)
g_sel = @(X, sel_j, L, Diagl) trace(X(:,sel_j)'*X(:,sel_j) * L * L')*0.5 - sum(log(diag(Diagl)))*size(X,1);
e = @(k,n) [zeros(k-1,1);1;zeros(n-k,1)];
[n, p] = size(X); 

% record
if (SAVE) 
    t_rec = zeros(p,TMAX);
    [lnm_rec, gval_rec, err_rec, normt_rec] = deal(t_rec);
end 
% col by col updates
bres = bini; n_row = 1:n; 
for j = reshape(vec,1,[]) 
    % update j column
    if (FLAG)
        fprintf('node: %d\n', j);
    end
    bres(:,j) = 0; 
    iter = 0;  err = Inf;  
    % parent candidate of j in pi
    j_ind = find(pi == j);
    pac_j = pi(1:j_ind); 
    sel_cols = reshape(pac_j,1,[]);
    X_nj = X(:,sel_cols); 
    % obs for col j for selected parents
    obs_j = n_row(X_ind(:,j) == 0);
    X_nj = X_nj(obs_j,:); 
    % complete obs Xj
    X_obs = X(obs_j,:); 
    [nj, pj] = size(X_nj); % dim: nj * pj
    XtX = X_nj' * X_nj; % pj * pj
    
    bj = zeros(length(sel_cols),1);
    bj(end) = sqrt(nj)/norm(X_nj(:,end)); % estimate
    t = max(1, norm(bj,'fro'));
     
    % bj: current point; z: updated point
    while j_ind~=1 && iter < TMAX && err >= TOL;    
        
        iter = iter+1;
        if (FLAG)
            fprintf(' iter: %d\n', iter);
        end
        bold = bj; 
        
        % initialization for each prox. grad.
        T2 = 0;
        derib = XtX * bj - nj / bj(end) * e(pj,pj);
        bnm = norm(derib,'fro');
        % record
        if (SAVE)
            t_rec(j,iter) = t; lnm_rec(j,iter) = bnm; 
            normt_rec(j,iter) = t/bnm; 
        end

        % find step size
        while T2 <= TMAX 
            % fprintf('  find stepsize: %d iter, %1.1f size, %1.1f adjusted \n', T2, t, t/lnm);
            T2 = T2+1;
            ztl = bj-t*derib/bnm;
            zoff = prox_th_mcp(lambda, gamma, ztl-ztl(end)*e(pj,pj), t/bnm);
            z = zoff + ztl(end)*e(pj,pj);
            
            if (g_sel(X_obs, sel_cols, z, z(end))  <= ...
                    g_sel(X_obs, sel_cols, bj, bj(end)) + ...
                    derib' * (z-bj) + norm(z-bj, 'fro')^2*bnm/(2*t) ) 
                break
            else
                t = beta * t;
            end
        end % end of step size search
        bj = z;
        % err = norm(l - lold,'fro')/demonimator_err;
        err = max(vecnorm(bj - bold)./max(1, norm(bj,'fro')));
        
        if (FLAG)
            fprintf('  find stepsize: %d iter, %1.1f size, %1.1f adjusted, nnz: %d, err: %1.1e \n', ...
                T2, t, t/bnm, nnz(bj), err);
        end
        % record
        if (SAVE) 
            temp = bini; temp(:,j) = 0; temp(sel_cols,j) = bj;
            gval_rec(j,iter) = g(X,P,temp, diag(diag(temp)));
            err_rec(j,iter) = err;  
        end
    end
    bres(sel_cols,j) = bj;
end

gval = 0;
for j = 1:p
    obs_j = n_row(X_ind(:,j) == 0);
    X_obs = X(obs_j,:); 
    z = bres(:,j);
    gval = gval + g_sel(X_obs, 1:p, z, z(j));
end
fval = gval + sum(sum(MCP(bres - diag(diag(bres)))));

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



