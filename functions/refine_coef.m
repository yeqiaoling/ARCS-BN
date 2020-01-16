function [] = refine_coef(X, bhat, varargin)
% object: perform conditional independence tests in order to remove FP edges
% input: data X, coefficient bhat
%   optional inputs: alpha: a vector of significance levels
% output:

%% inputs
parser = inputParser;
parser.KeepUnmatched = true;
addOptional(parser,'alpha', 1e-5)
addOptional(parser,'X_ind', 0)

parse(parser, varargin{:});
alpha = parser.Results.alpha;
X_ind = parser.Results.X_ind;


%% START
[~,p] = size(X);
pi = toposort(digraph(bhat)); I = eye(p); topo_p = I(pi,:);

for al = alpha
    tic;
    if (sum(X_ind) == 0)
        [Bhat, ~, ~] = constraint_est_dag_given_p_delete(X, topo_p, bhat, 'alpha', al);
    else 
        [Bhat, ~, ~] = constraint_est_dag_given_p_delete_intv(X, X_ind, pi, bhat, 'alpha', al);
    end
    time_use = toc;
    filename = ['adjMat_refined_' num2str(al) '.txt'];
    save(filename, 'Bhat', '-ascii')
end
end