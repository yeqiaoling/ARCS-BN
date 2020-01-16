function [Bhat, rhom, pvalm] = constraint_est_dag_given_p_delete_intv(data_full, data_ind, pi, B_hat, varargin)
% arguments
parser = inputParser;
parser.KeepUnmatched = true;
addOptional(parser,'alpha', 0.01)
parse(parser, varargin{:});
alpha = parser.Results.alpha;

%%
[~,p] = size(data_full);
Bhat = zeros(p);
rhom = zeros(p);
pvalm = zeros(p);
pi = reshape(pi,1,length(pi));

for j = 2:p % j-th node in permutation
    node = pi(j);
    pac = find(B_hat(:,node)~=0);
    
    pa_node = zeros(p,1);
    rho_node = zeros(p,1);
    pval_node = zeros(p,1)-inf;
    test_set = pac;
    
    intv_row = find(data_ind(:,node) == 1);
    data = data_full;
    data(intv_row,:) = [];
    [n,~] = size(data);
    while (~isempty(test_set) && ~isempty(pac))
        count = length(test_set);
        k = pac(count);
        test_pac = pac(pac~=k);
        [rho] = partialcorr(data(:,node),data(:,k),data(:,test_pac));
        zval = 0.5*sqrt(n-length(test_pac)-3)*log(abs(1+rho)/abs(1-rho));
        pval = 2 * normcdf(abs(zval) * -1);
        cutoff = norminv(1-alpha/2);
        
        rho_node(k) = rho;
        pval_node(k) = pval;
        
        if (abs(zval) > cutoff)
            pa_node(k) = 1;
        else
            pac(count) = [];
        end
        test_set(count) = [];
        
    end
    Bhat(:,node) = pa_node;
    rhom(:,node) = rho_node;
    pvalm(:,node) = pval_node;
end

