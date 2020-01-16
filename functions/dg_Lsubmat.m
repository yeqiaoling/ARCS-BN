function [Lsubmat] = dg_Lsubmat(Sigma, n, Lsub, vec)
%
% objective: find the gradient of g (negative log-likelihood) w.r.t
%            a submatrix of a lower triangular matrix
% notation: L - lower triangular matrix
%           Lsub - a submatrix of L, choosing certain columns of L
%           g(L,P) = n/2 tr(P \hat\Sigma P' L L') - n log(|L|) 
%           dg_L = n \Pi_L(P \hat\Sigma P' L ) - diag(1/diag(L))
%
% input: Sigma = P \hat\Sigma P'
%        Lsub: choose m columns of L ?p * m?
%        n: # of obs of X
% idea: this function calculate gradient of g w.r.t L, then take the
%       correspondign columns of L as an output
%
p = size(Lsub,1);
term1 =  n*Sigma*Lsub; % dim: p * m

% gradient of trace term
ZERO = zeros(p);
ZERO(:,vec) = term1; 

% gradient of lo\g term
LOG = zeros(p); LOG(:,vec) = Lsub; temp = diag(LOG);
term2 =  diag(temp);
term2 =  1./diag(term2) ;
term2(~isfinite(term2))=0;

% combine 
ZERO = ZERO - diag(term2)*n;
Lsubmat = tril(ZERO);
Lsubmat = Lsubmat(:,vec);