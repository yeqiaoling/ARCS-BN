function[B, Osq] = convert_PL_to_BO(l, sort_B_to_L)
% 
% objective: convert a coefficient matrix to another according to a certain
%            permutation
% idea: B -- by sort_B_to_L --> l
%
% input: l - a current coefficient matrix
%        sort_B_to_L: an order sorting B to l
% output: B and Omega square
%
p = size(l,1);
I = eye(p);

% default is sorting B according to an flip order of a topological sort,
% thus l is lower triangular matrix
if nargin < 2
    sort_B_to_L = flip(1:p);
end

l = I(sort_B_to_L,:)' * l * I(sort_B_to_L,:);
Om = diag(1./diag(l));
B = (diag(diag(l)) -  l)* Om;
Osq = Om * Om;