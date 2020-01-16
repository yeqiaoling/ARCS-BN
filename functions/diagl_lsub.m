function [diagl] = diagl_lsub(Lsub, vec)
%
% objective: given Lsub (a p*m submatrix of a lower triangular matrix), 
%            extract its diagonal elements and output it as a m*m diagonal
%            matrix 
%
p = size(Lsub,1);
ZERO = zeros(p);
ZERO(:,vec) = Lsub;
diagl = diag(diag(ZERO));
diagl = diagl(vec,vec);