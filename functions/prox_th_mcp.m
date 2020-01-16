function [prox] = prox_th_mcp(lambda, gamma, x, t)
% 
% objective: find the proximal operator for MCP penalty 
% input: x - matrix (n*p)
%        lambda & gamma: MCP parameter
%        t: step size in proximal operator
% ouput: prox - matrix (n*p)
%
x_abs = abs(x);

if (gamma > t)
    term1 = (x_abs < gamma*lambda) .* (x_abs >= t*lambda) .* gamma / (gamma-t) .* (x - sign(x)*t*lambda);
    term2 = x .* (x_abs >= lambda*gamma);
    prox = term1 + term2;
elseif (gamma <t)
    prox = (x_abs.^2 >= lambda^2*t*gamma) .* x;
else
    prox = (x_abs >= lambda*gamma) * x;
end
