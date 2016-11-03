function [flag,alphas,b,E]=seconddecison(alphas,E,a1,a2,K,Y,tol,C,b)
flag=1;
eta = 2 * K(a1,a2) - K(a1,a1) - K(a2,a2);
if eta>=0
    flag = 0;
    return;
end
if (Y(a1) == Y(a2))
    L = max(0, alphas(a2) + alphas(a1) - C);
    H = min(C, alphas(a2) + alphas(a1));
else
    L = max(0, alphas(a2) - alphas(a1));
    H = min(C, C + alphas(a2) - alphas(a1));
end

if (L == H)
    flag=0;
    return;
end

alpha_i_old = alphas(a1);
alpha_j_old = alphas(a2);
% Compute and clip new value for alpha j using (12) and (15).
alphas(a2) = alphas(a2) - (Y(a2) * (E(a1) - E(a2))) / eta;

% Clip
alphas(a2) = min (H, alphas(a2));
alphas(a2) = max (L, alphas(a2));
if (abs(alphas(a2) - alpha_j_old) < tol)
    % continue to next i. 
    % replace anyway
    %fprintf("update too slow\n");
    alphas(a2) = alpha_j_old;
    flag=0;
    return;
end
oldb=b;
alphas(a1) = alphas(a1) + Y(a1)*Y(a2)*(alpha_j_old - alphas(a2));
b1 = b - E(a1) ...
     - Y(a1) * (alphas(a1) - alpha_i_old) *  K(a1,a1)' ...
     - Y(a2) * (alphas(a2) - alpha_j_old) *  K(a1,a2)';
b2 = b - E(a2) ...
     - Y(a1) * (alphas(a1) - alpha_i_old) *  K(a1,a2)' ...
     - Y(a2) * (alphas(a2) - alpha_j_old) *  K(a2,a2)';
if (0 < alphas(a1) && alphas(a1) < C)
    b = b1;
elseif (0 < alphas(a2) && alphas(a2) < C)
    b = b2;
else
    b = (b1+b2)/2;
end
m=length(Y);
for i=1:m
    E(i) = b + sum (alphas.*Y.*K(:,i)) - Y(i);
end
flag=1;
% fprintf("old alpha:%d,%d alpha index %d %d and new alpha %d,%d and old b and new b %d,%d\n",alpha_i_old,alpha_j_old,a1,a2,alphas(a1),alphas(a2),oldb,b);  
end