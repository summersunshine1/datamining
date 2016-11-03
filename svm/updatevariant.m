function [flag,alphas,b]= updatevariant(i,j,alphas,C,b,Y,K,E)
alpha_i_old = alphas(i);
alpha_j_old = alphas(j);
flag=1;
if (Y(i) == Y(j))
    L = max(0, alphas(j) + alphas(i) - C);
    H = min(C, alphas(j) + alphas(i));
else
    L = max(0, alphas(j) - alphas(i));
    H = min(C, C + alphas(j) - alphas(i));
end

if (L == H)
    fprintf("L equal H\n");
    flag=0;
    return;
end

% Compute eta by (14).
eta = 2 * K(i,j) - K(i,i) - K(j,j);

% Compute and clip new value for alpha j using (12) and (15).
alphas(j) = alphas(j) - (Y(j) * (E(i) - E(j))) / eta;

% Clip
alphas(j) = min (H, alphas(j));
alphas(j) = max (L, alphas(j));
% Check if change in alpha is significant
% if (abs(alphas(j) - alpha_j_old) < tol)
    % continue to next i. 
    % replace anyway
    % fprintf("update too slow\n");
    % alphas(j) = alpha_j_old;
    % flag=0;
    % return;
% end
fprintf("update............\n");
% Determine value for alpha i using (16). 
alphas(i) = alphas(i) + Y(i)*Y(j)*(alpha_j_old - alphas(j));

% Compute b1 and b2 using (17) and (18) respectively. 
b1 = b - E(i) ...
     - Y(i) * (alphas(i) - alpha_i_old) *  K(i,i)' ...
     - Y(j) * (alphas(j) - alpha_j_old) *  K(i,j)';
b2 = b - E(j) ...
     - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
     - Y(j) * (alphas(j) - alpha_j_old) *  K(j,j)';

% Compute b by (19). 
fprintf("update  bbbbbbbb............\n");
oldb=b;
if (0 < alphas(i) && alphas(i) < C)
    b = b1;
elseif (0 < alphas(j) && alphas(j) < C)
    b = b2;
else
    b = (b1+b2)/2;
end
fprintf("old alpha:%d,%d and new alpha %d,%d and old b and new b %d,%d\n",alpha_i_old,alpha_j_old,alphas(i),alphas(j),oldb,b);
end