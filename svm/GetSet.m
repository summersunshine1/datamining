function [r,r1] = GetSet(alphas,c)
ra=find(alphas>0);%no bundary variant
rb=find(alphas<c);
r=intersect(ra,rb);
% if(size(r,2)==0)%if no 0<alphas<c,go over training example,and find examples against kkt
    % r=[1:1:m];
% end
r1a=find(alphas==0);%no bundary variant
r1b=find(alphas==c);
r1=union(r1a,r1b);%boundary variant
end