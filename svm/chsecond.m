function [flag,alphas,b,E]=chsecond(E,a1,K,Y,alphas,tol,C,b,minusE)
flag=1;
% if E(a1)<0
    % tempei=E(a1);
    % E(a1)=-inf;
    % [minv,minindex]=max(E(r,:));
    % E(a1)=tempei;
    % a2=minindex;
% else
    % tempei=E(a1);
    % E(a1)=inf;
    % [minv,minindex]=min(E(r,:));
    % E(a1)=tempei;
    % a2=minindex;
% end
minusE(a1)=-inf;
maxvalue=max(minusE);
maxindex=find(minusE==maxvalue);
% n=length(maxindex);
% randn=randperm(n);
% randni=randn(1);
a2=maxindex(1);
[reflag,tempalphas,tempb,tempE]=seconddecison(alphas,E,a1,a2,K,Y,tol,C,b);
if reflag==1
    alphas=tempalphas;
    b=tempb;
    E=tempE;
    return;
end
[r r1]=GetSet(alphas,C);
n=length(r);
n1=length(r1);
randindex1=randperm(n);%select variant randomly
randindex2=randperm(n1); 
for i=1:n
    a2=r(mod(i+randindex1(1),n)+1);
    [reflag,tempalphas,tempb,tempE]=seconddecison(alphas,E,a1,a2,K,Y,tol,C,b);
    if reflag==1
        alphas=tempalphas;
        b=tempb;
        E=tempE;
        return;
    end 
end
for i=1:n1
    a2=r1(mod(i+randindex2(1),n1)+1);
    [reflag,tempalphas,tempb,tempE]=seconddecison(alphas,E,a1,a2,K,Y,tol,C,b);
    if reflag==1
        alphas=tempalphas;
        b=tempb;
        E=tempE;
        return;
    end 
end
flag=0;
end
