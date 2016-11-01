function [flag,a2]=chsecond(E,r,a1,k)
flag=0;
if E(a1)<0
    tempei=E(a1);
    E(a1)=-inf;
    [minv,minindex]=max(E(r,:));
    E(a1)=tempei;
    a2=minindex;
    flag=1;
else
    tempei=E(a1);
    E(a1)=inf;
    [minv,minindex]=min(E(r,:));
    E(a1)=tempei;
    a2=minindex;
    flag=1;
end
eta = 2 * k(a1,a2) - k(a1,a1) - k(a2,a2);
if eta>0
   flag=1; 
end
end