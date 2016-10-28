function r=predict(fi0,fi1,fi,x,vacab)%given name in each test cases ,predict whether it's 1 or 0
m=size(x,1);
r=zeros(size(x));
n=size(vacab,2);
for i=1:m
    r0=0;
    r1=0;
    for j=1:n
        idx = strfind(x{i},vacab(j));
        if (size(idx,1))~=0
            if r1==0
                r1=fi1(j);
            else
                r1=r1*fi1(j);
            end
        else
            if r0==0
                r0=fi0(j);
            else
                r0=r0*fi0(j);
            end
        end    
    end
    r1=r1*fi;
    r0=r0*(1-fi);
    r(i)=(r1>r0);
end
end