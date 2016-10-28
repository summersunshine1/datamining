function r=multipredict(fi0,fi1,fi,x,vacab)
m=length(x);
r=zeros(size(x));
n=length(vacab);
for i=1:m
    r1=0;
    r0=0;
    l=length(x{i});
    for j=1:l
        s=x{i}(j);
        idx=find(s==vacab);
        li=length(idx);
        for k=1:li
            if r1==0
                r1 = fi1(idx(k));
            else
                r1=r1*fi1(idx(k));
            end
            if r0==0
                r0 = fi0(idx(k));
            else
                r0=r0*fi0(idx(k));
            end
    end
    r1=r1*fi;
    r0=r0*(1-fi);
    r(i)=(r1>r0);
end
end

        
    