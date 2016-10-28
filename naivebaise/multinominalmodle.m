function [fi1,fi0,fi] = multinominalmodle(name,result,vacab)
m=length(name);
n = length(vacab);
fi1 = zeros(size(1,n));
f10 = zeros(size(1,n));
totallength1 = 0;
totallength0 = 0;
for i=1:m
    if result(i)==1
        totallength1=totallength1+length(name{i});
    else
        totallength0=totallength0+length(name{i});
    end
end
for k=1:n
    c1=0;
    c0=0;
    for i=1:m
        l = length(name{i});
        for j=1:l
            s=name{i}(j);
            if s==vacab(k)&&result(i)==1
                c1++;
            end
            if s==vacab(k)&&result(i)==0
                c0++;
            end
        end
    end
    fi1(k) = (c1+1)/(totallength1+n);
    fi0(k) = (c0+1)/(totallength0+n);           
end
fi=sum(result)/m;
end