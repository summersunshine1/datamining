function [fi1,fi0,fi] = multivariatemodle(name,result,vacab)
n=length(vacab);
m=length(name);
fi1=zeros(1,n);
fi0=zeros(1,n);
total1=sum(result);
total0=m-total1;
for i=1:n
    c1=0;
    c0=0;
    for j=1:m
        idx=strfind(name{j},vacab(i));
        if size(idx,1)~=0 && result(j)==1
            c1++;
        end
        if size(idx,1)~=0 && result(j)==0
            c0++;
        end
    end
    fi1(i)=(c1+1)/(total1+2);
    fi0(i)=(c0+1)/(total0+2);
end
fi=total1/m;
end
