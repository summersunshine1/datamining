function [alphas,b]=select(alphas,c,k,y,b,E,tol)
m=size(y,1);
r=find(alphas>0&&alphas<c);%no bundary variant
if(size(r,2)==0)%if no 0<alphas<c,go over training example,and find examples against kkt
    r=[1:1:m];
end
r1=find(alphas==0||alphas==c);%boundary variant
n=length(r);
n1=length(r1);
choosefirst=1;
choosefirst1=1;
for i=1:m
    E(i) = b + sum (alphas.*y.*k(:,i)) - y(i);
end
times=0;
while true 
    times++;
    if(times>20)
        break;
    end
    fprintf("cycle times %d\n",times);
    while choosefirst<=n
        index=r(choosefirst);
        g(index) = b + sum (alphas.*y.*k(:,index));
        if KKT(alphas,index,g,y,c)==1%no boundary alpha conform kkt
            fprintf('kkt no boundary first1 %d\n', index);
            a1=index;
            break;
        end
        choosefirst++;
    end
    if choosefirst==n+1% find no boundary alpha conform kkt,so begin search boundary
        while choosefirst1<=n1
            index=r1(choosefirst1);
            g(index) = b + sum (alphas.*y.*k(:,index));
            if KKT(alphas,index,g,y,c)==1
                fprintf('kkt boundary first2 %d\n', index);
                a1=index;
                break;
            end
            choosefirst1++;
        end
    end
    if choosefirst1==n+1 %meet the stop condition
        fprintf("all condition met\n");
        break;
    end
        
    %choose scond alpha 
    choosesecond1=1;
    choosesecond2=1;
    while choosesecond1<=n
        [flag,tempa2]=chsecond(E,r,a1,k);
        if flag~=0%no boundary alpha can change greatly
            a2=tempa2;
            fprintf('kkt no boundary second1 %d\n', a2);
            break;
        end
        choosesecond1++;
    end
    if choosesecond1==n+1
        while choosesecond2<=n1%find no boundary alpha change greatly,so begin search boundary
            [flag,tempa2]=chsecond(E,r1,a1,k);
            if flag~=0
                fprintf('kkt boundary second2 %d\n', a2);
                a2=tempa2;
                break;
            end
            choosesecond2++;
        end
        if choosesecond2==n1+1%no alpha2 meet the condition so rechoose alpha1
            continue;
        end
    end
    fprintf('updtae alpha begin\n');
    [flag1,tempalphas,tempb]= updatevariant(a1,a2,alphas,c,b,y,k,E,tol);%update alpha and b
    if flag1==1      
        alphas=tempalphas;
        b=tempb;
        fprintf('updtae alpha and b\n');
        r=find(alphas>0&&alphas<c);%no bundary variant
        if(size(r,2)==0)%if no 0<alphas<c,go over training example,and find examples against kkt
            r=[1:1:m];
        end
        r1=find(alphas==0||alphas==c);%boundary variant
        n=length(r);
        n1=length(r1);
        choosefirst=1;
        choosefirst1=1;
    end
    fprintf('updtae alpha end\n');    
end
end
            
        

    