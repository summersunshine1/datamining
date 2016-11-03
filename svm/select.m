function [alphas,b]=select(alphas,c,k,y,b,E,tol)
m=size(y,1);
[r r1]=GetSet(alphas,c);
n=length(r);
n1=length(r1);
choosefirst=1;
choosefirst1=1;
for i=1:m
    E(i) = b + sum (alphas.*y.*k(:,i)) - y(i);
end
fprintf("no boundary %d and boundary %d\n",n,n1);
times=0;
%intial index
randindex1=randperm(n);%select variant randomly
randindex2=randperm(n1); 
while true 
    times++;
    % if(times>2000)
        % break;
    % end
    fprintf("cycle times %d\n",times);
    index=1;
    % while index<=m
        % g = b + sum (alphas.*y.*k(:,index));
        % if KKT(alphas,index,g,y,c)==1
            % break;
        % end
        % index++;
    % end
    % if index==m+1
       % fprintf("all condition met\n"); 
       % break;
    % end 
    %if choosefirst1>=n1+1
    % fprintf(".........update new r............\n");
    [r r1]=GetSet(alphas,c);
    n=length(r);
    n1=length(r1);
    choosefirst=1;
    choosefirst1=1;
    randindex1=randperm(n);%select variant randomly
    randindex2=randperm(n1); 
    % fprintf("no boundary %d and boundary %d \n",n,n1);
    %end
    % fprintf('choose first\n');
    firstfindflag=0;
    while choosefirst<=n
        index=r(mod(choosefirst+randindex1(1),n)+1);
        % index=choosefirst1;
        g = b + sum (alphas.*y.*k(:,index));
        if KKT(alphas,index,g,y,c)==1%no boundary alpha conform kkt
            fprintf('kkt no boundary first1 %d\n', index);
            a1=index;
            firstfindflag=1;
            choosefirst++;
            break;
        end
        choosefirst++;
    end
    if choosefirst==n+1&&firstfindflag==0% find no boundary alpha conform kkt,so begin search boundary
        while choosefirst1<=n1
            index=r1(mod(choosefirst1+randindex2(1),n1)+1);
            g = b + sum (alphas.*y.*k(:,index));
            if KKT(alphas,index,g,y,c)==1
                % fprintf('kkt boundary first2 %d\n', index);
                a1=index;
                firstfindflag=1;
                choosefirst1++;
                break;
            end
            choosefirst1++;
        end
    end
    if firstfindflag == 0
        fprintf("all condition meets");
        break;
    end
    % fprintf('choose second\n');    
    %choose scond alpha 

    flag=0;
    minusE=zeros(1,m);
    for i=1:m
        minusE(i)=abs(E(a1)-E(i));
    end
        
    [flag,tempalphas,tempb,tempE]=chsecond(E,a1,k,y,alphas,tol,c,b,minusE);
    if flag==1%no boundary alpha can change greatly
        alphas=tempalphas;
        b=tempb;
        E=tempE;
    end
    count=0;
    index=1;
    if flag==1    
        [r11 r12]=GetSet(alphas,c);
        n11=length(r11);
        n12=length(r12);
        while index<=m
            g = b + sum (alphas.*y.*k(:,index));
            if KKT(alphas,index,g,y,c)==1
                count++;
            end
            index++;
        end
        if count<10
            fprintf("training ends\n");
            break;
        end
        % fprintf("no boundary %d and boundary %d and against kkt %d\n",n,n1,count);
    end  
end
end
            
        

    