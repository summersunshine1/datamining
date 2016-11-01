function flag=KKT(alphas,index,g,y,c)
if alphas(index)>0&&alphas(index)<c&&g(index)*y(index)~=1
    flag=1;
    return;
end
if alphas(index)==0&&g(index)*y(index)<1
    flag=1;
    return;
end
if alphas(index)==c&&g(index)*y(index)>1
    flag=1;
    return;
end
flag=0;
end