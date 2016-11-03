function flag=KKT(alphas,index,g,y,c)
value=g*y(index);
delta=1e-3;
upvalue=delta+1;
lowvalue=1-delta;
if alphas(index)>0&&alphas(index)<c&&(value>upvalue||value<lowvalue)
    flag=1;
    return;
end
if alphas(index)==0&&value<lowvalue
    flag=1;
    return;
end
if alphas(index)==c&&value>upvalue
    flag=1;
    return;
end
flag=0;
end