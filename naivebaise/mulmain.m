clear all; close all; clc
[x,y]=textread("data.txt","%d%s");
m=size(x,1);
va=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','.'];
index=randperm(m);
[fi1,fi0,fi]=multivariatemodle(y(index(1:m/2),:),x(index(1:m/2),:),va);
r=predict(fi0,fi1,fi,y(index(m/2+1:m),:),va);
total = length(r);
correctnum = length(find(r==x(index(m/2+1:m),:)));
rate=correctnum/total;
fprintf("multivariate correct rate is %f\n",rate);
[fi1,fi0,fi]=multinominalmodle(y(index(1:m/2),:),x(index(1:m/2),:),va);
r=multipredict(fi0,fi1,fi,y(index(m/2+1:m),:),va);
total = length(r);
correctnum = length(find(r==x(index(m/2+1:m),:)));
rate=correctnum/total;
fprintf("multi nominal correct rate is %f\n",rate);