function tree = maketree(featurelabels,trainfeatures,targets,epsino)
tree=struct('pro',0,'value',-1,'child',[],'parentpro',-1);
[n,m] = size(trainfeatures); %where n represent total numbers of features,m represent total numbers of examples
cn = unique(targets);%different classes
l=length(cn);%totoal numbers of classes
if l==1%if only one class,just use the class to be the lable of the tree and return
    tree.pro=0;%reprensent leaf
    tree.value = cn;
    tree.child=[];
    return
end
if n==0% if feature number equals 0
    H = hist(targets, length(cn)); %histogram of class
   [ma, largest] = max(H); %ma is the number of class who has largest number,largest is the posion in cn
   tree.pro=0;
   tree.value=cn(largest);
   tree.child=[];
   return
end

pnode = zeros(1,length(cn));
%calculate info gain
for i=1:length(cn)
    pnode(i)=length(find(targets==cn(i)))/length(targets);
end
H=-sum(pnode.*log(pnode)/log(2));
maxium=-1;
maxi=-1;
g=zeros(1,n);
for i=1:n
    fn=unique(trainfeatures(i,:));%one feature has fn properties
    lfn=length(fn);
    gf=zeros(1,lfn);
    fprintf('feature numbers:%d\n',lfn);
    for k=1:lfn
        onefeature=find(fn(k)==trainfeatures(i,:));%to each property in feature,,calucute the number of this property
        for j=1:length(cn)
            oneinonefeature=find(cn(j)==targets(:,onefeature));
            ratiofeature=length(oneinonefeature)/length(onefeature);
            fprintf('feature %d, property %d, rationfeature:%f\n',i, fn(k),ratiofeature);
            if(ratiofeature~=0)
                gf(k)=gf(k)+(-ratiofeature*log(ratiofeature)/log(2));
            end
        end  
        ratio=length(onefeature)/m;
        gf(k)=gf(k)*ratio;
    end
    g(i)=H-sum(gf);
    fprintf('%f\n',g(i));
    if g(i)>maxium
        maxium=g(i);
        maxi=i;
    end
end

if maxium<epsino%when the max info gain is less than thredhold value
    H = hist(targets, length(cn)); %histogram of class
   [ma, largest] = max(H); %ma is the number of class who has largest number,largest is the posion in cn
   tree.pro=0;
   tree.value=cn(largest);
   tree.child=[];
   return
end

tree.pro=1;%1 represent it's a inner node,0 represents it's a leaf
tv=featurelabels(maxi);
tree.value=tv;
tree.child=[];
featurelabels(maxi)=[];

%split data according feature
[data,target,splitarr]=splitData(trainfeatures,targets,maxi);
%tree.child=zeros(1,length(data));
%build child tree;
fprintf('split data into %d\n',length(data));
for i=1:length(data)
   disp(data(i));
   fprintf('\n');
   disp(target(i));
   fprintf('\n');
end
fprintf('\n');

for i=1:size(data,1)
    result = zeros(size(data{i}));
    result=data{i};
    temptree=maketree(featurelabels,result,target{i},0);
    tree.pro=1;%1 represent it's a inner node,0 represents it's a leaf
    tree.value=tv;
    tree.child(i)=temptree;
    tree.child(i).parentpro = splitarr(i);
    fprintf('temp tree\n');
    disp(tree.child(1));
    fprintf('\n');
end
disp(tree);
fprintf("now root tree,tree has %d childs\n",size(tree.child,2));
fprintf('\n');
for i=1:size(data,1)
    disp(tree.child(i));
    fprintf('\n');
end
fprintf('one iteration ends\n');
end



    
    

