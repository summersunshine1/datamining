function D = C4_5(train_features, train_targets, inc_node,test_features)


[Ni, M]		= size(train_features); %输入向量为NI*M的矩阵，其中M表示训练样本个数，Ni为特征维数维数
inc_node    = inc_node*M/100; 

disp('Building tree') 
tree        = make_tree(train_features, train_targets, inc_node); 
 
%Make the decision region according to the tree %根据产生的数产生决策域
disp('Building decision surface using the tree') 
[n,m]=size(test_features);
targets		= use_tree(test_features, 1:m, tree, unique(train_targets)); %target里包含了对应的测试样本分类所得的类别数
 
D   		= targets; 
%END 
 
function targets = use_tree(features, indices, tree,  Uc) %target里包含了对应的测试样本分类所得的类别
 

targets = zeros(1, size(features,2)); %1*M的向量
 
if (tree.dim == 0) 
   %Reached the end of the tree 
   targets(indices) = tree.child; 
   return %child里面包含了类别信息，indeces包含了测试样本中当前测试的样本索引
end 
         

dim = tree.dim; %当前节点的特征参数
dims= 1:size(features,1); %dims为1-特征维数的向量
 
   %Discrete feature 
   in				= indices(find(features(dim, indices) <= tree.split_loc)); %in为左子树在原矩阵的index

   targets		= targets + use_tree(features(dims, :), in, tree.child_1, Uc); 
   in				= indices(find(features(dim, indices) >  tree.split_loc)); 

   targets		= targets + use_tree(features(dims, :), in, tree.child_2, Uc); 
return 
      
 
function tree = make_tree(features, targets, inc_node) 

[Ni, L]    					= size(features); 
Uc         					= unique(targets); %UC表示类别数
tree.dim						= 0; %数的维度为0
%tree.child(1:maxNbin)	= zeros(1,maxNbin); 
 
if isempty(features), %如果特征为空，退出
   return 
end 

%When to stop: If the dimension is one or the number of examples is small 
if ((inc_node > L) | (L == 1) | (length(Uc) == 1)), %剩余训练集只剩一个，或太小，小于inc_node，或只剩一类，退出
   H					= hist(targets, length(Uc)); %返回类别数的直方图
   [m, largest] 	= max(H); %更大的一类，m为大的值，即个数，largest为位置，即类别的位置
   tree.child	 	= Uc(largest); %直接返回其中更大的一类作为其类别
   return
end 
 
%Compute the node's I 
%计算现有的信息量
for i = 1:length(Uc), 
    Pnode(i) = length(find(targets == Uc(i))) / L; 
end 
Inode = -sum(Pnode.*log(Pnode)/log(2)); 
 
%For each dimension, compute the gain ratio impurity 
%This is done separately for discrete and continuous features 
delta_Ib    = zeros(1, Ni); 
S=[];
for i = 1:Ni, 
   data	= features(i,:); 
   temp=unique(data); 
      P	= zeros(length(Uc), 2); 
       
      %Sort the features 
      [sorted_data, indices] = sort(data); 
      sorted_targets = targets(indices); 
       %结果为排序后的特征和类别
      %Calculate the information for each possible split 
      I	= zeros(1, L-1); 
      
      for j = 1:L-1, 
         for k =1:length(Uc), 
            P(k,1) = length(find(sorted_targets(1:j) 		== Uc(k))); 
            P(k,2) = length(find(sorted_targets(j+1:end) == Uc(k))); 
         end 
         Ps		= sum(P)/L; %两个子树的权重 
         temp1=[P(:,1)]; 
         temp2=[P(:,2)]; 
         fo=[Info(temp1),Info(temp2)];
         %info	= sum(-P.*log(eps+P)/log(2)); %两个子树的I
         I(j)	= Inode - sum(fo.*Ps);    
      end 
      [delta_Ib(i), s] = max(I); 
      S=[S,s];
   
end
 
%Find the dimension minimizing delta_Ib  
%找到最大的划分方法
[m, dim] = max(delta_Ib); 

dims		= 1:Ni; 
tree.dim = dim; 

%Split along the 'dim' dimension 
%分裂树 
   %Continuous feature 
   [sorted_data, indices] = sort(features(dim,:)); 
   %tree.split_loc		= split_loc(dim); 
   %disp(tree.split_loc);
   S(dim)
   indices1=indices(1:S(dim))
   indices2=indices(S(dim)+1:end)
   tree.split_loc=sorted_data(S(dim))
   tree.child_1		= make_tree(features(dims, indices1), targets(indices1), inc_node); 
   tree.child_2		= make_tree(features(dims, indices2), targets(indices2), inc_node); 
%D = C4_5_new(train_features, train_targets, inc_node);