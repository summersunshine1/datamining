clear all; close all; clc
featurelabels=[1,2,3,4];
trainfeatures=[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3;
                0,0,1,1,0,0,0,1,0,0,0,0,1,1,0;
                0,0,0,1,0,0,0,1,1,1,1,1,0,0,0;
                1,2,2,1,1,1,2,2,3,3,3,2,2,3,1
                ];
targets=[0,0,1,1,0,0,0,1,1,1,1,1,1,1,0];
tree=maketree(featurelabels,trainfeatures,targets,0);
printTree(tree);
data=[2,0,0,1];
result=classify(data,tree);
fprintf('The result is %d\n',result);

