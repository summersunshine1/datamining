clear ; close all; clc
% load('ex6data1.mat');
% plotData(X, y);
% fprintf('Program paused. Press enter to continue.\n');
% pause;
% fprintf('\nTraining Linear SVM ...\n');
% C = 100;
% model = svmtrain(X, y, C, @linearKernel, 1e-3);
% visualizeBoundaryLinear(X, y, model);
% fprintf('Program paused. Press enter to continue.\n');
% pause;
load('ex6data2.mat');

% SVM Parameters
C = 1; sigma = 0.1;

% We set the tolerance and max_passes lower here so that the code will run
% faster. However, in practice, you will want to run the training to
% convergence.
model= svmtrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);

% fprintf('Program paused. Press enter to continue.\n');
% pause;

% fprintf('Program paused. Press enter to continue.\n');
% pause;