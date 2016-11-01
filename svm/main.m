clear ; close all; clc
load('ex6data1.mat');
plotData(X, y);
% fprintf('Program paused. Press enter to continue.\n');
% pause;
fprintf('\nTraining Linear SVM ...\n');
C = 100;
model = svmtrain(X, y, C, @linearKernel, 1e-3);
visualizeBoundaryLinear(X, y, model);

% fprintf('Program paused. Press enter to continue.\n');
% pause;