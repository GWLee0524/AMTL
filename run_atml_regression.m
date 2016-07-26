% AMTL for linear regression
%% single task learning.
disp('learn single task learning');
clear all;
stl_arr = [];
result = [];
perclass_arr = [];
time = [];
tic;

addpath(genpath(pwd));
file = 'synthetic_dataset';
load(file);

if(0) %if you want to use validation set, then 1 
    test_x = val_x;
    test_y= val_y;
    teindex =valindex;
end


T = length(trindex); % the number of tasks
d = size(train_x,2); % dimension
m = size(train_x,1); % the number of samples

lambda_stl = 10; %Hyper parameter for single task learning (p)

idx.tr = [trindex; size(train_x,1)+1];
idx.te = [teindex; size(test_x,1)+1];

train_x = [train_x ones(size(train_x,1),1)];
test_x = [test_x ones(size(test_x,1),1)];
    

param.stl_learn =1;
if param.stl_learn == 1;
W_old = linear_regression(train_x,train_y,idx,lambda_stl);
stl_err = predict_linear(W_old,test_x,test_y,idx);

disp(mean(stl_err));
end






%% learn AMTL model
mid_result = [];

imbal = 0; %imbalane option flag 
delta=zeros(T,1);
param.stl = 0;
lambda_ridge =10; %Hyper parameter for AMTL (p)

W_lasso_re = W_old; % For regression

if(imbal)
    param.sigma = 2; %Used in case of imbalanced dataset, if not, then 0 (p) 
    param.sf = 4/10*(1000^param.sigma); %Used in case of imbalanced dataset, if not, then 0 (p) 
else
param.sf = 1; 
param.sigma = 0; 
end
param.lambda = lambda_ridge;
param.lambda2 = T;
param.lambda_stl = lambda_stl;

B_lasso = zeros(T-1,T);
B_lasso_full = zeros(T,T);
EPS = 0.01;
diff1 = 1; diff2=1;
MaxIter = 5; %Maximum alternation (p)
iter = 1;

param.c_t = zeros(T,1);
for t = 1:T;
    n_t(t) = idx.tr(t+1)-idx.tr(t);
    param.c_t(t) = param.sf/n_t(t)^param.sigma;
end


J_arr = [];
err_cell = cell(MaxIter,1); B_cell = cell(MaxIter,1); W_cell = cell(MaxIter,1); 
while (diff1 > EPS || diff2 > EPS) && MaxIter >= iter;
    fprintf('iteration %d\n',iter);
    %fprintf('parameters : lambda-> %f, sf-> %f\n',lambda_ridge,param.sf);
    tic;
    B_before = B_lasso_full;
    for l = 1:T;
        n_t = idx.tr(l+1)-idx.tr(l);
        delta(l) = regression_loss(W_lasso_re,0,train_x,train_y,idx,l,param,'stl');
    end

    B_lasso_full = learn_B_regression(W_lasso_re,delta,param);
    param.B = B_lasso_full;

    W_new = learn_W_regression(W_lasso_re,B_lasso_full,train_x,train_y,idx,lambda_ridge,param);
    W_before = W_lasso_re;
    W_lasso_re = W_new;
    diff1 = norm(W_lasso_re-W_before,2);
    diff2 = norm(B_lasso_full-B_before,2);
    

    amtl_err = predict_linear(W_lasso_re,test_x,test_y,idx);
    B_cell{iter} = B_lasso_full; W_cell{iter} = W_lasso_re; err_cell{iter} = amtl_err;
    disp(mean(amtl_err));

    save(sprintf('result_%s',file));
    iter = iter+1;
    toc;
end
stl_arr = [stl_arr mean(stl_err)];
result = [result mean(amtl_err)];
fprintf('average stl: %f\n average amtl: %f\n',mean(stl_arr),mean(result));

% param.sf
% lambda_arr
% stl_arr  
% result 
mean(perclass_arr);


