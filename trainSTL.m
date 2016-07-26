function W_stl = trainSTL(X,Y,lambda)
% learn single task learning (one vs all)

T = length(unique(Y)); % the number of tasks
m = size(X,1); % the number of dataset
d = size(X,2); % dimensionality
W_stl = zeros(d,T);

%some options for optimization step
EPS = 0.00001;
stepsize = 0.01; %initial stepsize
MaxIter = 2000;
MaxLSIter = 10; % Max Line search iter
DebugFlag = 1;

parfor t = 1:T;
    y = Y == t;
    iter = 1;
    w_t = zeros(d,1);
    grad = 1/m*X'*(sigmoid(X*w_t)-y)+lambda/m*w_t;
    while( norm(grad,1)/d > EPS && iter<=MaxIter)
        Obj_before = -1/2/m*(y'*log(sigmoid(X*w_t)) + (1-y)'*log(1-sigmoid(X*w_t))) + lambda/2/m*w_t'*w_t;
        w_t_before = w_t;
        w_t = w_t - stepsize*grad;
        Obj_new = -1/2/m*(y'*log(sigmoid(X*w_t)) + (1-y)'*log(1-sigmoid(X*w_t))) + lambda/2/m*w_t'*w_t;
        inner_iter = 1;
        while Obj_new > Obj_before && inner_iter <= MaxLSIter;
            stepsize = stepsize*0.5;
            w_t = w_t_before - stepsize*grad;
            Obj_new = -1/2/m*(y'*log(sigmoid(X*w_t)) + (1-y)'*log(1-sigmoid(X*w_t))) + lambda/2/m*w_t'*w_t;
            inner_iter = inner_iter + 1;
        end
        %for debugging plot obj value
        if mod(iter,50)==1 && DebugFlag == 1;
            drawnow;
            hold on;
            plot(iter,Obj_new,'rx');
        end
        iter = iter+1;
    end
    W_stl(:,t) = w_t;
end


end