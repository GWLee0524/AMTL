function W = linear_regression(X,Y,idx,lambda)

% linear_regression codes

T = length(idx.tr)-1; % the number of tasks
d = size(X,2); % dimension
m = size(X,1); % the number of samples
W = zeros(d,T);

for t = 1:T;
    n_t = idx.tr(t+1)-idx.tr(t);
    W(:,t) = pinv(X(idx.tr(t):idx.tr(t+1)-1,:)'*X(idx.tr(t):idx.tr(t+1)-1,:)+lambda*eye(d)) * X(idx.tr(t):idx.tr(t+1)-1,:)' * Y(idx.tr(t):idx.tr(t+1)-1);
end

end