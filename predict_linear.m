function err = predict_linear(W,X,Y,idx)

%prediction for linear regression
T = length(idx.tr)-1;
%m = size(X,1);
%pred = zeros(m,T);
err = zeros(T,1);
for t = 1:T;
    %pred(t) = X{t} * W(:,t);
    err(t) = sqrt( (Y(idx.te(t):idx.te(t+1)-1)-X(idx.te(t):idx.te(t+1)-1,:)*W(:,t))'*(Y(idx.te(t):idx.te(t+1)-1) - X(idx.te(t):idx.te(t+1)-1,:)*W(:,t))/length(Y(idx.te(t):idx.te(t+1)-1)) ); 
end

end