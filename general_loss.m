function J = general_loss(W,X,Y,beta,lambda,t,type)
%This code can calculate the cost function of usual logistic regression and
%atml with logistic regression

d = size(X,2); % the number of features
n = max(Y); % the number of classes
m = length(Y); %the number of samples
y = zeros(m,1);
y = Y == t;
m_t = sum(y);

J = 0;

% In case that type is logistic
if strcmp(type,'logistic')
  
    J = -1/m_t*( y'*log(sigmoid(X*W(:,t))) + (1-y)'* log(1-sigmoid(X*W(:,t))) ) + lambda/(2*m_t)*W(:,t)'*W(:,t);
end
% In case that type is atml with logistic regresssion
if strcmp(type,'amtl')
    J = beta(:,t) * beta(:,t)' * lossfunc_t(W,X,y,t) + lambda* (W(:,t)-W*beta(:,t))'*(W(:,t)-W*beta(:,t)); 
end

if strcmp(type,'amtl_general')
    %J = -1/m_t *( y'*log(sigmoid(X*W(:,t))) + (1-y)'* log(1-sigmoid(X*W(:,t))) );
    J = -1/m *( y'*log(sigmoid(X*W(:,t))) + (1-y)'* log(1-sigmoid(X*W(:,t))) );
end

if strcmp(type,'regression')
    m_t = length(Y{t});
    J = 1/(2*m_t) * (Y{t} - X{t} * W(:,t))'*(Y{t} - X{t}*W(:,t));
end



end
