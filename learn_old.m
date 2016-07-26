function W = learn_old(X,Y,lambda)

%learn one vs all logistic regression model.

T = length(unique(Y)); % the number of classes
d = size(X,2); % the number of features
m = size(X,1); % the number of train samples

W = zeros(d,T); %initialize weight parameter(model parameter)
alpha = 5; %learning rate
eps = 0.001;
Maxiter = 15;
Out_maxiter = 8000;
m =length(Y);
parfor t = 1:T ;
    m = length(Y);
    y=zeros(m,1);
    y(find(Y==t)) = 1;
    m_t = sum(y);
    %gradient descent for each task
%     for j =1:5; %change it to while
%         %J_old = general_loss(W,X,Y,0,lambda,t,'logistic');
%         W_grad = 1/m*X'*(sigmoid(X*W(:,t))-y)+lambda/m*[W(1:end-1,t); 0];
%         W(:,t) = W(:,t)-alpha*W_grad;
%         %J_new = general_loss(W,X,Y,0,lambda,t,'logistic');
%     end
    W_grad = ones(d,1);
    W_t = W(:,t);
    m = m_t;
    Out_iter = 1;
    alpha_temp = alpha;
    while(norm(W_grad) > eps) &&(Out_maxiter > Out_iter )
        %alpha_temp = alpha;
        %J_old = general_loss(W,X,Y,0,lambda,t,'logistic');
        J_old = -1/(2*m) *( y'*log(sigmoid(X*W_t)) + (1-y)'*log(1-sigmoid(X*W_t)) ) + lambda/2/m*norm(W_t(1:end-1),'fro');
        W_grad = 1/m*X'*(sigmoid(X*W_t)-y)+lambda/m*[W_t(1:end-1); 0];
        %W_grad = 1/m*X'*(sigmoid(X*W(:,t))-y)+lambda/m * W(:,t);
        W_before = W_t;
        W_t = W_t-alpha_temp*W_grad;
        %J_new = general_loss(W,X,Y,0,lambda,t,'logistic');
        J_new = -1/(2*m) *( y'*log(sigmoid(X*W_t)) + (1-y)'*log(1-sigmoid(X*W_t)) ) + lambda/2/m*norm(W_t(1:end-1),'fro');
        iter=1;
        
        while ( J_new > J_old - alpha_temp*0.1*(W_grad'*W_grad) ) && (Maxiter > iter);
            alpha_temp = alpha_temp*0.5;
            W_t = W_before - alpha_temp * W_grad;
            %J_new = general_loss(W,X,Y,0,lambda,t,'logistic');
            J_new = -1/(2*m) *( y'*log(sigmoid(X*W_t)) + (1-y)'*log(1-sigmoid(X*W_t)) ) + lambda/2/m*norm(W_t(1:end-1),'fro');
%             iter
            iter = iter+1;
        end
%         if mod(Out_iter,200)==1;
%         fprintf('%d ) J is %6.4d\n',Out_iter,J_new);
       % X*W(:,t)
      
%         if(isnan(J_new))
%             m
%             y'*log(sigmoid(X*W(:,t))) + (1-y)'*log(1-sigmoid(X*W(:,t)))
%             log(sigmoid(X*W(:,t)))'
%             log(1-sigmoid(X*W(:,t)))'
%             pause;
%         end
        
 %       end
%         plot(Out_iter,J_new,'b.');hold on; drawnow;
%         end
%         J_new = general_loss(W,X,Y,0,lambda,t,'logistic');
%         norm(W)
%         pause;
          %  J_new
          W(:,t) = W_t;
         Out_iter = Out_iter + 1;
    end
    
    

end

% for k = 1:T;
%     init_W = zeros(d,1);
%     options = optimset('GradObj','on','MaxIter',100);
%     [theta] = fmincg( @(t)( lrCostFunction( t , X , (Y==k) , lambda ) ) , init_W , options) ;
%     W(:,k) = theta;
% end


end
