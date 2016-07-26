function W_new = learn_W_regression(W,beta,X,Y,idx,lambda,param)

%m = length(Y{1});
% T= max(Y);
% EPS = 0.001;
% alpha = 1;
normG = Inf;
%lambda_stl = param.lambda_stl;
lambda_stl = 0;
% k = 1;
% arm_b = 0.1; % beta for armijo condition
% Maxiter = 10;
% Out_maxiter = 500;
fopt = Inf;
Wopt = W;
d = size(W,1);
sigma = param.sigma;
T = length(idx.tr)-1;
EPS = 0.0001;
alpha = 0.005; %school 0.1
k = 1;
maxiter = 1000;
numsamples = 100;
report_interval = maxiter/20;
iter = 0;
G = zeros(size(W));
numtrains = zeros(1,T);
for t=1:T
    numtrains(t) = length(idx.tr(t+1)-idx.tr(t));
end
randidx = cell(T,1);
while (normG > EPS) && (maxiter > iter)
        % sample training examples to use for computing stochastic gradient
        randidx = [];
				numsample = [];
        for t=1:T
            numsample(t) = min(numsamples, numtrains(t));
            %list = find(Y==t);
            randidx{t}.index =  idx.tr(t)+randperm(numtrains(t), numsample(t))-1;
      	end
				
				
				prevG = G;
				g_weight = zeros(1,T);
				g_loss = zeros(size(W,1),T);
				parfor t=1:T
                    st_x = X(randidx{t}.index,:);
                    st_y = Y(randidx{t}.index);
						w = W(:,t);
						%y = zeros(sum(numsample),1);
		        		%y(find(st_y==t)) = 1;
						%m_t = numsample(t);
                        m_t = idx.tr(t+1)-idx.tr(t);
                        %m_t = size(X,1);
						if (param.stl)
                                g_reg = w;
                        else
                                g_reg = (w - W * beta(:,t)) -(W-W*beta)*beta(t,:)';
                                g_reg = lambda*g_reg;
                                %g_reg = 2*lambda*g_reg;
                                %g_reg = lambda*( (w-W*beta(:,t)) - sum(beta(t,:)) * ones(d,1));
                        end

                        if (param.stl)
							g_loss = lambda*(-st_x')*(st_y - st_x*W(:,t))/m_t;
                        else
							%g_loss = (lambda*(norm(beta(:,t),1)) + 1) *( (-st_x')*(st_y - st_x*W(:,t))/m_t + lambda_stl*W(:,t) )/m_t ;
							%g_weight(t) = (lambda*(norm(beta(:,t),1)) + 1);
							%g_loss(:,t) = 1/m_t* st_x' * (sigmoid( st_x* W(:,t)) - y);
                             %g_loss = param.sf/(m_t)^sigma*lambda*(1+norm(beta(t,:),1)) *(- (st_x') * (st_y - st_x*W(:,t)) )/m_t;
                            g_loss = param.c_t(t)*(1+norm(beta(t,:),1)) *(- (st_x') * (st_y - st_x*W(:,t)) )/m_t;
                        end
						G(:,t) = g_loss + g_reg;
                end

				for t=1:T
					%G(:,t) = g_weight(t)*g_loss(:,t)*exp(sigmoid(st_x*W(:,t)-y) - sigmoid(st_x*W) + g_reg;
				end
				stepsize = alpha/norm(G,'fro');
				W = W - stepsize * G;

        iter = iter + 1;
      	
				hold on;
				normG = norm(G-prevG,'fro');	
        if mod(iter,report_interval) == 1;
                            param.lambda = lambda;
							[f fobj fobj2 freg] = regression_loss(W,beta,X,Y,idx,0,param,'amtl');
							if (f < fopt)
									fopt = f;
									Wopt = W;
							end
							
							elapsed = toc;
							fprintf('\r%d) f = %4.6f, obj_weighted = %4.6f, obj_ori = %4.6f, reg = %4.6f ||G|| = %4.6f (elapsed time = %4.6f)\n', iter, f, fobj, fobj2, freg, normG, elapsed);		
							subplot(1,3,1), plot(iter,f,'b.');
        			drawnow;
                    
				end    
end



% for t = 1:T;
%     alpha_temp = alpha;
%     %iter=1;
%     %y = zeros(m,1);
%     %y(find(Y==t)) = 1;
%     %m_t = sum(y);
%     m_t = length(Y{t});
%     W_grad = T*lambda*(norm(beta(t,:),1) + 1)* 1/(2*m_t) * (Y{t} - X{t}*W(:,t))'*(Y{t} - X{t}*W(:,t))+ (W(:,t) - W * beta(:,t)) - (W-W*beta)*beta(t,:)';
%     
%     W_1 = W; W_2 = W;
%     Out_iter = 1;
%     while (norm(W_grad) > EPS) && (Out_maxiter > Out_iter);
%         W_2(:,t) = W(:,t);
%         W_grad = T*lambda*(norm(beta(t,:),1) + 1) * 1/(2*m_t) * (Y{t}-X{t}*W(:,t))' * (Y{t}-X{t}*W(:,t))  + (W(:,t) - W * beta(:,t)) -(W-W*beta)*beta(t,:)';
%         W_1(:,t) = W(:,t) - alpha_temp * W_grad;
%         
%         
%         %tic;
%         %Use armijo line search
%         loop_iter=1;
%         while ( cal_loss(W_1,W_1,beta,X,Y,lambda,0,'regression') > cal_loss(W_2,W_2,beta,X,Y,lambda,0,'regression') - alpha_temp*arm_b*(W_grad'*W_grad) ) && (Maxiter > loop_iter);
%                 alpha_temp = alpha_temp*0.5;
%                 W_1(:,t) = W_2(:,t)-alpha_temp*W_grad;
%                 loop_iter=loop_iter+1;
%         end
%         %toc;
%         W(:,t) = W_1(:,t);
%         Out_iter = Out_iter + 1;
% 
%         
%     end
% 
%     subplot(4,2,3);
%     drawnow;
%     hold on;
%     plot(k,cal_loss(W,W,beta,X,Y,lambda,0,'regression'),'b.');
%     k = k+1;
% 
% end
W_new = Wopt;

end
