function [Wopt, alpha] = learnW(W,beta,X,Y,lambda, param)
%use stochatic gradiendt descent;
%m = length(Y);
%W = randn(size(W));
T = max(Y);
sigma = param.sigma;
EPS = 0.0001;
alpha = 0.01;
k = 1;
maxiter = 2000;
numsamples = 5;
report_interval = maxiter/20;

numtrains = zeros(1,T);
for t=1:T
    numtrains(t) = length(find(Y==t));
end

%for bcdIter = 1:5
%for t = 1:T;
iter = 0;
fopt = Inf;
normG = Inf;
f_diff = Inf;
tic;
G = zeros(size(W));
histG = zeros(length(Y), T);
outbeta = sum(beta,2);
%[sorted, idx] = sort(outbeta, 'descend');
%for i=1:length(idx)
%	fprintf('%s: %4.4f\n', param.classes{idx(i)}, sum(beta(:,idx(i))));
%end
[f fobj fobj2] = cal_loss(W,W,beta,X,Y,lambda,0,'lasso',param);
while (normG > EPS) && (maxiter > iter) && f_diff > 0.00001
        % sample training examples to use for computing stochastic gradient
        randidx = [];
				numsample = [];
				for t=1:T
            numsample(t) = min(numsamples, numtrains(t));
            list = find(Y==t);
            randidx = [randidx; list(randperm(numtrains(t), numsample(t)))];
      	end
				
				st_x = X(randidx,:);
        st_y = Y(randidx);
				prevG = G;
				g_weight = zeros(1,T);
				g_loss = zeros(size(W,1),T);
				for t=1:T
						w = W(:,t);
						y = zeros(sum(numsample),1);
                        y(find(st_y==t)) = 1;
						m_t = numsample(t);
						if (param.stl)
							g_reg = w;
						else
							g_reg = (w - W * beta(:,t)) -(W-W*beta)*beta(t,:)';
							g_reg = lambda*g_reg;
                            %g_reg = lambda*g_reg;
						end

        		if (param.stl)
							g_loss = lambda*1/m_t* st_x' * (sigmoid( st_x* W(:,t)) - y);
        		else
							g_loss = (param.c_t(t)*(norm(beta(:,t),1)) + 1) * 1/m_t* st_x' * (sigmoid( st_x* W(:,t)) - y);
							%g_weight(t) = (lambda*(norm(beta(:,t),1)) + 1);
							%g_loss(:,t) = 1/m_t* st_x' * (sigmoid( st_x* W(:,t)) - y);
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
                f1 = f;
                if mod(iter,report_interval) == 1;
					[f fobj fobj2] = cal_loss(W,W,beta,X,Y,lambda,0,'lasso',param);
					if (f < fopt)
						fopt = f;
						Wopt = W;
					end
							
					elapsed = toc;
					fprintf('\r%d) f = %4.6f, obj_weighted = %4.6f, obj_ori = %4.6f, ||G|| = %4.6f (elapsed time = %4.6f)\n', iter, f, fobj, fobj2, normG, elapsed);		
					subplot(1,3,1), plot(iter,f,'b.');
        			drawnow;
                end   
                f2 = f;
                %f_diff = norm(f1-f2,'fro')/norm(f2,'fro');
end
end
