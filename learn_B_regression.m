function B = learn_B_regression(W,delta,param)

maxiter = 500;
eval_interval = maxiter / 20;
T = size(W,2);
if (isfield(param,'B'))
	B = param.B;
else
	B = ones(T)/T;
end
G = ones(T);
normG = Inf;
iter = 0;
alpha = 0.000001; %0.000001
maxlsiter = 100;
tau = 0.5;
b = 0.01;
linesearch = 1;

for t=1:T
	B(t,t) = 0;
	nont{t} = 1:T;
	nont{t}(t) = [];
	weight(:,t) = delta(nont{t});
    c_t(:,t) = param.c_t(nont{t});
	y(:,t) = W(:,t);
end
EPS = 0.001;
while (iter <= maxiter && normG > EPS) 
	iter = iter + 1;
	prevG = G;
	for t=1:T
		beta = B(nont{t},t);
		X = W(:,nont{t});
		%g(:,t) = X'*(X*beta-y(:,t)) + param.lambda*weight(:,t).*beta.*weight(:,t);
        %g(:,t) = X'*(X*beta-y(:,t)) + param.lambda*weight(:,t).*(beta>0);
        %g(:,t) = param.lambda*X'*(X*beta-y(:,t)) + weight(:,t).*(beta>0);
        g(:,t) = param.lambda*X'*(X*beta-y(:,t)) + c_t(:,t).*weight(:,t).*beta.*weight(:,t);
        G(nont{t},t) = g(:,t);
	end
	grad = G(:);
	beta = B(:);
	gradnew = zeros(size(beta));
	[maxval, mu] = max(beta);
	gradnew(find(grad-grad(mu) > 0 | beta==0)) = 0;
	idxusual = setdiff(find(beta>0),mu);
	gradnew(idxusual) = grad(idxusual)-grad(mu);
	nonzero = find(grad > 0);
	[minval, idx] = min(beta(nonzero)./grad(nonzero));
	% what if there is no nonzero entry?
	v = nonzero(idx);
	gradnew(mu) = grad(mu)-grad(v);
	grad = gradnew;
	G = reshape(grad,size(G));
	% search for step size using line search
	stepsize = alpha;
	finit = loss(W, B, weight, param.lambda,c_t);
	fnew = loss(W, B-stepsize*G, weight, param.lambda,c_t);
		
	lsiter = 0;
	if (linesearch)
		while (fnew > finit && lsiter < maxlsiter)
			lsiter = lsiter + 1;
			stepsize = tau*stepsize;
			fnew = loss(W, B-stepsize*G, weight, param.lambda,c_t);
		end
	end
	beta = beta - stepsize*grad;
	beta(v) = 0;
	%beta = max(0,beta);
	beta = param.lambda2*(beta./sum(beta));
	%beta = beta./sum(beta);
    beta(find(beta<0)) = 0;
	B = reshape(beta,size(B));	
	
	%beta = max(0,beta);
	%beta = sign(beta).*(max(0,abs(beta) - param.lambda2));
		
	normG = norm(G-prevG,'fro');	
	%beta = beta/sum(abs(beta));
	if (mod(iter, eval_interval) == 1)
		f = fnew;
		fprintf('%d) %2.4f, ||G||=%2.4f, stepsize =%2.10f\n', iter, f, normG, stepsize);
    end
    
end
function f = loss(W, B, weight, lambda,c_t)
T = size(W,2);
f = 0;
for t=1:T
	nont = 1:T;
	nont(t) = [];
	y = W(:,t);
	X = W(:,nont);
	beta = B(nont,t);
	f = f + lambda*norm(X*beta-y,2)^2 + norm(c_t(:,t).*weight(:,t).*beta,1);
end
f = f + sum(weight(:,1)) + weight(1,2);
