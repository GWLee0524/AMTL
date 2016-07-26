function [J J1 J2] = cal_loss(W,W_old,beta,X,Y,lambda1, lambda2, type,param)
sf = param.sf;
sigma = param.sigma;
lambda = param.lambda;
%calculate the loss
J=0;
J1=0;
J2 = 0;
T = length(unique(Y));
% if strcmp(type,'ridge')
%     for t = 1:T;
%         J_l = J_l+norm(beta(:,t),2)^2 * general_loss(W,X,Y,beta,0,t,'atml_general'); 
%     end
%     
%     J = J_l + lambda1 * norm( W-W_old*beta,2)^2;
% end
delta = zeros(T,1);
if strcmp(type,'ridge')
%     for t = 1:T;
%         J_l = J_l+(diag(delta) * beta(:,t) )' * (diag(delta) * beta(:,t) ) ;
%     end
		for t=1:T
            idx = find(Y == t);
    		delta(t) =  sqrt(general_loss(W,X(idx,:),Y(idx),0,lambda1,t,'amtl_general'));
        end
		%delta
		J1 = lambda1 * ( trace ( (diag(delta) * beta )'*(diag(delta)*beta) ) + sum(delta) );
		J2 = norm(W-W_old*beta,2)^2;
    J = J1 + J2;
end

if strcmp(type,'lasso')
    
    for t=1:T;
           m_t = sum(Y==t);
        delta(t) =  general_loss(W,X,Y,0,lambda1,t,'amtl_general');
	J1 = J1+delta(t)*( param.c_t(t)*norm(beta(t,:),1) +1);
    end
		%J1 = lambda1 * ( sum(sum(abs(diag(delta) * beta ))) + 1*sum(delta)) ;
	J1 = J1;	
	J2 = lambda* norm( W-W_old*beta,'fro')^2;
    J = J1 + J2;
end

if strcmp(type,'regression')
    for l = 1:T;
        delta(l) = general_loss(W,X,Y,0,lambda1,l,'regression') ;
    end
    J = 1/2* norm( W-W_old*beta,2)^2 + T*lambda1 * (sum(sum(abs(diag(delta) * beta))) + sum(delta));
end

if strcmp(type,'prox')
    for l = 1:T;
        delta(l) = sqrt(general_loss(W,X,Y,0,0,l,'amtl_general')) ;
    end
    J1 = lambda1 * norm(diag(delta) *beta,'fro') + sum(delta) + lambda2 * sum(sum(abs(beta)));
    J2 = 1/2 * norm(W-W_old*beta,'fro')^2;
    J = J1+J2;
end


end
