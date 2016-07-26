function [J J1 J2 J3] = regression_loss(W,beta,X,Y,idx,t,param,type)
sf = param.sf;
sigma = param.sigma;
T = length(idx.tr)-1;
d = size(X,2);
if strcmp(type,'stl')
    m_t = length(Y(idx.tr(t):idx.tr(t+1)-1));
    %m_t = size(X,1);
    %J = 1/(2*m_t) * (norm( Y(idx.tr(t):idx.tr(t+1)-1) - X(idx.tr(t):idx.tr(t+1)-1,:)*W(:,t),2)^2 +param.lambda_stl/2 * W(:,t)'*W(:,t));
    J = 1/(2*m_t) * (norm( Y(idx.tr(t):idx.tr(t+1)-1) - X(idx.tr(t):idx.tr(t+1)-1,:)*W(:,t),2)^2);
end

if strcmp(type,'amtl')
%     delta = zeros(T,1);
%     for t=1:T;
%         delta(t) = regression_loss(W,0,X,Y,idx,t,param,'stl');
%     end
%     lambda = param.lambda; %lambda2 = param.lambda2;
%     J1 = lambda*norm(diag(delta)*beta,'fro')^2/T; %fobj
%     J2 = sum(delta)/T; %fobj2
%     J3 = 1/(2*T) * norm(W-W*beta,'fro')^2; %freg
%     J = J1 + J2 + J3;

    delta = zeros(T,1);
    J1=0;
    J2 = 0;
    J3 = 0;
    J = 0;
    for t = 1:T;
        n_t = idx.tr(t+1)-idx.tr(t);
        delta(t) = regression_loss(W,0,X,Y,idx,t,param,'stl');
        J1 = J1 + (1+param.c_t(t)*norm(beta(t,:),1))*delta(t);
    end
    J1 = J1;
    J2 = param.lambda*norm( W-W*beta , 'fro')^2;
    J = J1+J2;
end

end