function [maximum pred ac perclass C] = predict(W,X,Y)
numclass = length(unique(Y));
C = zeros(numclass);
%prediction part
[maximum pred] = max(sigmoid(X*W),[],2);
% for i=1:length(Y)
% 	C(Y(i),pred(i)) = C(Y(i),pred(i)) + 1;
% end
C = confusionmat(Y,pred);
perclass = diag(C);
ac = sum(pred == Y)/length(Y);
end
