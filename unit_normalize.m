function normX = unit_normalize(X)

normX = zeros(size(X));
for i = 1:size(X,1)
    normX(i,:) = X(i,:)/norm(X(i,:));
end

end
