function [XNorm, mu, stddev] = featureNormalize(X)
% This function provides feature normalization by taking in the input X and
% calculating the normalized inputs along with the mean and standard
% deviation for each feature.
% X = (m x d) dimensions
% mean = (1 x d) 
% stddev = (1 x d)

% Declare variables
XNorm = X;
mu = zeros(1, size(X, 2));
stddev = zeros(1, size(X, 2));

% Calculates mean and std dev for each feature
for i=1:size(mu,2)
    mu(1,i) = mean(X(:,i)); 
    stddev(1,i) = std(X(:,i));
    XNorm(:,i) = (X(:,i)-mu(1,i))/stddev(1,i);
    XNorm(:,i)= (X(:,i)-mu(1,i));
end