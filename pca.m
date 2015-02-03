function [U, L] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, L] = pca(X) returns the eigenvectors U and the eigenvalues 
%   (on diagonal) in L of the covariance matrix of X
%

% Need number of examples
m = size(X,1);

% Covariance matrix "Sigma"
Sigma = (1/m) * X' * X;

% Compute the eigenvectors/eigenvalues of the covariance matrix
% Note: the results may not be ordered, so order in descending order
[U, L] = eig(Sigma); 
[L, ind] = sort(diag(L),"descend");
U = U(:,ind);
L = diag(L);

% NOTE: Could use Singular Value Decomposition instead of the `eig` function
%   as follows (leaving code here and commented out for educational purposes):
%[U,S,V] = svd(Sigma);

end
