%%%% Eigenfaces Demo
%% Generate the eigenfaces and run a demo.
clear ; close all; clc

fprintf('*** Eigenfaces ***\n');
fflush(stdout);

%% Loading and Visualizing Face Data
fprintf('Loading face dataset.\n\n');
fflush(stdout);
load('faces.mat')

n = 100;
fprintf('Visualizing first %d faces.\n\n', n);
fflush(stdout);
[plt, img] = displayImages(X(1:n, :));
saveas(plt, 'bin/faces_original.jpg')

%% PCA on Faces -> Eigenfaces
% Run PCA and visualize the eigenvectors which are in this case eigenfaces
fprintf('Running PCA on face dataset.\n\n');
fflush(stdout);

% Before running PCA, it is important to first normalize X 
[X_norm, mu, sigma] = normalizeFeatures(X);

[U, S] = pca(X_norm);

n = 36;
printf('Visualizing first %d eigenfaces.\n\n', n);
fflush(stdout);
[plt, img] = displayImages(U(:, 1:n)');
saveas(plt, 'bin/eigenfaces.jpg')

%% Dimension Reduction for Faces
% Project images to the eigen space using the top k eigenvectors 
fprintf('Dimension reduction for face dataset.\n\n');
fflush(stdout);

k = 100;
Z = X_norm * U(:, 1:k);

%% Recovery of Faces
% Recover the faces using the reduced encodings
fprintf('Visualizing the projected (reduced dimension) faces.\n\n');
fflush(stdout);

k = 100;
X_rec = Z * U(:,1:k)';

% Un-normalize recovered data
X_rec = bsxfun(@times, X_rec, sigma);
X_rec = bsxfun(@plus, X_rec, mu);

n = 100;
printf('Visualizing first %d recovered faces.\n\n', n);
fflush(stdout);
[plt, img] = displayImages(X_rec(1:n,:));
saveas(plt, 'bin/faces_recovered.jpg')

% Display the original faces and the recovered faces side-by-side
subplot(1, 2, 1);
[plt, img] = displayImages(X(1:n,:));
title('Original faces');
axis square;
subplot(1, 2, 2);
[plt, img] = displayImages(X_rec(1:n,:));
title('Recovered faces');
axis square;
saveas(plt, 'bin/faces_comparison.jpg')

close all

