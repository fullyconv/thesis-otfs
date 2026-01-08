
X=randn(10,20);
Y=randn(20,30);

% X is a-by-b, Y is b-by-c
[a,b] = size(X);
[b2,c] = size(Y);
assert(b==b2, 'Inner dims must match: size(X,2) == size(Y,1)');

% Z = reshape(X, a, b, 1) .* reshape(Y, 1, b, c);   % Z is a-by-b-by-c
Z = X .* permute(Y, [3 1 2]);   % X: a×b×1, permute(Y): 1×b×c  => Z: a×b×c
