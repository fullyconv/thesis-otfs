%%CreateData
datasize=101;
data=randn(datasize,datasize);
cfarsize=31;
cfargapsize=5;


x=linspace(-10,10,datasize);

coord=[2,5;
    8,2;
    -2,-2;
    -2,8;
    2,-8];

for i=1:size(coord,1)
    mysincx=sin(x-coord(i,1))./(x-coord(i,1));
    mysincy=sin(x-coord(i,2))./(x-coord(i,2));

    mysincx(isnan(mysincx))=1;
    mysincy(isnan(mysincy))=1;
    targetsinc=5*mysincx'*mysincy;
    data=data+targetsinc;
    % figure
    % surf(filtersinc*10)

end
% surf(data)

%%
myfilter=ones(cfarsize);

myfilter((cfarsize+1)/2-cfargapsize:(cfarsize+1)/2+cfargapsize,(cfarsize+1)/2-cfargapsize:(cfarsize+1)/2+cfargapsize)=0;
myfilter=myfilter/sum(myfilter,"all");
filtereddata=filter2(myfilter,data,"same");
detections=(data>(filtereddata+4));

figure
surf(x, x, data, 'EdgeColor', 'none');
hold on

% Extract coordinates of detections
[row, col] = find(detections);
plot3(x(col), x(row), data(sub2ind(size(data), row, col)), 'r.', 'MarkerSize', 15);

xlabel('X'); ylabel('Y'); zlabel('Amplitude');
title('CFAR Detections');
hold off

%%
% Get detection indices
[row, col] = find(detections);

% Convert indices to actual coordinate values
x_detect = x(col);
y_detect = x(row);

% Combine into coordinate pairs
detection_coords = [x_detect(:), y_detect(:)];

% idx = dbscan(detection_coords,1,5)
%%
%% Inputs
XY = detection_coords(:,1:2);      % use (x,y) detections

%% 1) Run DBSCAN (on standardized data for scale robustness)
[XYz, muZ, sigZ] = zscore(XY);     % standardize features
minPts = max(5, round(0.01*size(XY,1)));   % heuristic; tweak if needed

% k-distances to choose eps automatically
k = minPts;
[~, Dk] = knnsearch(XYz, XYz, 'K', k);
kdist = sort(Dk(:,end));                 % each point's distance to its k-th NN
eps = prctile(kdist, 90);                % heuristic eps (try 85–95)

labels = dbscan(XYz, eps, minPts);

%% 2) Build GMM init params from DBSCAN clusters (ignore noise label = -1)
clus = setdiff(unique(labels), -1);
K = numel(clus);

if K == 0
    error('DBSCAN found no clusters. Try reducing minPts or increasing eps.');
end

mu0 = zeros(K,2);
Sigma0 = zeros(2,2,K);
p0 = zeros(1,K);

N = size(XY,1);
for i = 1:K
    idx = labels == clus(i);
    Xi = XY(idx,:);                          % back on ORIGINAL scale
    mu0(i,:) = mean(Xi,1);
    Ci = cov(Xi);
    if any(isnan(Ci),'all') || rank(Ci) < 2
        Ci = eye(2)*1e-4;                    % safety for tiny clusters
    end
    Sigma0(:,:,i) = Ci + 1e-6*eye(2);        % regularize
    p0(i) = nnz(idx)/N;
end

startStruct = struct('mu', mu0, 'Sigma', Sigma0, 'ComponentProportion', p0);

%% 3) Fit GMM using DBSCAN init
GMModel = fitgmdist(XY, K, ...
    'Start', startStruct, ...
    'RegularizationValue', 1e-6, ...
    'Options', statset('MaxIter', 1000));

%% 4) Quick viz: DBSCAN clusters + GMM ellipses
figure  
imagesc(unique(XY(:,1)), unique(XY(:,2)), []); % empty heat backdrop
axis xy; hold on; colormap(gray);               

% color by DBSCAN label
cols = lines(max(K,1));
for i = 1:K
    plot(XY(labels==clus(i),1), XY(labels==clus(i),2), '.', 'Color', cols(i,:), 'MarkerSize', 12);
end
plot(XY(labels==-1,1), XY(labels==-1,2), 'k.', 'MarkerSize', 8); % noise (optional)

% draw 68% and 95% ellipses of fitted GMM
theta = linspace(0,2*pi,200);
Ucirc = [cos(theta); sin(theta)];
for k = 1:K
    mu = GMModel.mu(k,:); S = GMModel.Sigma(:,:,k);
    [U,Sv] = svd(S);
    for conf = [0.68, 0.95]
        r = sqrt(chi2inv(conf,2));
        E = (U*sqrt(Sv))*(r*Ucirc); pts = (E' + mu);
        plot(pts(:,1), pts(:,2));
    end
end
title(sprintf('DBSCAN→GMM init (K=%d), eps=%.3g, minPts=%d', K, eps, minPts));
xlabel('X'); ylabel('Y'); hold off;
