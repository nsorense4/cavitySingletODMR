clear
close all



%% Plot normalized transmission sweeps

filenames = {'4.2_0_1035_1050_5nmps.mat'};

f0Arr = [4.214, 12.6, 18.26];
QArr = [50, 30, 100];
color = {'k', [11,246,0]/255}; %[255,48,0]/255

filenames = {'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsResonanceSweeps/DNV_B14_QG3/4.2um_0/20260202/1040_1042nm_1nmps_averaged_norm.mat'};

f0Arr = [4.214, 12.6, 18.26];
QArr = [50, 30, 100];
f0Arr = [160];
QArr = [5];

Q = zeros([length(filenames), 2]);
N = length(filenames);

for i = 1:N
    data = load([filenames{i}]);
    T = data.TOut;
    W = data.W;
    % figure(10);
    % hold on
    % plot(W,T)
    % xlabel("wavelength, \lambda (nm)")
    % ylabel("transmission, T (au)")
    % ylim([0.65,1])
    % xlim([1047.9, 1048.075])

    
    lowerBoundS = [1040, 1e3, 5e2, 0.8, -inf, -inf];
    upperBoundS = [151042, 2e5, 1e4,1.1, inf, inf];

    [paramNames, params, ~, ~, ~, ~, ~] = fittingBroadSweeps(W, T, ...
        struct('plotFit', true, 'liveplot', true, ...
        'checkFinalFit',true, 'assumeThermoBool',false,'lowerBound', lowerBoundS, 'upperBound', upperBoundS, 'autofit', true, 'dataWidthFactorScalar', 3));


end



%% Define functions
function TOut = normData(T, W, WNorm, TNorm, NormOffset)
    TNorm = interp1(WNorm, TNorm, W, 'linear', 'extrap');
    
    TOut = (T - NormOffset)./(TNorm - NormOffset);
    TOut = TOut/max(TOut);
end

function [X, Y, Xs] = custFFT(x, y)
    N = length(x);
    xSamp = x(2) - x(1);
    Xs = 1/xSamp;
    X = Xs/N*(0:N-1);
    X = X(1:N/2+1);
    Y = fft(y);
    Y = Y(1:N/2+1);
    Y(2:end-1) = 2*Y(2:end-1);

end
    