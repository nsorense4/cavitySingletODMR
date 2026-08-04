clear
close all


directory = 'C:\Users\nicho\OneDrive - University of Calgary\Documents\gitProjects\dataBarclay\measurementsResonanceSweeps\DNV_B14_QG3\IR_Magnetometry\20250430\';
% green power 
filenameNorm = 'Background_100mA_lowpass1k_gain2e2_1000LP_avg.mat';
data = load([directory, filenameNorm]);
WNorm = data.scanData.W;
TNorm = data.scanData.T;
NoiseOffset = data.scanData.NoiseOffset;

%% Plot normalized transmission sweeps

filenames = {'4.2_0_1035_1050_5nmps.mat'};

f0Arr = [4.214, 12.6, 18.26];
QArr = [50, 30, 100];
color = {'k', [11,246,0]/255}; %[255,48,0]/255


filenames = {'4.2_0_1035_1050_5nmps.mat', '4.2_0_1035_1050_5nmps_greenOn.mat'};
% filenames = {'4.2_0_1048_1050_5nmps_greenOff.mat', '4.2_0_1048_1050_5nmps_greenOn.mat'};

f0Arr = [4.214, 12.6, 18.26];
QArr = [50, 30, 100];
f0Arr = [160];
QArr = [5];
data = load([directory, filenameNorm]);
filterBool = true;

Q = zeros([length(filenames), 2]);
N = length(filenames);

for i = 1:N
    data = load([directory, filenames{i}]);
    T = data.scanData.T;
    W = data.scanData.W;
    TOut = normData(T, W, WNorm, TNorm, NoiseOffset);
    [f, TOutFFT, fs]  = custFFT(W, TOut);
    figure(10);
    hold on
    % plot(W,TOut)
    xlabel("wavelength, \lambda (nm)")
    ylabel("transmission, T (au)")
    ylim([0.65,1])
    xlim([1047.9, 1048.075])

    if filterBool
        % Design notch filter
        TOutFilt = TOut;
        % Apply filter to your signal
        for j = 1:length(f0Arr)
            
            wo = f0Arr(j)/(fs/2);       % Normalized frequency
            bw = wo/QArr(j);            % Bandwidth
            % TOutFilt = bandstop(TOutFilt, [wo-bw, wo+bw]); 
        end
    
        % f_highpass = 15;
        % 
        % hpFilt = designfilt('highpassiir', ...
        % 'FilterOrder', 4, ...
        % 'HalfPowerFrequency', f_highpass, ...
        % 'SampleRate', fs);
        % TOutFilt = filtfilt(hpFilt, TOutFilt);        % Then high-pass

        f_lowpass =100;
        
        lpFilt = designfilt('lowpassiir', ...
            'FilterOrder', 4, ...
            'HalfPowerFrequency', f_lowpass, ...
            'SampleRate', fs);

        TOutFilt = filtfilt(lpFilt, TOutFilt);
        % TOutFilt = lowpass(TOutFilt, f_lowpass/(fs/2));        % Then low-pass
        
        [f, TOutFiltFFT, fs]  = custFFT(W, TOutFilt);
    
        figure(10);
        hold on
        plot(W,TOutFilt, color = color{i}, LineWidth=2)
    
        figure(11);
        semilogx(f, abs(TOutFFT))
        hold on
        semilogx(f, abs(TOutFiltFFT))
        xlabel("f (1/nm)")
        ylabel("|T(f)|")
    end
    [paramNames, params, resnorms, fitType] = fittingBroadSweeps(W, TOutFilt, struct('direction', 0, 'plotFit', false, 'dispBool', false,'contrastLimit', 0.03, 'checkDoubletBool', ...
                    false, 'livePlot', 0, 'filename', '', 'doubletCheckLimit', 1e4));
    Q(i,:) = params(3,:);
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
    