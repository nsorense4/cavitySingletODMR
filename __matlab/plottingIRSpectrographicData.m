clc; close all;

figure;

xlabel('wavelength (nm)')
ylabel('counts (au)')

hold on

wl = MWON10001400100sExposure150umSlitSpectra.Wavelength;
background = MWOFF10001400100sExposure150umSlitSpectraBypassFiberPL.S1/MWOFF10001400100sExposure150umSlitSpectraBypassFiberPL.S1(1)*0.8;
spectrumMWOn = MWON10001400100sExposure150umSlitSpectra.S1/MWON10001400100sExposure150umSlitSpectra.S1(1)-background;
spectrumMWOff = MWOFF10001400100sExposure150umSlitSpectra.S1/MWOFF10001400100sExposure150umSlitSpectra.S1(1)-background;

wl = withLPMWOFF10001700nm66umslit30sintDSon25082025.Wavelength;
background = withLPMWOFF10001700nm66umslit30sintDSon25082025.S1/withLPMWOFF10001700nm66umslit30sintDSon25082025.S1(1)*0 ;
spectrumMWOn = withLPMWON10001700nm66umslit30sintDSon25082025.S1;
spectrumMWOff = withLPMWOFF10001700nm66umslit30sintDSon25082025.S1;

% wl = withLPMWON10001700nm66umslit30sintDSon25082025.Wavelength;
% spectrumMWOn = withLPMWON10001700nm66umslit30sintDSon25082025.S1;
% spectrumMWOff = withLPMWOFF10001700nm66umslit30sintDSon25082025.S1;

plot(wl, spectrumMWOn, 'red')
plot(wl, spectrumMWOff, 'black')

xlim([1000,1700])
% ylim([0, inf])

figure


plot(wl, spectrumMWOn-spectrumMWOff)

xlabel('wavelength (nm)')
ylabel('resonant MW-induced change in counts (au)')
hold on

xlim([1000,1400])
% ylim([0, inf])

%%
clc; close all

dirData = 'C:\Users\nicho\OneDrive - University of Calgary\Documents\gitProjects\dataBarclay\measurementsOpticalSpectra\B14\B14_4p2_IR_Emission_Spectrum\20250903\';

filePattern = fullfile(dirData, '*.dat'); % change extension if needed
files = dir(filePattern);

% Preallocate results (optional, depends on analysis)
results = struct();
N = length(files);
countsTotalNorm = zeros([N, 1]);
countsTotal = zeros([N, 1]);

freqs = linspace(2.856, 2.886, 16);

figure;

xlabel('wavelength (nm)')
ylabel('counts (au)')

hold on

wlBackground = MWOFF10001400100sExposure150umSlitSpectraBypassFiberPL.Wavelength;
background = MWOFF10001400100sExposure150umSlitSpectraBypassFiberPL.S1/MWOFF10001400100sExposure150umSlitSpectraBypassFiberPL.S1(1)*1500-100;
[wlBackground, idx] = unique(wlBackground, 'stable');
background = background(idx);

plot(wlBackground, background)

for k = 1:N
    % Get full file path
    baseFileName = files(k).name;
    fullFileName = fullfile(dirData, baseFileName);
    fprintf('Now processing %s\n', baseFileName);

    % Load data
    dataStruct = importdata(fullFileName);
    
    wl = dataStruct.data(:,1);
    counts = dataStruct.data(:,2);

    background_interp = interp1(wlBackground, background, wl, 'linear', 'extrap');
    % counts = counts - background_interp;
    filtNorm = (wl<=1039)&(wl>=1004);
    filt = (wl>1039);

    countsTotalNorm(k) = sum(counts(filtNorm));
    countsTotal(k) = sum(counts(filt));

    plot(wl, counts)
end


figure

plot(freqs, countsTotal./countsTotalNorm/min(countsTotal./countsTotalNorm), 'r-o')
xlabel('microwave frequency (GHz)')
ylabel('counts (au)')

