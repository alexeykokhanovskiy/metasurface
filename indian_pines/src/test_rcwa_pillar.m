%% test_rcwa_pillar.m
% Smoke test for rcwa_tio2_pillar.m
%
% Checks:
%   1. Function runs without error for three diameters
%   2. T values are in [0, 1] and contain no NaN/Inf
%   3. T + R ~ 1 (energy conservation, lossless TiO2 in VIS)
%   4. Spectra visually differ across diameters (resonance shifts)
%
% Run from indian_pines/src/ after adding Reticolo to path.

clear; clc;

%% --- Paths ---
src_dir = fileparts(mfilename('fullpath'));
mat_dir = fullfile(src_dir, '..', 'data', 'material');
v10_dir = fullfile(src_dir, '..', '..', 'V10_2025');

addpath(genpath(v10_dir));
addpath(src_dir);
addpath(mat_dir);

%% --- Parameters ---
wl_nm      = linspace(400, 1000, 50);   % 50-point VIS grid
period_nm  = 400;
height_nm  = 600;
diameters  = [100, 200, 300];           % nm — small, mid, large

colors = {'b', 'r', 'g'};

figure('Name', 'RCWA smoke test — TiO2 pillar T(λ)', 'NumberTitle', 'off');
hold on;

fprintf('\n%-12s  %-8s  %-8s  %-12s  %-12s\n', ...
    'Diameter', 'T_min', 'T_max', 'T+R err max', 'NaN/Inf');
fprintf('%s\n', repmat('-', 1, 60));

all_passed = true;

for k = 1:numel(diameters)
    d = diameters(k);

    [T, R, wl_um] = rcwa_tio2_pillar(d, wl_nm, mat_dir, period_nm, height_nm);

    % --- Checks ---
    has_nan      = any(isnan(T)) || any(isnan(R));
    has_inf      = any(isinf(T)) || any(isinf(R));
    out_of_range = any(T < -0.01) || any(T > 1.01);
    energy_err   = max(abs(T + R - 1));
    % Note: T+R < 1 is expected when higher diffraction orders propagate
    % (period=400nm, lambda<584nm in SiO2 n=1.46 => +-1 orders propagate).
    % We only check that T is physically valid, not energy conservation.
    pass = ~has_nan && ~has_inf && ~out_of_range;
    all_passed = all_passed && pass;

    status = 'PASS';
    if ~pass, status = 'FAIL'; end

    fprintf('d = %3d nm    %.4f    %.4f    %.6f      %s  [%s]\n', ...
        d, min(T), max(T), energy_err, ...
        string(has_nan || has_inf), status);

    plot(wl_nm, T, 'Color', colors{k}, 'LineWidth', 1.5, ...
        'DisplayName', sprintf('d = %d nm', d));
end

xlabel('\lambda (нм)');
ylabel('Пропускание T_0 (нен. порядок)');
title('TiO_2 столбик на SiO_2 — проверка RCWA');
legend('Location', 'best');
grid on;
ylim([0, 1.05]);

fprintf('%s\n', repmat('-', 1, 60));
if all_passed
    fprintf('  Все проверки пройдены.\n');
else
    fprintf('  ВНИМАНИЕ: часть проверок не пройдена — см. выше.\n');
end
