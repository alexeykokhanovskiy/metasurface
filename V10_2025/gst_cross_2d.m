%% RETICOLO-2D: Air / GST cross (50 nm) / GST film (200 nm) / SiO2
% Normal incidence from air, 2–3 µm, period 4 x 4 µm

clear; clc;

% -----------------------------
% 1. Wavelength grid (µm)
% -----------------------------
lambda = linspace(2.0, 3.0, 301);   % 2–3 µm
nPoints = numel(lambda);

% -----------------------------
% 2. Dispersion: n(λ), k(λ)
%    >>> Replace the example data with your own <<<
% -----------------------------
% Example placeholder data (constant indices)
lambda_exp_GST  = [2.0 3.0];        % µm
n_GST_exp       = [4.0 4.0];        % example
k_GST_exp       = [0.2 0.2];        % example

lambda_exp_SiO2 = [2.0 3.0];        % µm
n_SiO2_exp      = [1.44 1.44];      % SiO2 ~ const
k_SiO2_exp      = [0    0   ];

% Interpolate to simulation λ grid
n_GST  = interp1(lambda_exp_GST,  n_GST_exp,  lambda, 'pchip');
k_GST  = interp1(lambda_exp_GST,  k_GST_exp,  lambda, 'pchip');
n_SiO2 = interp1(lambda_exp_SiO2, n_SiO2_exp, lambda, 'pchip');
k_SiO2 = interp1(lambda_exp_SiO2, k_SiO2_exp, lambda, 'pchip');

n_air   = 1.0;
n_GST_c = n_GST  + 1i*k_GST;
n_SiO2_c= n_SiO2 + 1i*k_SiO2;

% -----------------------------
% 3. Periods & Fourier orders
% -----------------------------
Px = 4.0;   % µm
Py = 4.0;   % µm
period = [Px, Py];

% Number of Fourier harmonics in x and y
%  nn = [Nx, Ny] => (2*Nx+1) x (2*Ny+1) plane waves
nn = [5, 5];        % start with 5–7; increase for convergence tests

% Normal incidence: k_parallel = 0, azimuth δ = 0
k_parallel  = 0;
angle_delta = 0;

% Set base polarisation (TE vs TM; for normal incidence it doesn’t matter)
parm = res0(1);     % TE

parm=res0;         % default parameters for "parm" 
parm.res1.champ=1; % the eletromagnetic field is calculated accurately 


% Optional: use symmetries (cross centered at (0,0))
parm.sym.x   = 0;
parm.sym.y   = 0;
parm.sym.pol = 0;   % let code decide

% -----------------------------
% 4. Geometry of GST cross (in-plane)
% -----------------------------
% Arms are centered at (0,0)
L_arm = 1.5;   % total arm length (µm)  -> along x or y
W_arm = 0.5;   % arm width (µm)
R = 0.1;

% GST layer thicknesses
h_cross = 0.050;   % 50 nm  (µm)
h_GST   = 0.200;   % 200 nm (µm) underlying film

% -----------------------------
% 5. Allocate reflectance arrays
% -----------------------------
R_TE = zeros(1, nPoints);   % specular 0,0 order, TE incidence
R_TM = zeros(1, nPoints);   % specular 0,0 order, TM incidence

% -----------------------------
% 6. Wavelength loop
% -----------------------------
for il = 1:nPoints

    wl = lambda(il);          % current wavelength (µm)

    % --- 6.1 Define textures at this λ ---
    %
    % texture{1}: uniform air (superstrate)
    % texture{2}: cross-shaped GST in air background (grating layer)
    % texture{3}: uniform GST film
    % texture{4}: uniform SiO2 substrate

    textures = cell(1,4);

    % Top: air
    textures{1} = { n_air };

    % Cross layer: background air, two GST rectangles forming a cross
    % Rectangle: [cx, cy, Lx, Ly, ninclusion, shapeFlag]
    % shapeFlag = 1 => rectangle (see Reticolo doc)
%     textures{2} = { n_air, ...
%         [0, 0, L_arm, W_arm, n_GST_c(il), 1], ...   % horizontal arm
%         [0, 0, W_arm, L_arm, n_GST_c(il), 1]  ...   % vertical arm
%         };

    textures{2} = {n_air,[0, 0, L_arm, W_arm, n_GST_c(il), 1],[ L_arm/2,  W_arm/2,  2*R, 2*R, n_GST_c(il), 2],[ L_arm/2, -W_arm/2,  2*R, 2*R, n_GST_c(il), 2], ...
        [-L_arm/2,  W_arm/2,  2*R, 2*R, n_GST_c(il), 2], ...
        [-L_arm/2, -W_arm/2,  2*R, 2*R, n_GST_c(il), 2],...

        % Vertical arm (rounded rectangle)
        [0, 0, W_arm, L_arm, n_GST_c(il), 1],...
        [ L_arm/2,  W_arm/2,  2*R, 2*R, n_GST_c(il), 2], ...
        [ L_arm/2, -W_arm/2,  2*R, 2*R, n_GST_c(il), 2], ...
        [-L_arm/2,  W_arm/2,  2*R, 2*R, n_GST_c(il), 2], ...
        [-L_arm/2, -W_arm/2,  2*R, 2*R, n_GST_c(il), 2]};

    % Uniform GST film
    textures{3} = { 2 };

    % Substrate: SiO2
    textures{4} = { 1};

%     Optional: quick visual check of textures when debugging
    if il == 1
        parm.res1.trace = 1;
    end

    % --- 6.2 Define stack (Profile) ---
    % From top to bottom:
    %   Layer 1: thickness 0, texture 1 (air superstrate)
    %   Layer 2: thickness h_cross, texture 2 (GST cross in air)
    %   Layer 3: thickness h_GST, texture 3 (uniform GST film)
    %   Layer 4: thickness 0, texture 4 (SiO2 substrate)

    Profile = { [0, h_cross, h_GST, 0], [1, 2, 3, 4] };

    % --- 6.3 Eigenmodes for all textures ---
    aa = res1(wl, period, textures, nn, k_parallel, angle_delta, parm);

    % --- 6.4 Diffraction solution ---
    two_D = res2(aa, Profile);

    % --- 6.5 Extract specular reflection (order (0,0)) ---

    % TE incidence from top
    R_TE(il) = two_D.TEinc_top_reflected.efficiency{0,0};

    % TM incidence from top
    R_TM(il) = two_D.TMinc_top_reflected.efficiency{0,0};

end

% Unpolarized reflectance (optional)
R_unpol = 0.5*(R_TE + R_TM);

% -----------------------------
% 7. Plot spectra
% -----------------------------
figure; hold on;
plot(lambda, R_TE, 'LineWidth', 1.5);
plot(lambda, R_TM, '--', 'LineWidth', 1.5);
plot(lambda, R_unpol, '-.', 'LineWidth', 1.5);
xlabel('\lambda (\mum)');
ylabel('Specular reflection R_{0,0}');
legend('TE', 'TM', 'Unpolarized', 'Location', 'best');
grid on;
title('Reflection of GST-cross metasurface (normal incidence from air)');
