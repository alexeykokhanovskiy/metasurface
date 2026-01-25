%% Reticolo: reflection of Air / GST (200 nm) / SiO2
% All lengths in micrometers (µm)

clear; clc;

% -----------------------------
% 1. Wavelength grid (µm)
% -----------------------------
lambda = linspace(1, 4.0, 401);  
nPoints = numel(lambda);

% -----------------------------
% 2. Your n,k data (example placeholders)
%    -> replace with your data & interpolation
% -----------------------------
% Example: you have experimental data at lambda_exp (µm)
% lambda_exp_GST, n_GST_exp, k_GST_exp, etc.
% Here I just put stupid constants to make the code runnable.
% lambda_exp_GST  = [2 3];           % µm
% n_GST_exp       = [4.0 4.1];       % example
% k_GST_exp       = [0.2 0.25];      % example
% 
% lambda_exp_SiO2 = [2 3];           % µm
% n_SiO2_exp      = [1.44 1.44];     % almost dispersionless in NIR
% k_SiO2_exp      = [0 0];           % negligible
% 
% % Interpolate to simulation wavelengths
% n_GST = interp1(lambda_exp_GST,  n_GST_exp,  lambda, 'pchip');
% k_GST = interp1(lambda_exp_GST,  k_GST_exp,  lambda, 'pchip');
% 
% n_SiO2 = interp1(lambda_exp_SiO2, n_SiO2_exp, lambda, 'pchip');
% k_SiO2 = interp1(lambda_exp_SiO2, k_SiO2_exp, lambda, 'pchip');


am_gsst_nk = readmatrix('./a_gst_frantz_pr.txt');
cr_gsst_nk = readmatrix('./c_gst_frantz_pr.txt');
sio2_nk = readmatrix("malitson_sio2_n");

n_GSTa = interp1(am_gsst_nk(:,1),am_gsst_nk(:,2), lambda);
k_GSTa = interp1(am_gsst_nk(:,3),am_gsst_nk(:,4), lambda);

n_GSTc = interp1(cr_gsst_nk(:,1),cr_gsst_nk(:,2), lambda);
k_GSTc = interp1(cr_gsst_nk(:,3),cr_gsst_nk(:,4), lambda);

n_SiO2 = interp1(sio2_nk(:,1),sio2_nk(:,2), lambda);

% Complex indices
n_air   = 1.0 + 0i;
n_GSTa_c = n_GSTa + 1i*k_GSTa;
n_GSTc_c = n_GSTc + 1i*k_GSTc;
n_SiO2_c= n_SiO2;

% -----------------------------
% 3. Reticolo: basic parameters
% -----------------------------
period = 1.0;      % arbitrary (µm) – for homogeneous stack it doesn’t matter
nn     = 1;        % only zero-order Fourier harmonic (thin film stack)
k_parallel = 0;    % normal incidence

% Polarization: TE (1) or TM (-1); for normal incidence they are identical
parm = res0(1);    % TE

% GST thickness
d_GSTa = 0.15;     % µm  
d_GSTc = 0.05;     % µm  

% Allocate arrays for reflection and transmission
R = zeros(1, nPoints);   % reflectance
T = zeros(1, nPoints);   % transmittance

% -----------------------------
% 4. Wavelength loop
% -----------------------------
for il = 1:nPoints

    wl = lambda(il);   % current wavelength (µm)

    % --- 4.1 Define textures (homogeneous in x) ---
    % textures{j} = {n_j}; here: 1=air (top), 2=GST, 3=SiO2 (bottom)
    textures = cell(1,3);
    textures{1} = {n_air};          % superstrate
    textures{2} = {n_GSTc_c(il)};    % GST film
    textures{3} = {n_GSTa_c(il)};    % GST film
    textures{4} = {n_SiO2_c(il)};   % substrate

    % --- 4.2 Define layer stack (Profile) ---
    % Profile = {[thicknesses ...], [textureIndices ...]}
    % From top to bottom: Air (0), GST (d_GST), SiO2 (0)
    Profile = {[0, d_GSTc*0, d_GSTa*0+0.2, 0], [1, 2, 3, 4]};

    % --- 4.3 Eigenmodes for each texture at this wavelength ---
    aa = res1(wl, period, textures, nn, k_parallel, parm);

    % --- 4.4 Diffraction solution (thin-film reflection / transmission) ---
    result = res2(aa, Profile);

    % --- 4.5 Extract 0th-order reflection & transmission (incident from top) ---
    % Only order m = 0 is propagating in a homogeneous stack
    R(il) = result.inc_top_reflected.efficiency{0};    % reflectance
    T(il) = result.inc_top_transmitted.efficiency{0};  % transmittance

end

% -----------------------------
% 5. Plot spectrum
% -----------------------------
% figure;
plot(lambda, R, 'LineWidth', 1.5); 
hold on;
plot(lambda, T, '--', 'LineWidth', 1.5);
xlabel('\lambda (\mum)');
ylabel('Efficiency');
legend('R', 'T', 'Location', 'best');
grid on;
title('Air / GST(200 nm) / SiO_2  at normal incidence');
