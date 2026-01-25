%% RETICOLO-2D: Air / GST cross (50 nm) / GST film (200 nm) / SiO2
% Period: 4 x 4 µm
% Normal incidence from air
% Spectrum: 2–3 µm
%
% Cross = two orthogonal "stadium" arm
% 
% 
% 
% s (rectangle + 2 circular end caps)
% Arms are 50 nm thick, made of GST, on top of a 200 nm uniform GST film.
function [R_TE, T_TE] = r_t_calc(wl_min, wl_max, h_cross)
    % clear all; clc;
    folderToAdd = 'C:\Users\alexey\YandexDisk\Projects\metasurface_spectral_filters\signal_processing_PCM_metasurfaces\V10_2025\reticolo_allege_v10\';
    addpath(folderToAdd);
    folderToAdd = 'C:\Users\alexey\YandexDisk\Projects\metasurface_spectral_filters\signal_processing_PCM_metasurfaces\V10_2025\results\';
    addpath(folderToAdd);
    h_sio2arr = linspace(0, 0.5, 10);
    for h_sio2 = [0.2]
        %h_sio2arr(2:end)
    
    
    % =========================
    % 1. WAVELENGTH GRID (µm)
    % =========================
    lam_N = 100;
    lambda   = linspace(wl_min, wl_max, lam_N);   % 2–3 µm
    nPoints  = numel(lambda);
    
    dq = parallel.pool.DataQueue;
    % afterEach(dq, @(i) fprintf('lam index finished %d\n', i));
    %afterEach(dq, @count_upd);
    
    
    % =========================
    % 2. DISPERSION (PLACEHOLDERS!)
    %    >>> Replace with your real n(λ), k(λ) data <<<
    % =========================
    
    am_gsst_nk = readmatrix('C:\Users\alexey\YandexDisk\Projects\metasurface_spectral_filters\signal_processing_PCM_metasurfaces\V10_2025\results\new_antennas\a_gst_frantz_pr.txt');
    cr_gsst_nk = readmatrix('C:\Users\alexey\YandexDisk\Projects\metasurface_spectral_filters\signal_processing_PCM_metasurfaces\V10_2025\results\new_antennas\c_gst_frantz_pr.txt');
    sio2_nk = readmatrix('C:\Users\alexey\YandexDisk\Projects\metasurface_spectral_filters\signal_processing_PCM_metasurfaces\V10_2025\results\new_antennas\malitson_sio2_n.txt');
    
    n_GSTa = interp1(am_gsst_nk(:,1),am_gsst_nk(:,2), lambda);
    k_GSTa = interp1(am_gsst_nk(:,3),am_gsst_nk(:,4), lambda);
    
    n_GSTc = interp1(cr_gsst_nk(:,1),cr_gsst_nk(:,2), lambda);
    k_GSTc = interp1(cr_gsst_nk(:,3),cr_gsst_nk(:,4), lambda);
    
    n_SiO2 = interp1(sio2_nk(:,1), sio2_nk(:,2), lambda);
    
    % Complex indices
    n_air   = 1.0 + 0i;
    n_GSTa_c = n_GSTa + 1i*k_GSTa;
    n_GSTc_c = n_GSTc + 1i*k_GSTc;
    n_SiO2_c= n_SiO2;
    
    % =========================
    % 3. PERIODS & FOURIER ORDERS
    % =========================
    Px     = 1;                   % µm
    Py     = 1;                   % µm
    period = [Px, Py];
    
    % Number of Fourier harmonics [Nx, Ny]:
    % retained orders are m=-Nx..Nx, n=-Ny..Ny
    nn = [15, 15];                    % increase for convergence tests
    
    % Normal incidence (θ = 0 → k_parallel = 0)
    k_parallel  = 0;
    angle_delta = 0;            
    
    % RETICOLO parameter struct
    parm = res0;                    % default parameters
    parm.sym.x = 0;
    parm.sym.y = 0;
    parm.sym.pol = 1;
    
    parm.not_io = 1; 
    % Example tweaks if you like:
    % parm.res1.champ = 1;         % more accurate field calc
    % parm.res1.trace = 1;         % texture plots, etc.
    
    % =========================
    % 4. GEOMETRY: GST CROSS WITH STADIUM ARMS
    % =========================
    % All dimensions in µm
    
    % ---------- Cross geometry (EDIT THESE) ----------
    
    % Vertical arm:
    Ly = Py;        % total length 
    Wy = 0.5;        % total width 
    Lsp = 1.5; 
    % vert_st = 0.2;
    
    % Number of points to polygonize circles (for ellipse primitive)
    Nf = 16;         % Nf >= 8 recommended for smooth fillets
    
    % Thicknesses:
        % h_cross = 0.090; % 50 nm (µm) – patterned GST layer
        h_GST   = 0.23-h_cross; % 200 nm (µm) – underlying uniform GST film
    % h_sio2 = 0.25;
    
    % =========================
    % 5. ALLOCATE REFLECTANCE ARRAYS
    % =========================
    R_TE = zeros(1, nPoints);   % TE, specular (0,0)
    T_TE = zeros(1, nPoints);   % TE, specular (0,0)
    
    R_TE1 = zeros(1, nPoints);   % TE, specular (0,0)
    T_TE1 = zeros(1, nPoints);   % TE, specular (0,0)
    R_TE11 = zeros(1, nPoints);   % TE, specular (0,0)
    T_TE11 = zeros(1, nPoints);   % TE, specular (0,0)
    % R_TM = zeros(1, nPoints);   % TM, specular (0,0)
    
    % =========================
    % 6. MAIN WAVELENGTH LOOP
    % =========================
    parfor il = 1:nPoints
    
        
        wl         = lambda(il)        % current wavelength (µm)
    
        % ----- 6.1: BUILD TEXTURES AT THIS λ -----
        %
        % texture{1}: uniform air (superstrate)
        % texture{2}: GST cross (stadium arms) in air background
        % texture{3}: uniform GST film
        % texture{4}: uniform SiO2 substrate
    
        textures = cell(1,4);
    
        % -- Texture 1: Top (air) --
        textures{1} = { n_air };
    
        % ---------- Stadium Vertical Arm ----------
        R_v  = Wy/2;
        Lc_v = Ly - 2*R_v;             % central rectangle length
        y0_v = Lc_v/2;                 % center position of circles
    
    
        Vert = {
            % central rectangle (vertical bar)
            [0, 0, Wy, Ly, n_GSTc_c(il), 1], ...
        };
    
    %     Vert2 = {
    %         % central rectangle (vertical bar)
    %         [Lsp/2, 0, Wy, Lc_v, n_GSTc_c(il), 1], ...
    %         % bottom circular end cap
    %         [Lsp/2, -y0_v, 2*R_v, 2*R_v, n_GSTc_c(il), Nf], ...
    %         % top circular end cap
    %         [Lsp/2,  y0_v, 2*R_v, 2*R_v, n_GSTc_c(il), Nf]
    %     };
    
        % -- Texture 2: union of both arms in air background --
        textures{2} = [{n_GSTa_c(il)}, Vert{:}];
    
        % -- Texture 3: uniform GST film --
        textures{3} = { n_GSTa_c(il) };
    
        % -- Texture 4: uniform SiO2 substrate --
        textures{4} = { n_SiO2_c(il) };
    
        % (Optional) visualize textures once for debugging:
    %     if il == 1
    %         parm.res1.trace = 1;
    %     end
    
        % ----- 6.2: DEFINE LAYER STACK (PROFILE) -----
        % From top to bottom:
        %   Layer 1: thickness 0        → texture 1 (air superstrate)
        %   Layer 2: thickness h_cross  → texture 2 (GST cross in aGST)
        %   Layer 3: thickness h_GST    → texture 3 (uniform GST film)
        %   Layer 4: thickness 0        → texture 4 (SiO2 substrate)
    
        Profile = { [0, h_sio2, h_cross, h_GST, 0], [1, 4, 2, 3, 4] };
    
        % ----- 6.3: EIGENMODES FOR ALL TEXTURES -----
        aa = res1(wl, period, textures, nn, k_parallel, angle_delta, parm);
    
        % ----- 6.4: DIFFRACTION SOLUTION -----
        two_D = res2(aa, Profile);
    
        % ----- 6.5: SPECULAR REFLECTION (m=0, n=0) -----
        % TE incidence from top:
        R_TE(1,il) = sum(two_D.TEinc_top_reflected.efficiency);
        T_TE(1,il) = sum(two_D.TEinc_top_transmitted.efficiency);  % transmittance
        R_TE1(1,il) = two_D.TEinc_top_reflected.efficiency{1,0};
        T_TE1(1,il) = two_D.TEinc_top_transmitted.efficiency{1, 0};  % transmittance
        R_TE11(1,il) = two_D.TEinc_top_reflected.efficiency{1,1};
        T_TE11(1,il) = two_D.TEinc_top_transmitted.efficiency{1, 1};  % transmittance
    %     % TM incidence from top:
    %     R_TM(il) = two_D.TMinc_top_reflected.efficiency{0,0};
        send(dq, il);
    end
    
    % % Unpolarized specular reflectance:
    % % R_unpol = 0.5*(R_TE + R_TM);
    % 
    % % =========================
    % % 7. PLOT SPECTRA
    % % =========================
    % % figure; 
    % plot(lambda, R_TE, 'LineWidth', 1.5);
    % hold on;
    % plot(lambda, T_TE,'--', 'LineWidth', 1.5);
    % % plot(lambda, R_TM, '--', 'LineWidth', 1.5);
    % % plot(lambda, R_unpol, '-.', 'LineWidth', 1.5);
    % xlabel('\lambda (\mum)');
    % ylabel('R, T');
    % legend('R', 'T', 'Location', 'best');
    % % legend('TE', 'TM', 'Unpolarized', 'Location', 'best');
    % % grid on;
    % xlim([min(lambda), max(lambda)])
    % ylim([0, 0.8])
    
    % save(sprintf("results/R_TE_sio2h_%.0f", h_sio2*1000), 'R_TE')
    % save(sprintf("results/T_TE_sio2h_%.0f", h_sio2*1000), 'T_TE')
    % retio;
    end
end 

function count_upd(il)
    persistent doneit
    if isempty(doneit)
        doneit = 0;
    end
    doneit = doneit+1;
    Nit = evalin('base', 'lam_N');
    fprintf('completed %3d / %3d \n', doneit, Nit)

end
