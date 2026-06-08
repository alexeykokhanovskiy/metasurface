function [T_unpol, R_unpol, wl_um] = rcwa_tio2_pillar(diameter_nm, wl_query_nm, mat_dir, period_nm, height_nm, nn_order)
% RCWA_TIO2_PILLAR  Zeroth-order transmission of a TiO2 circular nanopillar
%                   on SiO2, normal incidence, unpolarised light.
%
%   [T, R, wl_um] = rcwa_tio2_pillar(diameter_nm, wl_query_nm, mat_dir,
%                                     period_nm, height_nm)
%
%   Inputs
%     diameter_nm  : pillar diameter in nm  (scalar)
%     wl_query_nm  : wavelength vector in nm (e.g. linspace(400,1000,50))
%     mat_dir      : path to folder containing TiO2_ALD_nk.csv and
%                    SiO2_fused_silica_nk.csv
%     period_nm    : square unit cell period in nm  (default 400)
%     height_nm    : pillar height in nm            (default 600)
%
%   Outputs
%     T_unpol      : zeroth-order transmittance, unpolarised (1 x N_wl)
%     R_unpol      : zeroth-order reflectance,   unpolarised (1 x N_wl)
%     wl_um        : wavelength vector in µm

if nargin < 4, period_nm = 400; end
if nargin < 5, height_nm = 600; end
if nargin < 6, nn_order  = 7;   end

% -------------------------------------------------------------------------
% Unit conversion: Reticolo works in microns
% -------------------------------------------------------------------------
wl_um  = wl_query_nm(:)' * 1e-3;   % row vector, µm
P_um   = period_nm  * 1e-3;         % µm
H_um   = height_nm  * 1e-3;         % µm
d_um   = diameter_nm * 1e-3;        % µm

N_wl = numel(wl_um);

% -------------------------------------------------------------------------
% Load dispersive refractive indices
% -------------------------------------------------------------------------
tio2_path = fullfile(mat_dir, 'TiO2_ALD_nk.csv');
sio2_path = fullfile(mat_dir, 'SiO2_fused_silica_nk.csv');

n_tio2 = load_nk(tio2_path, wl_query_nm);   % complex, (N_wl x 1)
n_sio2 = load_nk(sio2_path, wl_query_nm);
n_air  = ones(N_wl, 1);

% -------------------------------------------------------------------------
% Reticolo parameters
% -------------------------------------------------------------------------
parm = res0;             % full default initialisation of all parm fields
parm.res1.champ = 1;     % accurate field calculation
parm.res1.trace = 0;     % suppress verbose output
parm.sym.x   = 0;        % no symmetry reduction
parm.sym.y   = 0;
parm.sym.pol = 0;

period = [P_um, P_um];   % square unit cell
nn     = [nn_order, nn_order];   % Fourier harmonics: (2*nn_order+1)^2 total
k_par  = 0;              % normal incidence
delta  = 0;              % azimuth angle

% -------------------------------------------------------------------------
% Redirect Reticolo's retURD9*.mat cache files to a dedicated subfolder
% -------------------------------------------------------------------------
script_dir = fileparts(mfilename('fullpath'));
cache_dir  = fullfile(script_dir, 'cache');
if ~exist(cache_dir, 'dir'), mkdir(cache_dir); end
orig_dir = pwd;
cd(cache_dir);

% -------------------------------------------------------------------------
% Wavelength loop
% -------------------------------------------------------------------------
T_TE = zeros(1, N_wl);
T_TM = zeros(1, N_wl);
R_TE = zeros(1, N_wl);
R_TM = zeros(1, N_wl);

for il = 1:N_wl
    wl = wl_um(il);

    % Textures (defined at each wavelength due to dispersion)
    % Texture 1: air superstrate (semi-infinite)
    % Texture 2: TiO2 pillar (circle) in air  — shapeFlag 2 = ellipse/circle
    %            [cx, cy, Lx, Ly, n_inclusion, shapeFlag]
    %            Lx = Ly = d  =>  circle of diameter d
    % Texture 3: SiO2 substrate (semi-infinite)
    textures = { ...
        { n_air(il) }, ...
        { n_air(il), [0, 0, d_um, d_um, n_tio2(il), 2] }, ...
        { n_sio2(il) } ...
    };

    % Stack: superstrate (0 thick) | pillar layer (H) | substrate (0 thick)
    Profile = { [0, H_um, 0], [1, 2, 3] };

    % Eigenmodes
    aa = res1(wl, period, textures, nn, k_par, delta, parm);

    % S-matrix solution
    S  = res2(aa, Profile);

    T_TE(il) = real(get_order00(S.TEinc_top_transmitted));
    T_TM(il) = real(get_order00(S.TMinc_top_transmitted));
    R_TE(il) = real(get_order00(S.TEinc_top_reflected));
    R_TM(il) = real(get_order00(S.TMinc_top_reflected));
end

cd(orig_dir);

T_unpol = 0.5 * (T_TE + T_TM);
R_unpol = 0.5 * (R_TE + R_TM);
end


function val = get_order00(field)
% Extract the (0,0) zeroth-order efficiency from a Reticolo output field.
% Falls back to the first element if no explicit (0,0) order is found.
    ord = field.order;
    if size(ord, 2) >= 2
        idx = find(ord(:,1) == 0 & ord(:,2) == 0, 1);
    else
        idx = find(ord(:,1) == 0, 1);
    end
    if isempty(idx)
        idx = 1;
    end
    val = field.efficiency(idx);
end
