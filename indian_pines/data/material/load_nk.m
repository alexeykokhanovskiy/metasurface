function [n_complex, n_real, k_imag] = load_nk(csv_path, wl_query_nm)
% LOAD_NK  Interpolate refractive index from an nk CSV file.
%
%   [n_complex, n_real, k_imag] = LOAD_NK(csv_path, wl_query_nm)
%
%   Reads a CSV with any number of comment lines (starting with '#') and
%   one optional text header line (wavelength_nm,n,k), then linearly
%   interpolates to wl_query_nm.
%
%   Returns complex index n - i*k (Reticolo sign convention).

    fid = fopen(csv_path, 'r');
    if fid < 0
        error('load_nk:open', 'Cannot open %s', csv_path);
    end

    wl_tbl = [];
    n_tbl  = [];
    k_tbl  = [];

    while ~feof(fid)
        line = strtrim(fgetl(fid));
        if isempty(line),      continue; end  % blank line
        if line(1) == '#',     continue; end  % comment
        if isletter(line(1)),   continue; end  % text header (starts with a letter)

        vals = sscanf(line, '%f,%f,%f');
        if numel(vals) < 3,    continue; end  % malformed row

        wl_tbl(end+1) = vals(1);              %#ok<AGROW>
        n_tbl(end+1)  = vals(2);              %#ok<AGROW>
        k_tbl(end+1)  = vals(3);              %#ok<AGROW>
    end
    fclose(fid);

    if isempty(wl_tbl)
        error('load_nk:empty', 'No numeric data found in %s', csv_path);
    end

    wl_query_nm = wl_query_nm(:);

    if any(wl_query_nm < wl_tbl(1)) || any(wl_query_nm > wl_tbl(end))
        warning('load_nk:extrap', ...
                'Query wavelength outside table range [%.0f, %.0f] nm', ...
                wl_tbl(1), wl_tbl(end));
    end

    n_real = interp1(wl_tbl, n_tbl, wl_query_nm, 'linear', 'extrap');
    k_imag = interp1(wl_tbl, k_tbl, wl_query_nm, 'linear', 'extrap');
    k_imag = max(k_imag, 0);

    n_complex = n_real - 1i * k_imag;
end
