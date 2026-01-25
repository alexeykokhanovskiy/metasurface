
files = dir('*.mat');

% extract numbers at the end of each filename
nums = zeros(numel(files),1);
for k = 1:numel(files)
    % find the integer after the last underscore "_"
    tokens = regexp(files(k).name, '_(\d+)\.mat$', 'tokens');
    nums(k) = str2double(tokens{1}{1});


%     
end

% sort by the extracted numbers
[~, idx] = sort(nums);
files = files(idx);


R_arr = [];
for k = 2:2:numel(files)
    data = load(files(k).name);
    disp(['Loaded: ' files(k).name]);
R_arr = [R_arr; data.T_TE];
    % ---- your processing here ----
    % e.g. data.variableName
end

%%
figure
h_sio2arr = linspace(0, 0.5, 10)*1000;
lam_arr = linspace(1, 6 ,2500);
imagesc(lam_arr, h_sio2arr, R_arr)
xlabel('\lambda (\mum)');
colorbar
ylabel('h_{SiO2}, (nm)');

%%
subplot(1,2,2)
k = 0;
for i = 1:2:10
plot(lam_arr(520:700), R_arr(i,520:700)+k*0.3*0, 'displayname', sprintf('%.0f',h_sio2arr(i)) )
hold on
k = k+1;
end
xlabel('\lambda (\mum)');
ylabel('R')
% R_arrs(3) = R_arr)

