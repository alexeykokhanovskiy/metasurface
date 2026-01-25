import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def normalize_spec(array):
    array -= np.min(array)
    array /= np.max(array)
    return array

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def crop_wl(wl_min, wl_max, data, metasurfaсe_idx):
    wl_min_idx = find_nearest(data[:,1], wl_min)
    wl_max_idx = find_nearest(data[:,1], wl_max)
    return data[wl_min_idx:wl_max_idx,1], data[wl_min_idx:wl_max_idx,metasurfaсe_idx]

def load_experimental_curve(wl_min, wl_max, experimental_file_path, col_index):
    data = np.genfromtxt(experimental_file_path, delimiter=',')
    wl, signal = crop_wl(wl_min, wl_max, data, col_index)
    return wl, signal

def load_modeling_curve(modeling_file_path):
    data = np.genfromtxt(modeling_file_path, delimiter=',')
    return data[:, 0], data[:, 1]

def calc_mae_model_experimental(wl_exp, wl_model, signal_exp, signal_model):
    """Mean Absolute Error (MAE)"""
    f_model = interp1d(wl_model, signal_model, kind='cubic', fill_value="extrapolate")
    error = np.mean(np.abs(f_model(wl_exp) - signal_exp))
    return error

def calc_mse_model_experimental(wl_exp, wl_model, signal_exp, signal_model):
    """Mean Squared Error (MSE) — то, что у вас раньше было под именем MAE"""
    f_model = interp1d(wl_model, signal_model, kind='cubic', fill_value="extrapolate")
    error = np.mean((f_model(wl_exp) - signal_exp) ** 2)
    return error

def calc_rmse_model_experimental(wl_exp, wl_model, signal_exp, signal_model):
    """Root Mean Squared Error (RMSE)"""
    f_model = interp1d(wl_model, signal_model, kind='cubic', fill_value="extrapolate")
    mse = np.mean((f_model(wl_exp) - signal_exp) ** 2)
    error = np.sqrt(mse)
    return error

def calc_weighted_mse_model_experimental(wl_exp, wl_model, signal_exp, signal_model, weights=None):
    """Weighted MSE — полезно для акцента на важных областях спектра"""
    f_model = interp1d(wl_model, signal_model, kind='cubic', fill_value="extrapolate")
    if weights is None:
        weights = np.ones_like(signal_exp)
    error = np.mean(weights * (f_model(wl_exp) - signal_exp) ** 2)
    return error


def calc_peak_aware_loss(wl_exp, wl_model, signal_exp, signal_model, peak_weight=3.0, derivative_weight=0.5,
                         detect_broad_peaks=True):
    """
    Peak-aware loss function that emphasizes fitting peaks and preserves spectral shape.
    Improved to detect both sharp and broad peaks.

    Args:
        wl_exp: Experimental wavelengths
        wl_model: Model wavelengths
        signal_exp: Experimental signal
        signal_model: Model signal
        peak_weight: Weight multiplier for peak regions (default: 3.0)
        derivative_weight: Weight for derivative matching (default: 0.5)
        detect_broad_peaks: If True, uses more sensitive detection for broad peaks (default: True)

    Returns:
        Combined loss value
    """
    f_model = interp1d(wl_model, signal_model, kind='cubic', fill_value="extrapolate")
    predicted = f_model(wl_exp)

    # 1. Base MSE
    mse = np.mean((predicted - signal_exp) ** 2)

    # 2. Identify peaks and valleys in experimental data
    grad_exp = np.gradient(signal_exp)
    grad2_exp = np.gradient(grad_exp)

    if detect_broad_peaks:
        # More sensitive detection for broad, low-contrast peaks
        # Method 1: Use local maxima detection
        # A point is a peak if it's higher than its neighbors
        window = 3  # Look at neighbors within this window
        is_local_max = np.zeros_like(signal_exp, dtype=bool)
        for i in range(len(signal_exp)):
            start = max(0, i - window)
            end = min(len(signal_exp), i + window + 1)
            if signal_exp[i] == np.max(signal_exp[start:end]):
                is_local_max[i] = True

        # Method 2: Regions where first derivative changes from positive to negative
        # (slope goes from increasing to decreasing)
        is_inflection = np.zeros_like(signal_exp, dtype=bool)
        for i in range(1, len(grad_exp) - 1):
            if grad_exp[i-1] > 0 and grad_exp[i+1] < 0:  # Slope changes from + to -
                is_inflection[i] = True

        # Method 3: Second derivative approach with relaxed threshold
        # For broad peaks, curvature is less negative
        curvature_threshold = -np.std(grad2_exp) * 0.2  # More lenient (was 0.5)
        is_curved_down = grad2_exp < curvature_threshold

        # Combine methods: peaks are local maxima OR inflection points with downward curvature
        # AND signal is above median (to avoid noise in low-signal regions)
        median_signal = np.percentile(signal_exp, 50)  # More lenient than 75th percentile
        is_peak_region = (is_local_max | (is_inflection & is_curved_down)) & (signal_exp > median_signal)

        # Valley detection: local minima below median
        is_local_min = np.zeros_like(signal_exp, dtype=bool)
        for i in range(len(signal_exp)):
            start = max(0, i - window)
            end = min(len(signal_exp), i + window + 1)
            if signal_exp[i] == np.min(signal_exp[start:end]):
                is_local_min[i] = True

        valley_threshold = np.percentile(signal_exp, 25)
        is_valley_region = is_local_min & (signal_exp < valley_threshold)
    else:
        # Original sharp peak detection
        peak_threshold = np.percentile(signal_exp, 75)
        is_peak_region = (grad2_exp < -np.std(grad2_exp) * 0.5) & (signal_exp > peak_threshold)

        valley_threshold = np.percentile(signal_exp, 25)
        is_valley_region = (grad2_exp > np.std(grad2_exp) * 0.5) & (signal_exp < valley_threshold)

    # Create weight array
    weights = np.ones_like(signal_exp)
    weights[is_peak_region] = peak_weight  # Emphasize peaks
    weights[is_valley_region] = peak_weight * 0.5  # Moderate emphasis on valleys

    # Weighted MSE
    weighted_mse = np.mean(weights * (predicted - signal_exp) ** 2)

    # 3. Derivative matching to preserve shape
    grad_pred = np.gradient(predicted)
    derivative_error = np.mean((grad_pred - grad_exp) ** 2)

    # 4. Correlation penalty to preserve overall shape
    correlation = np.corrcoef(predicted, signal_exp)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    correlation_penalty = 1.0 - correlation

    # Combined loss
    total_loss = weighted_mse + derivative_weight * derivative_error + 0.2 * correlation_penalty

    return total_loss


def calc_multi_scale_loss(wl_exp, wl_model, signal_exp, signal_model, alpha=0.6, beta=0.3, gamma=0.1):
    """
    Multi-scale loss combining global fit, local features, and shape preservation.

    Args:
        wl_exp: Experimental wavelengths
        wl_model: Model wavelengths
        signal_exp: Experimental signal
        signal_model: Model signal
        alpha: Weight for global MSE (default: 0.6)
        beta: Weight for high-value regions (default: 0.3)
        gamma: Weight for shape correlation (default: 0.1)

    Returns:
        Combined loss value
    """
    f_model = interp1d(wl_model, signal_model, kind='cubic', fill_value="extrapolate")
    predicted = f_model(wl_exp)

    # 1. Global MSE
    global_mse = np.mean((predicted - signal_exp) ** 2)

    # 2. Emphasize high-value regions (peaks)
    # Create weights based on signal magnitude
    magnitude_weights = 1.0 + 2.0 * (signal_exp / np.max(signal_exp))
    weighted_mse = np.mean(magnitude_weights * (predicted - signal_exp) ** 2)

    # 3. Shape preservation via correlation
    correlation = np.corrcoef(predicted, signal_exp)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    shape_penalty = 1.0 - correlation

    # Combined loss
    total_loss = alpha * global_mse + beta * weighted_mse + gamma * shape_penalty

    return total_loss


def calc_adaptive_weighted_loss(wl_exp, wl_model, signal_exp, signal_model, sensitivity=2.0):
    """
    Adaptive loss that automatically adjusts weights based on local signal variance.
    High variance regions (peaks, valleys) get more weight.

    Args:
        wl_exp: Experimental wavelengths
        wl_model: Model wavelengths
        signal_exp: Experimental signal
        signal_model: Model signal
        sensitivity: Controls how much to emphasize high-variance regions (default: 2.0)

    Returns:
        Weighted loss value
    """
    f_model = interp1d(wl_model, signal_model, kind='cubic', fill_value="extrapolate")
    predicted = f_model(wl_exp)

    # Calculate local variance using rolling window
    window_size = max(3, len(signal_exp) // 20)  # Adaptive window size

    local_variance = np.zeros_like(signal_exp)
    for i in range(len(signal_exp)):
        start = max(0, i - window_size // 2)
        end = min(len(signal_exp), i + window_size // 2 + 1)
        local_variance[i] = np.var(signal_exp[start:end])

    # Create adaptive weights
    # Normalize variance to [0, 1] and scale
    norm_variance = local_variance / (np.max(local_variance) + 1e-10)
    weights = 1.0 + sensitivity * norm_variance

    # Weighted MSE
    weighted_mse = np.mean(weights * (predicted - signal_exp) ** 2)

    return weighted_mse

def calc_chi_squared_model_experimental(wl_exp, wl_model, signal_exp, signal_model, sigma=None):
    """Chi-squared (полезно при известном шуме)"""
    f_model = interp1d(wl_model, signal_model, kind='cubic', fill_value="extrapolate")
    if sigma is None:
        # Грубо оцениваем шум как локальное стандартное отклонение или константу
        sigma = np.std(signal_exp) * np.ones_like(signal_exp)
    error = np.sum((f_model(wl_exp) - signal_exp) ** 2 / sigma ** 2)
    return error

def calc_hybrid_model_experimental(wl_exp, wl_model, signal_exp, signal_model, gamma=1.0):
    """RMSE + штраф за низкую корреляцию (хорошо ловит форму кривой)"""
    f_model = interp1d(wl_model, signal_model, kind='cubic', fill_value="extrapolate")
    predicted = f_model(wl_exp)
    rmse = np.sqrt(np.mean((predicted - signal_exp) ** 2))
    corr, _ = pearsonr(predicted, signal_exp)
    # corr может быть NaN при нулевой вариации — защищаемся
    if np.isnan(corr):
        corr = 0.0
    error = rmse + gamma * (1 - corr)
    return error



if __name__ == '__main__':

    wl_model, signal_model = load_modeling_curve('R_matlab.csv')
    plt.plot(wl_model, signal_model, marker='o')
    plt.show()

    # wl_exp, signal_exp = load_experimental_curve(wl_min= 1.7, wl_max = 4.0, experimental_file_path = './LIPPS_sio2_polarized.csv', col_index =6)
    # signal_exp /=100
    # wl_model, signal_model = load_modeling_curve('R_matlab.csv')
    # err = calc_mae_model_experimental(wl_exp, wl_model, signal_exp, signal_model)
    # print(err)
    # plt.plot(wl_model, signal_model)
    # plt.plot(wl_exp, signal_exp)
    #
    # plt.show()



# wl, lipss = crop_wl(wl_min, wl_max, data, 4)
# data = np.genfromtxt('./LIPPS_sio2_polarized.csv', delimiter=',')
# wl, lipss_covered = crop_wl(wl_min, wl_max, data, 6)
#
# # plt.plot(wl, signal_ref)
#
#
#
# data = np.genfromtxt('R_matlab.csv', delimiter=',')
# f_model = interp1d(data[:,0], data[:,1], kind='cubic', fill_value="extrapolate")
#
# error = np.mean((f_model(wl) - lipss_covered/100)**2)
# print(error)
# plt.plot(wl,lipss_covered/100)
# plt.plot(wl,f_model(wl))
# plt.show()


