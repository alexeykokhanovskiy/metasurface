import numpy as np
import matplotlib.pyplot as plt
from data_processing import load_experimental_curve

# ===============================
# LOAD EXPERIMENTAL DATA
# ===============================
WL_MIN = 2.5
WL_MAX = 4.0
EXP_FILE = './LIPPS_sio2_polarized.csv'
EXP_COL = 6

wl_exp, signal_exp = load_experimental_curve(
    wl_min=WL_MIN, wl_max=WL_MAX,
    experimental_file_path=EXP_FILE,
    col_index=EXP_COL
)
signal_exp /= 100.0

# ===============================
# PEAK DETECTION ALGORITHM - IMPROVED FOR BROAD PEAKS
# ===============================
# Calculate gradients
grad_exp = np.gradient(signal_exp)
grad2_exp = np.gradient(grad_exp)

# Method 1: Use local maxima detection
window = 3  # Look at neighbors within this window
is_local_max = np.zeros_like(signal_exp, dtype=bool)
for i in range(len(signal_exp)):
    start = max(0, i - window)
    end = min(len(signal_exp), i + window + 1)
    if signal_exp[i] == np.max(signal_exp[start:end]):
        is_local_max[i] = True

# Method 2: Regions where first derivative changes from positive to negative
is_inflection = np.zeros_like(signal_exp, dtype=bool)
for i in range(1, len(grad_exp) - 1):
    if grad_exp[i-1] > 0 and grad_exp[i+1] < 0:  # Slope changes from + to -
        is_inflection[i] = True

# Method 3: Second derivative approach with relaxed threshold (for broad peaks)
curvature_threshold = -np.std(grad2_exp) * 0.2  # More lenient (was 0.5 for sharp peaks)
is_curved_down = grad2_exp < curvature_threshold

# Combine methods: peaks are local maxima OR inflection points with downward curvature
# AND signal is above median (to avoid noise in low-signal regions)
median_signal = np.percentile(signal_exp, 50)  # More lenient than 75th percentile
is_peak_region = (is_local_max | (is_inflection & is_curved_down)) & (signal_exp > median_signal)

# Valley detection: local minima below 25th percentile
is_local_min = np.zeros_like(signal_exp, dtype=bool)
for i in range(len(signal_exp)):
    start = max(0, i - window)
    end = min(len(signal_exp), i + window + 1)
    if signal_exp[i] == np.min(signal_exp[start:end]):
        is_local_min[i] = True

valley_threshold = np.percentile(signal_exp, 25)
is_valley_region = is_local_min & (signal_exp < valley_threshold)

# Create weight array for visualization
weights = np.ones_like(signal_exp)
peak_weight = 3.0
weights[is_peak_region] = peak_weight
weights[is_valley_region] = peak_weight * 0.5

# ===============================
# VISUALIZATION
# ===============================
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# Plot 1: Original signal
axes[0].plot(wl_exp, signal_exp, 'b-', linewidth=2, label='Experimental Signal')
axes[0].set_xlabel('Wavelength (μm)', fontsize=11)
axes[0].set_ylabel('Reflectance', fontsize=11)
axes[0].set_title('Experimental Spectrum', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot 2: First derivative (slope)
axes[1].plot(wl_exp, grad_exp, 'g-', linewidth=2, label='First Derivative')
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1].set_xlabel('Wavelength (μm)', fontsize=11)
axes[1].set_ylabel('dR/dλ', fontsize=11)
axes[1].set_title('First Derivative (Slope)', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Plot 3: Second derivative (curvature) with peak/valley identification
axes[2].plot(wl_exp, grad2_exp, 'r-', linewidth=2, label='Second Derivative')
axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[2].axhline(y=-np.std(grad2_exp) * 0.2, color='orange', linestyle='--',
                alpha=0.5, label='Peak Threshold (broad)')
axes[2].axhline(y=np.std(grad2_exp) * 0.5, color='purple', linestyle='--',
                alpha=0.5, label='Valley Threshold')
axes[2].fill_between(wl_exp, grad2_exp, 0, where=is_peak_region,
                       color='orange', alpha=0.3, label='Detected Peaks')
axes[2].fill_between(wl_exp, grad2_exp, 0, where=is_valley_region,
                       color='purple', alpha=0.3, label='Detected Valleys')
axes[2].set_xlabel('Wavelength (μm)', fontsize=11)
axes[2].set_ylabel('d²R/dλ²', fontsize=11)
axes[2].set_title('Second Derivative (Curvature) - Broad Peak Detection', fontsize=13, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].legend(fontsize=9, loc='upper right')

# Plot 4: Original signal with highlighted peak and valley regions + weights
axes[3].plot(wl_exp, signal_exp, 'b-', linewidth=2, label='Experimental Signal', zorder=1)
axes[3].axhline(y=median_signal, color='orange', linestyle='--', alpha=0.5,
                label=f'Peak Region Threshold (50th %ile = {median_signal:.3f})')
axes[3].axhline(y=valley_threshold, color='purple', linestyle='--', alpha=0.5,
                label=f'Valley Region Threshold (25th %ile = {valley_threshold:.3f})')

# Highlight peak regions
axes[3].scatter(wl_exp[is_peak_region], signal_exp[is_peak_region],
                color='red', s=80, marker='o', label=f'Peak Regions (weight={peak_weight})',
                zorder=3, edgecolors='darkred', linewidths=1.5)

# Highlight valley regions
axes[3].scatter(wl_exp[is_valley_region], signal_exp[is_valley_region],
                color='magenta', s=60, marker='s', label=f'Valley Regions (weight={peak_weight*0.5})',
                zorder=2, edgecolors='darkmagenta', linewidths=1.5)

axes[3].set_xlabel('Wavelength (μm)', fontsize=11)
axes[3].set_ylabel('Reflectance', fontsize=11)
axes[3].set_title('Peak and Valley Regions Highlighted (Broad Peak Detection)', fontsize=13, fontweight='bold')
axes[3].grid(True, alpha=0.3)
axes[3].legend(fontsize=9, loc='best')

plt.tight_layout()
plt.suptitle('Broad Peak Detection Algorithm for Low-Contrast Peaks', fontsize=14, fontweight='bold', y=0.998)
plt.subplots_adjust(top=0.96)
plt.savefig('peak_detection_analysis_broad.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to 'peak_detection_analysis_broad.png'")
plt.show()

# ===============================
# STATISTICS
# ===============================
print("\n" + "="*70)
print("PEAK DETECTION STATISTICS")
print("="*70)
print(f"Total data points: {len(signal_exp)}")
print(f"Peak regions detected: {np.sum(is_peak_region)} points ({100*np.sum(is_peak_region)/len(signal_exp):.1f}%)")
print(f"Valley regions detected: {np.sum(is_valley_region)} points ({100*np.sum(is_valley_region)/len(signal_exp):.1f}%)")
print(f"Normal regions: {len(signal_exp) - np.sum(is_peak_region) - np.sum(is_valley_region)} points")

print(f"\nSignal statistics:")
print(f"  Min: {signal_exp.min():.4f}")
print(f"  Max: {signal_exp.max():.4f}")
print(f"  Mean: {signal_exp.mean():.4f}")
print(f"  Median: {median_signal:.4f}")
print(f"  Peak threshold (50th %ile): {median_signal:.4f}")
print(f"  Valley threshold (25th %ile): {valley_threshold:.4f}")

print(f"\nPeak detection methods:")
print(f"  Local maxima detected: {np.sum(is_local_max)} points")
print(f"  Inflection points detected: {np.sum(is_inflection)} points")
print(f"  Downward curvature regions: {np.sum(is_curved_down)} points")
print(f"  Combined peak regions: {np.sum(is_peak_region)} points ({100*np.sum(is_peak_region)/len(signal_exp):.1f}%)")

print(f"\nWeights applied:")
print(f"  Peak regions: {peak_weight}x")
print(f"  Valley regions: {peak_weight * 0.5}x")
print(f"  Normal regions: 1.0x")

print(f"\nGradient statistics:")
print(f"  First derivative std: {np.std(grad_exp):.6f}")
print(f"  Second derivative std: {np.std(grad2_exp):.6f}")
print(f"  Curvature threshold (broad peaks): {curvature_threshold:.6f} (negative)")
print(f"  Original sharp peak threshold would be: {-np.std(grad2_exp) * 0.5:.6f} (negative)")
print(f"  Valley detection threshold: {np.std(grad2_exp) * 0.5:.6f} (positive)")

# Find wavelengths of peaks
peak_wavelengths = wl_exp[is_peak_region]
if len(peak_wavelengths) > 0:
    print(f"\nPeak wavelengths:")
    # Group consecutive peaks
    peak_groups = []
    current_group = [peak_wavelengths[0]]
    for wl in peak_wavelengths[1:]:
        if wl - current_group[-1] < (wl_exp[1] - wl_exp[0]) * 2:  # If close to previous
            current_group.append(wl)
        else:
            peak_groups.append(current_group)
            current_group = [wl]
    peak_groups.append(current_group)

    for i, group in enumerate(peak_groups):
        center_wl = np.mean(group)
        print(f"  Peak {i+1}: centered around {center_wl:.3f} μm (width: {len(group)} points)")

print("="*70)
