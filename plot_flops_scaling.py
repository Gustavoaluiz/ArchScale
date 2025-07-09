# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Set global font sizes (from plot_accuracy.py)
plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24, # Default legend fontsize from plot_accuracy
    'figure.titlesize': 24
})

# --- Data ---
depths = np.array([8, 12, 16, 20, 24])
baseline_loss = np.array([2.6746, 2.2836, 2.0831, 1.9454, 1.8360])
mup_plus_plus_loss = np.array([2.5621, 2.2440, 2.0505, 1.9164, 1.8172])
sambay_loss = np.array([2.5051, 2.2073, 2.0227, 1.8963, 1.7999]) 
sambayoco_loss = np.array([2.5014, 2.2040, 2.0213, 1.8957, 1.79928]) 
ori_mup_loss = np.array([2.5972,2.2647,2.0682,1.9263,1.8252])
# --- Calculation Constants (from pretrain.py and user info) ---
ar = 128         # Aspect ratio for transformer
d0 = 16          # Base depth
t0 = 1e11        # Base tokens (100B)
mult_transformer = 14.5 * (ar ** 2) # Parameter multiplier for Transformer++ N(d) = mult * d^3
n0_transformer = mult_transformer * (d0**3) # Base parameters for Transformer++ at d0

# --- Function to Calculate FLOPs ---
def calculate_flops(depth, mult, n0, t0):
    """Calculates estimated FLOPs based on 6NT scaling."""
    if depth <= 0:
        return np.nan
    # Calculate target parameters (N)
    n_target = mult * (depth**3)
    # Calculate target tokens (T) using Chinchilla-style scaling relative to base
    tokens = t0 * n_target / n0
    # Estimate FLOPs (C) ~ 6 * N * T
    flops = 6 * n_target * tokens
    return flops

# --- Calculate FLOPs for each depth ---
# We use the transformer parameters for FLOP calculation as requested
flops_axis = np.array([calculate_flops(d, mult_transformer, n0_transformer, t0) for d in depths])

# Filter out NaN FLOPs and corresponding losses for plotting
valid_indices_baseline = ~np.isnan(baseline_loss) & ~np.isnan(flops_axis)
valid_indices_mup_plus_plus = ~np.isnan(mup_plus_plus_loss) & ~np.isnan(flops_axis)
valid_indices_sambay = ~np.isnan(sambay_loss) & ~np.isnan(flops_axis)
valid_indices_sambayoco = ~np.isnan(sambayoco_loss) & ~np.isnan(flops_axis)
valid_indices_ori_mup = ~np.isnan(ori_mup_loss) & ~np.isnan(flops_axis)
# --- Define the power law function to fit: L = A * D^(-b) + C ---
def power_law_with_offset(D, A, b, C):
    # Ensure D is positive before taking power for -b, handle D=0 if necessary
    # For FLOPs, D should always be > 0
    return A * D**(-b) + C

# --- Plotting ---
plt.figure(figsize=(16, 12)) # Updated figure size

# --- Data series to plot and fit ---
data_series_to_plot = {
    'Transformer++ (SP)': (baseline_loss, valid_indices_baseline, 'o'),
    'Transformer++ (μP)': (ori_mup_loss, valid_indices_ori_mup, 'v'),
    'Transformer++ (μP++)': (mup_plus_plus_loss, valid_indices_mup_plus_plus, 'd'),
    'Samba+YOCO (μP++)': (sambayoco_loss, valid_indices_sambayoco, 's'), # Changed marker for distinction
    'SambaY (μP++)': (sambay_loss, valid_indices_sambay, '^'), # Changed marker for distinction
}

colors = plt.cm.get_cmap('tab10', len(data_series_to_plot))

for i, (label, (loss_data, valid_indices, marker_style)) in enumerate(data_series_to_plot.items()):
    current_flops = flops_axis[valid_indices]
    current_loss = loss_data[valid_indices]

    if len(current_flops) > 1:
        # Plot original data points as markers only
        plt.plot(current_flops, current_loss, marker=marker_style, linestyle='none', label=label, color=colors(i),markersize=16,)

        # Initial guesses for parameters A, b, C
        c_guess = np.min(current_loss) * 0.95 
        b_guess = 0.05 # Exponent for FLOPs might be smaller
        if c_guess >= current_loss[0]:
            c_guess_safe = current_loss[0] * 0.99 # Ensure C is below first point
        else:
            c_guess_safe = c_guess
        # Handle potential zero or negative (current_loss[0] - c_guess_safe)
        term_A = current_loss[0] - c_guess_safe
        if term_A <= 0 : term_A = 1e-9 # small positive number
        a_guess = term_A * (current_flops[0]**b_guess)
        if a_guess <= 0: a_guess = 1.0 # Ensure A is positive

        initial_params = [a_guess, b_guess, c_guess_safe] # Use c_guess_safe
        # Bounds: A>0, b can be small positive, 0 < C < min_loss
        param_bounds = ([0, 1e-6, 0], [np.inf, 1.0, np.min(current_loss)]) 
        
        try:
            params, covariance = curve_fit(power_law_with_offset, current_flops, current_loss, 
                                           p0=initial_params, bounds=param_bounds, maxfev=10000) # Increased maxfev
            A, b, C_fit = params
            
            # Calculate R-squared for goodness of fit
            fitted_loss_at_data_points = power_law_with_offset(current_flops, A, b, C_fit)
            ss_res = np.sum((current_loss - fitted_loss_at_data_points)**2)
            ss_tot = np.sum((current_loss - np.mean(current_loss))**2)
            if ss_tot == 0: # Avoid division by zero if all y values are the same
                r_squared = 1.0 if ss_res == 0 else 0.0 # Or handle as NaN/None
            else:
                r_squared = 1 - (ss_res / ss_tot)

            # Generate FLOPs for plotting the fitted line smoothly on log scale
            plot_flops = np.geomspace(current_flops.min(), current_flops.max(), 100)
            fitted_loss_for_plotting = power_law_with_offset(plot_flops, A, b, C_fit)
            
            plt.plot(plot_flops, fitted_loss_for_plotting, linestyle='--', color=colors(i), label=f'{label} (A={A:.2f},b={b:.2f},C={C_fit:.2f})',linewidth=3)
        except RuntimeError as e:
            print(f"Could not fit {label} to power law with offset: {e}")
            pass
    elif len(current_flops) == 1:
        plt.plot(current_flops, current_loss, marker=marker_style, linestyle='none', label=label, color=colors(i),markersize=16,)

# Set x-axis to log scale as FLOPs span orders of magnitude
plt.xscale('log')
# Optional: Set y-axis to log scale if you want to see if it's a power law
# plt.yscale('log')

# Add labels and title
plt.xlabel("Estimated Training FLOPs")
plt.ylabel("Validation Loss") # Change label if using log scale for y
#plt.title("Scaling d8-d24 (100M-3.3B): Loss vs. Estimated Compute (FLOPs)")

# Add legend
plt.legend(fontsize=20) # Explicitly keeping 'small' for this plot as per original

# Add grid
plt.grid(True, which='both', linestyle='--', alpha=0.7, linewidth=1.5) # Updated grid style

# Improve layout and display
plt.tight_layout()
plt.savefig('scaling_d8-d24.png', dpi=300, bbox_inches='tight')
plt.show()