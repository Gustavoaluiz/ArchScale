# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Set global font sizes for LaTeX-like appearance (from plot_accuracy.py)
plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 22, # plot_graph.py uses 22, plot_accuracy.py uses 24, let's use 22 to match plot_graph specific
    'figure.titlesize': 24
})

# Data from the table
tokens = np.array([100, 200, 300, 400, 500, 600])

baseline_loss =np.array([2.07070348, 2.01282085, 1.99077425, 1.97531637, 1.96365183,1.95407671])
mup_loss = np.array([2.05160775, 2.00272247, 1.97841883, 1.96564275, 1.95346724, 1.946053  ])
sambay_mup_loss = np.array([2.0242, 1.9766, 1.9548, 1.9385, 1.9311, 1.92245]) 
sambayoco_mup_loss = np.array([2.0212, 1.9736, 1.9509, 1.9377, 1.9272, 1.9209]) 
nobsscale_loss = np.array([2.05136352, 1.99953216, 1.97367817, 1.95834678, 1.94749461, 1.93951834])
mup_normal_loss = np.array([ 2.0966, 2.0283,1.996, 1.9770, 1.9617,1.95586])
mup_wsd_loss = np.array([2.0301,1.9812,1.9604,1.9447,1.9373,1.9306])
mup_lrsc_loss = np.array([2.0454, 1.9939, 1.9760,1.9612,1.9517,1.9466])
mup_untie_lecun_loss = np.array([2.0318,1.9818,1.9604,1.9425,1.9348,1.9272])
ori_mup_loss = np.array([2.0708,2.0162,1.9905,1.9698,1.9589,2.0075])
mup_lrsc_lecun_loss = np.array([2.0293, 1.9831, 1.9655, 1.9497, 1.9403,1.9330])
# --- Plotting ---
plt.figure(figsize=(16, 12)) # Adjust figure size for better readability

# Define the power law function to fit: L = A * D^(-b) + C
def power_law_with_offset(D, A, b, C):
    return A * D**(-b) + C

plt.figure(figsize=(20, 12))  # Increased figure size for better readability

# Data series to plot and fit
data_series = {
    # 'SP': baseline_loss,
    # 'μP': ori_mup_loss,
    # 'μP++': nobsscale_loss,
    # 'μP++ (Batch Scaling)': mup_loss,
    # 'μP++ (Normal Init.)': mup_normal_loss,

    # 'Transformer++ (SP)': baseline_loss,
    # 'Transformer++ (μP++)': nobsscale_loss,
    # 'Samba+YOCO (μP++)': sambayoco_mup_loss,
    # 'SambaY (μP++)': sambay_mup_loss,
    
    'μP++ ': mup_untie_lecun_loss,   
    'μP++ (WSD)': mup_wsd_loss, 
    'μP++ (LR Scaling + Indep. WD)': mup_lrsc_lecun_loss, 

}

markers = ['o','d', 's',  '^',  'v', '>', '<', 'p', '*', 'h', '8', 'D', 'P', 'X', '4', '3', '2']  # Added one more marker
colors = plt.cm.get_cmap('tab10', len(data_series))  # Using tab20 colormap for better color distinction

for i, (label, loss_data) in enumerate(data_series.items()):
    # Filter out NaNs for both tokens and loss_data
    valid_mask = ~np.isnan(loss_data)
    current_tokens = tokens[valid_mask]
    current_loss = loss_data[valid_mask]

    if len(current_tokens) > 1: # Need at least 2 points to fit a line label=label,
        # Plot original data points as markers only
        plt.plot(current_tokens, current_loss, marker=markers[i % len(markers)], linestyle='none', label=label, color=colors(i),markersize=16,)

        # Initial guesses for parameters A, b, C
        # These are heuristics and might need tuning if fitting fails
        c_guess = np.min(current_loss) * 0.95 # Asymptotic loss slightly below min observed
        b_guess = 0.1 # Small positive exponent
        # Ensure c_guess is less than the first loss point for a_guess calculation
        if c_guess >= current_loss[0]:
            c_guess_safe = current_loss[0] * 0.95
        else:
            c_guess_safe = c_guess
        a_guess = (current_loss[0] - c_guess_safe) * (current_tokens[0]**b_guess)
        if a_guess <= 0: # A should ideally be positive
            a_guess = 1.0

        initial_params = [a_guess, b_guess, c_guess]
        param_bounds = ([0, 0, 0], [np.inf, np.inf, np.min(current_loss)]) # Bounds: A>0, b>0, 0 < C < min_loss

        try:
            params, covariance = curve_fit(power_law_with_offset, current_tokens, current_loss, 
                                           p0=initial_params, bounds=param_bounds, maxfev=5000)
            A, b, C_fit = params
            
            # Calculate R-squared for goodness of fit
            fitted_loss_at_data_points = power_law_with_offset(current_tokens, A, b, C_fit)
            ss_res = np.sum((current_loss - fitted_loss_at_data_points)**2)
            ss_tot = np.sum((current_loss - np.mean(current_loss))**2)
            if ss_tot == 0: # Avoid division by zero if all y values are the same
                r_squared = 1.0 if ss_res == 0 else 0.0
            else:
                r_squared = 1 - (ss_res / ss_tot)
            
            # Generate y-values for the fitted line
            # Ensure tokens for plotting are sorted for a smooth line, especially if original tokens aren't.
            plot_tokens = np.geomspace(current_tokens.min(), current_tokens.max(), 100)
            fitted_loss_for_plotting = power_law_with_offset(plot_tokens, A, b, C_fit)
            
            # Plot the fitted line
            plt.plot(plot_tokens, fitted_loss_for_plotting, linestyle='--', color=colors(i), label=f'{label} (A={A:.2f},b={b:.2f},C={C_fit:.2f})',linewidth=3)
        except RuntimeError:
            print(f"Could not fit {label} to power law with offset.")
            # Optionally, plot just the points if fit fails, or skip plotting fit
            pass # Currently, just prints a message and skips fit line

    elif len(current_tokens) == 1:
        # If only one point, just plot the point (as marker)
        plt.plot(current_tokens, current_loss, marker=markers[i % len(markers)], linestyle='none', label=label, color=colors(i))


plt.xscale('log')   
# plt.yscale('log') # Y-axis should be linear for L = A*D^(-b) + C

plt.xlabel("Training Tokens (Billions)", fontsize=24)
plt.ylabel("Validation Loss", fontsize=24)
plt.legend()#bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)  # Moved legend outside
plt.grid(True, which='both', linestyle='--', alpha=0.7, linewidth=1.5)
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to accommodate external legend
plt.savefig('scaling_data_1B_mup_abl_untie.png', dpi=300, bbox_inches='tight')
plt.show()
