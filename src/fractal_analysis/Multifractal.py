"""
Multifractal Analysis of Concrete Crack Images

This script performs multifractal analysis on concrete crack images using the box-counting method
and multifractal spectrum (MFS) calculation. It provides various fractal dimensions and 
singularity spectrum characteristics.

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.optimize import curve_fit
from skimage import io, color, filters
from skimage.morphology import skeletonize
import warnings
warnings.filterwarnings('ignore')

def get_partition_function(image, box_sizes, q_values):
    """Calculates the partition function Z(q, eps) for various box sizes."""
    height, width = image.shape
    results = np.zeros((len(q_values), len(box_sizes)))

    for i, size in enumerate(box_sizes):
        # Determine number of boxes in each dimension
        n_y = height // size
        n_x = width // size
        
        # Reshape and sum to get mass in each box (box-counting)
        # We crop the image to fit the box size perfectly
        cropped = image[:n_y*size, :n_x*size]
        boxes = cropped.reshape(n_y, size, n_x, size).sum(axis=(1, 3))
        
        # Normalize to get probabilities (P_i)
        probs = boxes.flatten()
        probs = probs[probs > 0] # Remove empty boxes to avoid log(0)
        probs /= probs.sum()

        for j, q in enumerate(q_values):
            if abs(q - 1.0) < 1e-9:
                # Handling the q=1 singularity using Shannon Entropy
                results[j, i] = np.sum(probs * np.log(probs))
            else:
                # Standard partition function for q != 1
                results[j, i] = np.log(np.sum(probs**q))
                
    return results

def multifractal_analysis(image_path, invert: bool):
    # 1. Load and Preprocess
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # the algorithm is built for w&b image, not b&w image
    if invert:
        img = cv2.bitwise_not(img)
    
    # Normalize image mass
    img = img.astype(float)
    if img.max() > 0:
        img /= img.sum()
    
    # 2. Parameters
    # Box sizes should be powers of 2 for best results
    box_sizes = np.array([2, 4, 8, 16, 32, 64, 128])
    q_values = np.linspace(-5, 5, 41)
    
    # 3. Compute Partition Functions
    Z = get_partition_function(img, box_sizes, q_values)
    
    # 4. Calculate Dq (Generalized Dimensions)
    # Dq is the slope of log(Z) vs log(eps)
    dq_values = []
    log_eps = np.log(box_sizes)
    
    for j, q in enumerate(q_values):
        y = Z[j, :]
        slope, _ = np.polyfit(log_eps, y, 1)
        
        if abs(q - 1.0) < 1e-9:
            dq = slope # For q=1, the slope itself is D1
        else:
            dq = slope / (q - 1.0)
        dq_values.append(dq)

    return q_values, dq_values

def calculate_singularity_spectrum(q_values, dq_values):
    # Calculate tau(q)
    tau = np.array(dq_values) * (np.array(q_values) - 1)
    
    # Calculate alpha (derivative of tau with respect to q)
    # We use numerical differentiation
    alpha = np.gradient(tau, q_values)
    
    # Calculate f(alpha)
    f_alpha = (q_values * alpha) - tau
    
    return alpha, f_alpha

def analyze_multifractality(q_values, dq_values, alpha, f_alpha):

    FD_pos = np.where(q_values == 0)[0][0]
    ID_pos = np.where(q_values == 1)[0][0]
    CD_pos = np.where(q_values == 2)[0][0]

    # Fractal Dimension
    FD = dq_values[::-1][FD_pos]  

    # Information Dimension
    ID = dq_values[::-1][ID_pos]

    # Correlation Dimension
    CD = dq_values[::-1][CD_pos]

    # Capacity
    C = dq_values[0]

    max_f_alpha_pos = np.where(f_alpha == FD)[0][0]
    # max_f_alpha_pos = np.argmax(f_alpha)

    # Singularity Spectrum Width
    W = np.max(alpha) - np.min(alpha)
    
    # Dimensional Difference
    DD = abs(f_alpha[0] - f_alpha[-1])

    # Left Branch Area
    LBA = np.trapz(f_alpha[:max_f_alpha_pos+1], alpha[:max_f_alpha_pos+1])

    # Right Branch Area
    RBA = np.trapz(f_alpha[max_f_alpha_pos:], alpha[max_f_alpha_pos:])

    return W, DD, LBA, RBA, FD, ID, CD, C
    
