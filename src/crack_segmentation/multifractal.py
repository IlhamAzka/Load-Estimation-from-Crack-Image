import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy import ndimage
from sklearn.linear_model import LinearRegression

# IMG_PATH = "./Fractal Database/20241201_090401_color_mask.png"
# IMG_PATH = "./Fractal Database/101200056.bmp"
# IMG_PATH = "./test_crop/mask/20241201_090401_crop_3_color_mask.png"

def load_image(image_path):
    """Load and convert image to grayscale."""
    # img = Image.open(image_path).convert('L')
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    return np.array(img)

def box_counting_multifractal(image, q_values, box_sizes=None):
    """
    Perform multifractal analysis using box-counting method.
    
    Parameters:
    - image: 2D numpy array (grayscale image)
    - q_values: array of q moments to compute
    - box_sizes: array of box sizes (default: powers of 2)
    
    Returns:
    - tau_q: scaling exponents
    - D_q: generalized fractal dimensions
    - f_alpha: multifractal spectrum (singularity strength)
    - alpha: Holder exponents
    """
    h, w = image.shape
    
    # Normalize image to [0, 1]
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)
    
    # Default box sizes
    if box_sizes is None:
        max_size = min(h, w) // 4
        box_sizes = [2**i for i in range(2, int(np.log2(max_size)) + 1)]
    
    # Storage for partition functions
    Z_q = {q: [] for q in q_values}
    
    for box_size in box_sizes:
        n_boxes_h = h // box_size
        n_boxes_w = w // box_size
        
        # Calculate mass in each box
        masses = []
        for i in range(n_boxes_h):
            for j in range(n_boxes_w):
                box = image_norm[i*box_size:(i+1)*box_size, 
                                j*box_size:(j+1)*box_size]
                mass = np.sum(box)
                if mass > 0:
                    masses.append(mass)
        
        # Normalize masses to probabilities
        total_mass = sum(masses)
        if total_mass > 0:
            probabilities = [m / total_mass for m in masses]
            
            # Calculate partition function for each q
            for q in q_values:
                if q == 1:
                    # Special case for q=1 (information dimension)
                    Z = sum([p * np.log(p) if p > 0 else 0 for p in probabilities])
                else:
                    Z = sum([p**q for p in probabilities if p > 0])
                Z_q[q].append(Z if Z > 0 else 1e-10)
    
    # Calculate tau(q) from scaling
    tau_q = []
    log_box_sizes = np.log(box_sizes)
    
    for q in q_values:
        if q == 1:
            # For q=1, use entropy scaling
            log_Z = np.array(Z_q[q])
            valid = np.isfinite(log_Z)
            if np.sum(valid) > 2:
                reg = LinearRegression()
                reg.fit(log_box_sizes[valid].reshape(-1, 1), log_Z[valid])
                tau = reg.coef_[0]
            else:
                tau = 0
        else:
            log_Z = np.log(np.array(Z_q[q]))
            valid = np.isfinite(log_Z)
            if np.sum(valid) > 2:
                reg = LinearRegression()
                reg.fit(log_box_sizes[valid].reshape(-1, 1), log_Z[valid])
                tau = reg.coef_[0] * (q - 1)
            else:
                tau = 0
        tau_q.append(tau)
    
    tau_q = np.array(tau_q)
    
    # Calculate D(q) - generalized fractal dimensions
    D_q = np.zeros_like(tau_q)
    for i, q in enumerate(q_values):
        if q != 1:
            D_q[i] = tau_q[i] / (q - 1)
        else:
            # Information dimension (limit as q->1)
            if i > 0 and i < len(q_values) - 1:
                D_q[i] = (tau_q[i+1] - tau_q[i-1]) / (q_values[i+1] - q_values[i-1])
            else:
                D_q[i] = tau_q[i]
    
    # Calculate f(alpha) spectrum using Legendre transform
    alpha = np.gradient(tau_q, q_values)
    f_alpha = q_values * alpha - tau_q
    
    return tau_q, D_q, f_alpha, alpha, box_sizes, Z_q

def plot_multifractal_results(q_values, tau_q, D_q, f_alpha, alpha, image):
    """Plot the results of multifractal analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # tau(q) - Mass exponent
    axes[0, 1].plot(q_values, tau_q, 'b-o', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('q (moment order)', fontsize=11)
    axes[0, 1].set_ylabel('τ(q)', fontsize=11)
    axes[0, 1].set_title('Mass Exponent τ(q)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # D(q) - Generalized dimensions
    axes[1, 0].plot(q_values, D_q, 'r-o', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('q (moment order)', fontsize=11)
    axes[1, 0].set_ylabel('D(q)', fontsize=11)
    axes[1, 0].set_title('Generalized Fractal Dimensions D(q)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=2, color='k', linestyle='--', alpha=0.3, label='D=2 (2D image)')
    axes[1, 0].legend()
    
    # f(alpha) - Multifractal spectrum
    axes[1, 1].plot(alpha, f_alpha, 'g-o', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('α (Hölder exponent)', fontsize=11)
    axes[1, 1].set_ylabel('f(α)', fontsize=11)
    axes[1, 1].set_title('Multifractal Spectrum f(α)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def analyze_multifractality(D_q, alpha, f_alpha):
    """Analyze the degree of multifractality."""
    # Spectral width
    alpha_width = np.max(alpha) - np.min(alpha)
    
    # Check if monofractal (D_q should be constant)
    D_q_variation = np.max(D_q) - np.min(D_q)

    D_0 = D_q[len(D_q)//2 - 5]
    D_1 = D_q[len(D_q)//2]
    D_2 = D_q[len(D_q)//2 + 5]

    return D_0, D_1, D_2, alpha_width, D_q_variation
    
    # print("=" * 60)
    # print("MULTIFRACTAL ANALYSIS RESULTS")
    # print("=" * 60)
    # print(f"D(0) - Capacity dimension:     {D_q[len(D_q)//2 - 5]:.4f}")
    # print(f"D(1) - Information dimension:  {D_q[len(D_q)//2]:.4f}")
    # print(f"D(2) - Correlation dimension:  {D_q[len(D_q)//2 + 5]:.4f}")
    # print(f"\nSpectral width Δα:             {alpha_width:.4f}")
    # print(f"D(q) variation:                {D_q_variation:.4f}")
    
    # if alpha_width > 0.5:
    #     print(f"\n✓ Strong multifractality detected (Δα = {alpha_width:.4f})")
    # elif alpha_width > 0.2:
    #     print(f"\n~ Moderate multifractality (Δα = {alpha_width:.4f})")
    # else:
    #     print(f"\n✗ Weak multifractality / Monofractal (Δα = {alpha_width:.4f})")
    # print("=" * 60)