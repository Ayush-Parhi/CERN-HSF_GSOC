import numpy as np
import matplotlib.pyplot as plt
import struct
import os
from scipy import stats
import time

# Set random seed for reproducibility
np.random.seed(42)

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_distributions(n_samples=100000):
    """Generate floating-point numbers from three distributions."""
    # Uniform distribution [0, 1]
    uniform_data = np.random.uniform(0, 1, n_samples)
    
    # Gaussian distribution (mean=0, std=1)
    gaussian_data = np.random.normal(0, 1, n_samples)
    
    # Exponential distribution (scale=1)
    exponential_data = np.random.exponential(1, n_samples)
    
    return {
        'uniform': uniform_data,
        'gaussian': gaussian_data,
        'exponential': exponential_data
    }

def compress_float(value, n_bits_to_zero):
    """Zero out n least significant bits of a float's mantissa."""
    # Convert float to its binary representation
    binary = struct.unpack('!I', struct.pack('!f', value))[0]
    
    # Create a mask to zero out the least significant bits
    # IEEE 754 single precision: 1 sign bit, 8 exponent bits, 23 mantissa bits
    mask = 0xFFFFFFFF << n_bits_to_zero
    
    # Apply the mask to zero out the bits
    compressed_binary = binary & mask
    
    # Convert back to float
    compressed_value = struct.unpack('!f', struct.pack('!I', compressed_binary))[0]
    
    return compressed_value

def compress_array(arr, n_bits_to_zero):
    """Apply compression to an array of floats."""
    return np.array([compress_float(val, n_bits_to_zero) for val in arr])

def save_to_binary(data, filepath):
    """Save array data to a binary file."""
    # Ensure the directory exists
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    # Save the file
    with open(filepath, 'wb') as f:
        data.tofile(f)

def calculate_statistics(data):
    """Calculate statistical parameters of the data."""
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data)
    }

def calculate_mse(original, compressed):
    """Calculate Mean Squared Error between original and compressed data."""
    return np.mean((original - compressed) ** 2)

def plot_distributions(original, compressed, dist_name, n_bits):
    """Plot original and compressed distributions."""
    # Create plots directory for this distribution
    plots_dir = os.path.join("results", dist_name, "plots")
    ensure_directory_exists(plots_dir)
    
    plt.figure(figsize=(12, 6))
    
    # Histogram of original and compressed data
    plt.subplot(121)
    plt.hist(original, bins=50, alpha=0.5, label='Original')
    plt.hist(compressed, bins=50, alpha=0.5, label='Compressed')
    plt.title(f'{dist_name.capitalize()} Distribution\n({n_bits} bits zeroed)')
    plt.legend()
    
    # Q-Q plot
    plt.subplot(122)
    stats.probplot(original, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {dist_name.capitalize()} Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'compression_{n_bits}bits.png'))
    plt.close()

def plot_mse_vs_compression(mse_data, file_sizes):
    """Plot MSE vs. compression level for each distribution."""
    bits_to_zero = list(range(8, 17))
    
    # Create summary plots directory
    summary_dir = os.path.join("results", "summary")
    ensure_directory_exists(summary_dir)
    
    plt.figure(figsize=(15, 10))
    
    # MSE vs. bits zeroed
    plt.subplot(211)
    for dist in mse_data:
        plt.semilogy(bits_to_zero, mse_data[dist], marker='o', label=dist.capitalize())
    plt.title('Mean Squared Error vs. Compression Level')
    plt.xlabel('Number of Bits Zeroed')
    plt.ylabel('Mean Squared Error (log scale)')
    plt.grid(True)
    plt.legend()
    
    # File size vs. bits zeroed
    plt.subplot(212)
    for dist in file_sizes:
        # Calculate compression ratio - but make sure to exclude the original file size
        # and only use the compressed file sizes that correspond to bits_to_zero
        original_size = file_sizes[dist][0]
        compressed_sizes = file_sizes[dist][1:]  # Skip the original file size
        
        # Now the dimensions will match
        compression_ratios = [original_size / size for size in compressed_sizes]
        plt.plot(bits_to_zero, compression_ratios, marker='o', label=dist.capitalize())
    
    plt.title('Compression Ratio vs. Compression Level')
    plt.xlabel('Number of Bits Zeroed')
    plt.ylabel('Compression Ratio')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'compression_analysis.png'))
    plt.close()
    
    # Create individual plots for each distribution
    for dist_name in mse_data:
        dist_summary_dir = os.path.join("results", dist_name, "summary")
        ensure_directory_exists(dist_summary_dir)
        
        plt.figure(figsize=(12, 10))
        
        # MSE plot
        plt.subplot(211)
        plt.semilogy(bits_to_zero, mse_data[dist_name], marker='o', color='blue')
        plt.title(f'{dist_name.capitalize()} Distribution: MSE vs. Compression Level')
        plt.xlabel('Number of Bits Zeroed')
        plt.ylabel('Mean Squared Error (log scale)')
        plt.grid(True)
        
        # Compression ratio plot
        plt.subplot(212)
        original_size = file_sizes[dist_name][0]
        compressed_sizes = file_sizes[dist_name][1:]
        compression_ratios = [original_size / size for size in compressed_sizes]
        plt.plot(bits_to_zero, compression_ratios, marker='o', color='green')
        plt.title(f'{dist_name.capitalize()} Distribution: Compression Ratio')
        plt.xlabel('Number of Bits Zeroed')
        plt.ylabel('Compression Ratio')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(dist_summary_dir, 'compression_summary.png'))
        plt.close()

def main():
    # Create the main results directory
    ensure_directory_exists("results")
    
    # Step 1: Generate distributions
    print("Generating distributions...")
    distributions = generate_distributions(n_samples=1000000)  # 1M samples for each distribution
    
    # Step 2-7: Implement compression, save files, and analyze results
    bits_to_zero_range = range(8, 17)  # 8 to 16 bits
    
    # Store MSE and file sizes for different compression levels
    mse_data = {dist_name: [] for dist_name in distributions}
    file_sizes = {dist_name: [] for dist_name in distributions}
    
    # Process each distribution
    for dist_name, original_data in distributions.items():
        print(f"\nProcessing {dist_name} distribution...")
        
        # Create distribution-specific directories
        dist_dir = os.path.join("results", dist_name)
        ensure_directory_exists(dist_dir)
        
        # Create subdirectories for each distribution
        binary_dir = os.path.join(dist_dir, "binary")
        plots_dir = os.path.join(dist_dir, "plots")
        summary_dir = os.path.join(dist_dir, "summary")
        
        for directory in [binary_dir, plots_dir, summary_dir]:
            ensure_directory_exists(directory)
        
        # Save original data
        original_filename = os.path.join(binary_dir, "original.bin")
        save_to_binary(original_data, original_filename)
        original_size = os.path.getsize(original_filename)
        original_stats = calculate_statistics(original_data)
        
        print(f"Original statistics for {dist_name}:")
        for stat, value in original_stats.items():
            print(f"  {stat}: {value}")
        
        # Store original file size
        file_sizes[dist_name].append(original_size)
        
        # Create a statistics file for this distribution
        stats_file = os.path.join(summary_dir, "statistics.txt")
        with open(stats_file, "w") as f:
            f.write(f"{dist_name.upper()} DISTRIBUTION ANALYSIS\n")
            f.write("="*50 + "\n\n")
            
            f.write("ORIGINAL DATA STATISTICS:\n")
            for stat, value in original_stats.items():
                f.write(f"  {stat}: {value}\n")
            f.write("\n")
        
        # Process different compression levels
        for n_bits in bits_to_zero_range:
            print(f"  Compressing with {n_bits} bits zeroed...")
            
            # Compress data
            compressed_data = compress_array(original_data, n_bits)
            
            # Save compressed data
            compressed_filename = os.path.join(binary_dir, f"compressed_{n_bits}bits.bin")
            save_to_binary(compressed_data, compressed_filename)
            compressed_size = os.path.getsize(compressed_filename)
            
            # Calculate statistics for compressed data
            compressed_stats = calculate_statistics(compressed_data)
            
            print(f"  Compressed statistics ({n_bits} bits):")
            for stat, value in compressed_stats.items():
                print(f"    {stat}: {value}")
                
            # Calculate MSE
            mse = calculate_mse(original_data, compressed_data)
            mse_data[dist_name].append(mse)
            print(f"  MSE: {mse}")
            
            # Store file size
            file_sizes[dist_name].append(compressed_size)
            
            # Plot distributions
            plot_distributions(original_data, compressed_data, dist_name, n_bits)
            
            # Print size comparison
            compression_ratio = original_size / compressed_size
            print(f"  File size: {compressed_size} bytes (ratio: {compression_ratio:.2f}x)")
            
            # Append statistics to the distribution's statistics file
            with open(stats_file, "a") as f:
                f.write(f"COMPRESSION WITH {n_bits} BITS ZEROED:\n")
                f.write(f"  MSE: {mse}\n")
                for stat, value in compressed_stats.items():
                    f.write(f"  {stat}: {value}\n")
                f.write(f"  File size: {compressed_size} bytes\n")
                f.write(f"  Compression ratio: {compression_ratio:.2f}x\n")
                f.write("\n")
    
    # Step 8: Plot MSE vs. compression level summary plots
    plot_mse_vs_compression(mse_data, file_sizes)
    
    # Create overall summary file
    summary_dir = os.path.join("results", "summary")
    ensure_directory_exists(summary_dir)
    
    with open(os.path.join(summary_dir, "compression_summary.txt"), "w") as f:
        f.write("LOSSY FLOATING-POINT COMPRESSION ANALYSIS\n")
        f.write("="*50 + "\n\n")
        
        f.write("SUMMARY OF RESULTS:\n")
        f.write("-"*30 + "\n\n")
        
        for dist_name in mse_data:
            f.write(f"{dist_name.upper()} DISTRIBUTION:\n")
            
            # Original file size
            original_size = file_sizes[dist_name][0]
            f.write(f"  Original file size: {original_size} bytes\n\n")
            
            f.write("  Compression results:\n")
            for i, n_bits in enumerate(range(8, 17)):
                compressed_size = file_sizes[dist_name][i+1]
                ratio = original_size / compressed_size
                mse = mse_data[dist_name][i]
                
                f.write(f"    {n_bits} bits zeroed:\n")
                f.write(f"      File size: {compressed_size} bytes\n")
                f.write(f"      Compression ratio: {ratio:.2f}x\n")
                f.write(f"      MSE: {mse:.8e}\n\n")
            
        f.write("\nCONCLUSIONS:\n")
        f.write("-"*30 + "\n\n")
        
        # Find optimal compression levels based on different criteria
        best_compression = {}
        lowest_mse = {}
        
        for dist_name in mse_data:
            # Best compression ratio
            compression_ratios = [file_sizes[dist_name][0] / size for size in file_sizes[dist_name][1:]]
            best_idx = compression_ratios.index(max(compression_ratios))
            best_compression[dist_name] = best_idx + 8  # Convert index to bits zeroed
            
            # Lowest MSE (will typically be at lowest compression)
            lowest_idx = mse_data[dist_name].index(min(mse_data[dist_name]))
            lowest_mse[dist_name] = lowest_idx + 8
        
        f.write("Optimal compression levels:\n")
        for dist_name in best_compression:
            f.write(f"  {dist_name.capitalize()} distribution:\n")
            f.write(f"    Best compression: {best_compression[dist_name]} bits zeroed\n")
            f.write(f"    Lowest error: {lowest_mse[dist_name]} bits zeroed\n\n")
        
        f.write("USE CASE RECOMMENDATIONS:\n")
        f.write("  High-precision computing: 8 bits zeroed (minimal precision loss)\n")
        f.write("  Balanced approach: 12 bits zeroed (good compression with moderate precision loss)\n")
        f.write("  Limited storage resources: 16 bits zeroed (maximum compression with significant precision loss)\n")
  
    print("\nAnalysis complete! Results saved to disk.")
    print(f"Results are organized in the 'results/' directory:")
    print(f"  - results/uniform/binary/  : Binary files for uniform distribution")
    print(f"  - results/uniform/plots/   : Plots for uniform distribution")
    print(f"  - results/uniform/summary/ : Summary for uniform distribution")
    print(f"  - results/gaussian/        : Similar structure for gaussian distribution")
    print(f"  - results/exponential/     : Similar structure for exponential distribution")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")