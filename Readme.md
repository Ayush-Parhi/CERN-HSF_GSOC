# Lossy Floating-Point Compression

This project explores lossy floating-point compression by manipulating the least significant bits of the mantissa. It analyzes how different compression levels affect various probability distributions and evaluates the trade-offs between storage savings and precision loss.

## Overview

Floating-point numbers follow the IEEE 754 standard, with a 32-bit single-precision format consisting of:
- 1 sign bit
- 8 exponent bits
- 23 mantissa bits

This project systematically zeroes out 8-16 of the least significant mantissa bits to investigate compression efficiency across different statistical distributions.

## Features

- Generates 1 million floating-point samples from three distributions:
  - Uniform [0, 1]
  - Gaussian (Normal) with mean=0, std=1
  - Exponential with scale=1
- Implements bit-level manipulation of IEEE 754 floating-point representation
- Analyzes compression impact using:
  - Statistical parameters (mean, std, min, max, median, skewness, kurtosis)
  - Mean Squared Error (MSE) calculations
  - Histograms and Q-Q plots
  - Compression ratio measurements

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- SciPy

## Usage

```bash
python main.py
```

The script will:
1. Generate the distributions
2. Apply compression at different bit levels (8-16)
3. Save original and compressed data to binary files
4. Calculate statistical metrics and MSE
5. Generate visualizations
6. Create comprehensive summary reports

## Project Structure

```
results/
├── uniform/
│   ├── binary/      # Binary files of original and compressed data
│   ├── plots/       # Distribution visualizations
│   └── summary/     # Statistical analysis reports
├── gaussian/        # Similar structure for gaussian distribution
├── exponential/     # Similar structure for exponential distribution
└── summary/         # Overall comparison and recommendations
```

## Key Findings

### Compression Efficiency

The compression ratio improves as more bits are zeroed:
- 8 bits: ~1.2-1.5x compression
- 12 bits: ~2.0-2.5x compression
- 16 bits: ~3.0-4.0x compression

### Error Analysis

Mean Squared Error (MSE) increases exponentially with the number of bits zeroed, but the impact varies by distribution:
- Uniform distribution: Most sensitive to compression
- Gaussian distribution: Moderate sensitivity
- Exponential distribution: Least affected by compression

### Statistical Impact

- Measures of central tendency (mean, median) are generally well-preserved
- Measures of dispersion (std, min, max) show increasing deviation with higher compression
- Higher-order moments (skewness, kurtosis) are most affected by compression

## Use Case Recommendations

Based on the analysis:

1. **High-precision computing (8 bits zeroed)**
   - Minimal precision loss (<0.01% in statistical parameters)
   - Modest storage savings (~1.5x)
   - Suitable for scientific computing requiring high accuracy

2. **Balanced approach (12 bits zeroed)**
   - Moderate precision loss (~0.1-1% in statistical parameters)
   - Good compression ratio (~2.5x)
   - Appropriate for data analysis and visualization applications

3. **Limited storage resources (16 bits zeroed)**
   - Significant precision loss (>1% in statistical parameters)
   - Maximum compression (~4x)
   - Suitable for applications where trends and patterns are more important than exact values

## Visualization Examples

The project generates various visualizations to help understand the impact of compression:

- Histograms comparing original and compressed distributions
- Q-Q plots to assess normality
- MSE vs. compression level plots
- Compression ratio analysis

## License

[MIT License](LICENSE)

## Acknowledgments

This project was developed as part of the Google Summer of Code (GSoC) task for CERN-HSF.
