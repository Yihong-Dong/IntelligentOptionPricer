# Intelligent Option Pricer

Author: DONG Yihong  
Date: 2025-03-22

## Environment

Use the following command to config the environment with conda.

```bash
conda create --name option-pricer -y
conda activate option-pricer
conda install tensorflow -y
conda install conda-forge::yfinance -y
conda install scipy matplotlib -y
```

If with Linux system, the environment can be set up with the following command.

```bash
conda env create -f environment.yml
```

## Running

### Model Running or Benchmark

In the current directory, use `python benchmark.py`, input your options to run the prediction or benchmark.

### Risk Free Yield Curve

In the current directory, use `python risk_free_curve.py`, a dialogue will pop up showing the US Treasury Yield Curve.

See `US Treasury Yield Curve.png` image as an example.

### Volatility Surface

In the current directory, use `python volatility_surface.py`, a dialogue will pop up showing the Volatility Surface of the designated ticker (^SPX by default, we can change the argument to show the volatility of other stocks or ETFs as well, like NVDA, BABA, etc.).

See `Volatility Surface for SPX.png` image as an example.

## Outcome

### Test 1: S = 5400, T = 1 Year

Benchmark Results and Error Comparison:

| Method                  | Call Price | Put Price | Time (ms) | Call Error (%) | Put Error (%) |
|-------------------------|------------|-----------|-----------|----------------|---------------|
| **Black-Scholes**       | 1002.88    | 510.52    | 0.50      | 46.11          | 118.87        |
| Monte Carlo (1,000 sim) | 1048.40    | 493.13    | 8.13      | 52.74          | 111.42        |
| Monte Carlo (10,000 sim)| 984.21     | 516.21    | 163.38    | 43.39          | 121.31        |
| **Binomial Tree**       | 1003.98    | 511.62    | 0.55      | 46.27          | 119.35        |
| **Neural Network**      | 5410.16    | -0.26     | 129872.25 | 688.19         | 100.11        |

---

### Test 2: S = 6000, T = 0.5 Years

Benchmark Results and Error Comparison:

| Method                    | Call Price | Put Price | Time (ms) | Call Error (%) | Put Error (%) |
|---------------------------|------------|-----------|-----------|----------------|---------------|
| **Black-Scholes**         | 454.43     | 660.66    | 0.43      | 226.41         | 72.37         |
| Monte Carlo (1,000 sim)   | 449.07     | 663.22    | 7.57      | 222.56         | 73.04         |
| Monte Carlo (10,000 sim)  | 449.12     | 670.09    | 172.00    | 222.60         | 74.83         |
| **Binomial Tree**         | 454.65     | 660.88    | 0.53      | 226.57         | 72.43         |
| **Neural Network**        | 6012.87    | 0.26      | 127534.12 | 4218.97        | 99.93         |

---

## Conclusion

The results from the benchmark tests highlight the performance and accuracy of different option pricing methods:

1. **Black-Scholes and Binomial Tree Models**:
   - Both methods demonstrated **high computational efficiency**, with execution times under 1 millisecond.
   - However, they exhibited **significant errors** in pricing, particularly for call options (errors ranging from 46% to 226%). This suggests potential issues with the input parameters or assumptions in the models.

2. **Monte Carlo Simulation**:
   - The Monte Carlo method showed **reasonable accuracy** compared to Black-Scholes and Binomial Tree, with errors slightly lower in some cases.
   - As expected, increasing the number of simulations (from 1,000 to 10,000) improved accuracy but at the cost of **higher computation time**.

3. **Neural Network Model**:
   - The neural network performed poorly, with **extremely high errors** (up to 4218.97% for call options) and **prohibitively long computation times** (over 127 seconds). This indicates that the model may require further tuning, more training data, or a different architecture to improve its performance.

### Key Takeaways:
- **Black-Scholes and Binomial Tree** are suitable for fast, approximate pricing but may need refinement to reduce errors.
- **Monte Carlo** provides a balance between accuracy and computation time, especially with a higher number of simulations.
- The **Neural Network** model, while promising, is not yet ready for practical use and requires significant improvements.

### Future Work:
- Investigate and refine the input parameters for Black-Scholes and Binomial Tree to reduce errors.
- Optimize the neural network architecture and training process to improve accuracy and reduce computation time.
- Explore additional variance reduction techniques for Monte Carlo simulations to further enhance performance.
