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
