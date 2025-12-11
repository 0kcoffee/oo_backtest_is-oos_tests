# IS/OOS Validation Script

## Description

`main.py` is a comprehensive In-Sample/Out-of-Sample (IS/OOS) validation tool for Option Omega strategy backtests. The script performs rigorous statistical validation by:

1. **Splitting data** into three periods: In-Sample (IS), Validation OOS, and Final OOS
2. **Calculating performance metrics** (CAGR, Max Drawdown, Sharpe Ratio, Sortino Ratio, Win Rate, MAR Ratio, etc.) for each period
3. **Comparing performance** between periods using ratio-based validation to detect overfitting
4. **Running Monte Carlo simulations** on the OOS data to assess robustness and validate against 8 industry-grade checks
5. **Generating a comprehensive HTML report** with visualizations including:
   - Cumulative equity curve with period boundaries
   - P/L per contract distribution histogram
   - Monte Carlo equity curves (best, median, worst cases)

The script supports flexible trade allocation strategies (percentage-based or fixed contracts) and dynamically scales P/L based on the allocation method.

## User Input Variables

All configuration variables are located at the top of the script (lines 17-27):

### Data Input
- **`CSV_PATH`** (command-line argument, optional)
  - Path to the CSV file containing trade log data
  - Passed as a command-line argument when running the script
  - Default: `"example_backtests/Monday-2-4-DC.csv"`
  - Required columns: `Date Opened`, `Premium`, `Avg. Closing Cost`, `No. of Contracts`
  - Optional columns: `P/L` (if available, used for more accurate per-contract P/L calculation)
  - Example: `python main.py strategy_backtest_logs/trade-log-rics.csv`

### Data Splitting
- **`IS_FRAC`** (float, default: `0.6`)
  - Fraction of data to use for In-Sample (training) period
  - Must be between 0 and 1
  - Example: `0.6` means 60% of trades are used for IS

- **`VAL_FRAC`** (float, default: `0.2`)
  - Fraction of data to use for Validation OOS (development) period
  - Must be between 0 and 1
  - Remaining fraction `1 - IS_FRAC - VAL_FRAC` is used for Final OOS (testing)
  - Example: `0.2` means 20% of trades are used for Validation OOS, leaving 20% for Final OOS

### Monte Carlo Simulation
- **`N_MC_PATHS`** (int, default: `1000`)
  - Number of Monte Carlo simulation paths to run
  - Higher values provide more statistical confidence but take longer to compute
  - Recommended: 1000-10000 for production analysis
  - Example: `1000` runs 1000 bootstrap simulations

### Initial Capital
- **`RISK_CAPITAL`** (float, default: `100000`)
  - Initial equity/capital to start the backtest
  - Used for calculating returns, equity curves, and contract allocation
  - Example: `100000` means starting with $100,000

### Trade Allocation Strategy
- **`ALLOCATION_TYPE`** (string, default: `"percent"`)
  - Allocation method: `"percent"` or `"contracts"`
  - **`"percent"`**: Allocates a percentage of current equity per trade
  - **`"contracts"`**: Uses a fixed number of contracts per trade
  - Example: `"percent"` for dynamic position sizing

- **`ALLOCATION_VALUE`** (float, default: `0.04`)
  - Interpretation depends on `ALLOCATION_TYPE`:
    - If `ALLOCATION_TYPE = "percent"`: Percentage of equity per trade (0.04 = 4%)
    - If `ALLOCATION_TYPE = "contracts"`: Fixed number of contracts per trade
  - For percentage-based: Uses `Margin Req.` column to calculate number of contracts
  - Contracts are rounded down (no fractional contracts)
  - Example: `0.04` with `"percent"` means 4% of equity per trade

## How to Run

### Prerequisites
- Python 3.7+

### Installation

1. **Install dependencies** from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or if using a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Basic Usage

1. **Configure the script** (optional) by editing the variables at the top of `main.py`:
   ```python
   IS_FRAC = 0.6
   VAL_FRAC = 0.2
   # ... other variables
   ```

2. **Run the script** with your CSV file:
   ```bash
   python main.py strategy_backtest_logs/your_trade_log.csv
   ```
   
   Or use the default example CSV path (example_backtests/Monday-2-4-DC.csv):
   ```bash
   python main.py
   ```
   
   Or if using a virtual environment:
   ```bash
   .venv/bin/python main.py strategy_backtest_logs/your_trade_log.csv
   ```
   
   For help and usage information:
   ```bash
   python main.py --help
   ```

3. **View the results**:
   - Terminal output shows color-coded validation results and statistics
   - HTML report is saved as: `is_oos_report_{filename}.html`
   - Example: `is_oos_report_Monday-2-4-DC.html`
   - Open the HTML file in a web browser to view the full report with visualizations

### Example Usage

```bash
# Run with a specific CSV file
python main.py strategy_backtest_logs/trade-log-rics.csv

# Run with default CSV path (strategy_backtest_logs/Monday-2-4-DC.csv)
python main.py
```

### Example Configuration

You can customize the script by editing variables at the top of `main.py`:

```python
# Split: 60% IS, 20% Validation OOS, 20% Final OOS
IS_FRAC = 0.6
VAL_FRAC = 0.2

# Run 5000 Monte Carlo simulations for robust validation
N_MC_PATHS = 5000

# Start with $100,000
RISK_CAPITAL = 100000

# Allocate 5% of equity per trade
ALLOCATION_TYPE = "percent"
ALLOCATION_VALUE = 0.05
```

## Output

### Terminal Output
- Overall statistics and trade allocation summary
- Performance metrics for each period (IS, Validation OOS, Final OOS)
- Color-coded validation verdicts:
  - 🟢 Green: Good signs (OOS performance within acceptable range)
  - 🟡 Yellow: Warning signs (some degradation)
  - 🔴 Red: Bad signs (significant overfitting or degradation)
- Monte Carlo validation results with 8 industry-grade checks

### HTML Report
- **Overall Summary**: Key metrics comparison across all periods
- **Cumulative Equity Curve**: Single continuous line with vertical markers at period boundaries
- **P/L Per Contract Distribution**: Histogram showing trade P/L distribution
- **Monte Carlo Validation**: 
  - Summary statistics (worst, median, best cases)
  - Equity curves for best, median, and worst MC paths
  - Detailed validation checks and verdicts

## CSV File Requirements

The input CSV file must contain the following columns:

**Required:**
- `Date Opened`: Trade opening date (will be parsed as datetime)
- `Premium`: Total premium received/paid for the trade
- `Avg. Closing Cost`: Total closing cost for the trade
- `No. of Contracts`: Number of contracts in the trade

**Optional but recommended:**
- `P/L`: Actual profit/loss (if available, provides more accurate per-contract P/L)
- `Margin Req.`: Margin requirement per contract (required for percentage-based allocation)
- `Time Opened`: Time of trade opening (optional, for more precise sorting)

**Note:** The script automatically calculates per-contract values from the total values and number of contracts.

## Validation Logic

The script performs two types of validation:

1. **IS/OOS Comparison**: Compares performance ratios (not absolute values) between periods:
   - Good signs: OOS CAGR within ±50% of IS, similar drawdowns, stable win rates
   - Bad signs: Metrics explode by 3-5x, significant degradation

2. **Monte Carlo Validation**: 8 industry-grade checks:
   - Expected Max Drawdown vs. Observed
   - Probability of Positive Return (PoPR)
   - Left-Tail Severity
   - Variance of Pathwise Outcomes
   - Drawdown–Return Efficiency
   - Catastrophic Failure Probability
   - Stability of Trade/Return Sequence
   - Noise-to-Signal Ratio (NSR)

## Tips

- **For accurate results**: Ensure your CSV has accurate `P/L` column if available
- **For percentage allocation**: Include `Margin Req.` column to properly calculate contracts
- **For robust validation**: Use at least 1000 Monte Carlo paths (more is better)
- **For meaningful splits**: Ensure you have enough trades (recommended: 100+ trades)
- **Fractional contracts**: The script automatically rounds down to whole contracts (no fractional contracts in real trading)
