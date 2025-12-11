import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import argparse

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# Default CSV path (can be overridden via command-line argument)
DEFAULT_CSV_PATH = "example_backtests/Monday-2-4-DC.csv"
IS_FRAC = 0.6
VAL_FRAC = 0.2
N_MC_PATHS = 1000
RISK_CAPITAL = 100000

# Trade allocation configuration
# Set ALLOCATION_TYPE to either "percent" or "contracts"
# If "percent": ALLOCATION_VALUE is the percentage of equity to allocate per trade (e.g., 0.1 = 10%)
# If "contracts": ALLOCATION_VALUE is the fixed number of contracts per trade
ALLOCATION_TYPE = "percent"  # Options: "percent" or "contracts"
ALLOCATION_VALUE = 0.04  # % of equity per trade, or fixed number of contracts

def load_data(path):
    df = pd.read_csv(path)
    df["Date Opened"] = pd.to_datetime(df["Date Opened"])
    if "Time Opened" in df.columns:
        df["Time Opened"] = pd.to_timedelta(df["Time Opened"] + "")
        df = df.sort_values(["Date Opened", "Time Opened"])
    else:
        df = df.sort_values("Date Opened")
    df = df.reset_index(drop=True)
    return df

def add_per_contract_pl(df):
    """
    Calculate per-contract P/L based on Premium, Avg. Closing Cost, and No. of Contracts.
    Also calculates per-contract opening premium and closing cost.
    
    If P/L is available, use it directly for accuracy, otherwise calculate from Premium and Avg. Closing Cost.
    """
    if "No. of Contracts" not in df.columns:
        raise ValueError("CSV must contain 'No. of Contracts' column.")
    if "Premium" not in df.columns:
        raise ValueError("CSV must contain 'Premium' column.")
    if "Avg. Closing Cost" not in df.columns:
        raise ValueError("CSV must contain 'Avg. Closing Cost' column.")
    
    # Calculate per-contract values
    df["contracts"] = df["No. of Contracts"].fillna(1).astype(float)
    df["premium_per_contract"] = df["Premium"] / df["contracts"]
    df["closing_cost_per_contract"] = df["Avg. Closing Cost"] / df["contracts"]
    
    # Calculate per-contract P/L
    # Use actual P/L if available (most accurate, includes all fees/commissions)
    if "P/L" in df.columns:
        df["pl_per_contract"] = df["P/L"] / df["contracts"]
    else:
        # Fallback: calculate from Premium and Avg. Closing Cost
        # Note: This may not match actual P/L due to fees, commissions, etc.
        df["pl_per_contract"] = df["premium_per_contract"] - df["closing_cost_per_contract"]
    
    return df

def calculate_contracts_and_pl(df, current_equity, allocation_type=ALLOCATION_TYPE, allocation_value=ALLOCATION_VALUE):
    """
    Calculate the number of contracts to trade and adjusted P/L based on allocation strategy.
    
    Args:
        df: DataFrame with trade data
        current_equity: Current equity value (for percentage-based allocation)
        allocation_type: "percent" or "contracts"
        allocation_value: Percentage (0-1) or number of contracts
    
    Returns:
        DataFrame with added columns: contracts_to_trade, adjusted_pl, adjusted_margin
    """
    df = df.copy()
    
    if "Margin Req." not in df.columns:
        raise ValueError("CSV must contain 'Margin Req.' column for contract calculation.")
    
    if "pl_per_contract" not in df.columns:
        raise ValueError("DataFrame must have 'pl_per_contract' column. Call add_per_contract_pl() first.")
    
    if "contracts" not in df.columns:
        raise ValueError("DataFrame must have 'contracts' column. Call add_per_contract_pl() first.")
    
    # Get margin requirement per contract
    # If Margin Req. is total, divide by original contracts; otherwise it's per contract
    # We'll assume Margin Req. is total margin, so per-contract margin = Margin Req. / contracts
    df["margin_per_contract"] = df["Margin Req."] / df["contracts"].replace(0, 1)
    
    if allocation_type == "percent":
        # Calculate number of contracts based on percentage of equity
        # contracts = (equity * allocation_value) / margin_per_contract
        allocation_amount = current_equity * allocation_value
        # Round DOWN to avoid fractional contracts (floor operation)
        df["contracts_to_trade"] = np.floor(allocation_amount / df["margin_per_contract"]).fillna(0).astype(int)
        # Ensure at least 1 contract if we have allocation and margin is sufficient
        df.loc[(df["contracts_to_trade"] == 0) & (allocation_amount > 0) & (df["margin_per_contract"] > 0), "contracts_to_trade"] = 1
    elif allocation_type == "contracts":
        # Use fixed number of contracts (round down to ensure integer)
        df["contracts_to_trade"] = int(np.floor(allocation_value))
    else:
        raise ValueError(f"Invalid ALLOCATION_TYPE: {allocation_type}. Must be 'percent' or 'contracts'.")
    
    # Calculate adjusted P/L based on contracts_to_trade
    # Scale the per-contract P/L by the number of contracts we're actually trading
    df["adjusted_pl"] = df["pl_per_contract"] * df["contracts_to_trade"]
    
    # Calculate adjusted margin requirement
    df["adjusted_margin"] = df["margin_per_contract"] * df["contracts_to_trade"]
    
    return df

def add_returns(df, use_adjusted=True):
    """
    Calculate returns based on adjusted P/L (if available) or original P/L.
    
    Args:
        df: DataFrame with trade data
        use_adjusted: If True, use adjusted_pl if available; otherwise use P/L
    """
    # Determine which P/L column to use
    if use_adjusted and "adjusted_pl" in df.columns:
        pl_col = "adjusted_pl"
    elif "P/L" in df.columns:
        pl_col = "P/L"
    else:
        raise ValueError("CSV must contain a 'P/L' column or 'adjusted_pl' column.")
    
    # Prefer "Funds at Close" as it represents actual account balance
    # This gives more accurate returns than margin-based calculation
    if "Funds at Close" in df.columns:
        # Calculate return as change in funds / previous funds
        # This is equivalent to P/L / previous funds
        base = df["Funds at Close"].shift(1).replace(0, np.nan)
        ret = df[pl_col] / base
    elif "adjusted_margin" in df.columns:
        # Use adjusted margin for return calculation
        margin = df["adjusted_margin"].replace(0, np.nan)
        ret = df[pl_col] / margin
        # Cap returns at -1.0 (100% loss) to prevent negative equity
        ret = ret.clip(lower=-1.0)
    elif "Margin Req." in df.columns:
        # Fallback to margin-based returns, but cap at -1.0 to prevent account wipeout
        margin = df["Margin Req."].replace(0, np.nan)
        ret = df[pl_col] / margin
        # Cap returns at -1.0 (100% loss) to prevent negative equity
        ret = ret.clip(lower=-1.0)
    else:
        raise ValueError("CSV must contain either 'Margin Req.', 'adjusted_margin', or 'Funds at Close'.")
    
    df["trade_ret"] = ret.fillna(0.0)
    return df

def split_is_oos(df):
    n = len(df)
    n_is = int(n * IS_FRAC)
    n_val = int(n * VAL_FRAC)
    is_df = df.iloc[:n_is].copy()
    val_df = df.iloc[n_is:n_is + n_val].copy()
    test_df = df.iloc[n_is + n_val:].copy()
    return is_df, val_df, test_df

def equity_curve(rets, starting_capital=RISK_CAPITAL, equity_series=None):
    """
    Calculate equity curve from returns.
    
    Args:
        rets: Array of returns
        starting_capital: Initial capital
        equity_series: Optional array of equity values to use as base (for dynamic allocation)
    
    Returns:
        Array of equity values
    """
    if equity_series is not None:
        # Use provided equity series as base for each return
        eq = []
        for i, r in enumerate(rets):
            if i < len(equity_series):
                base_equity = equity_series[i]
            else:
                base_equity = starting_capital if i == 0 else eq[-1]
            # Cap returns at -1.0 to prevent negative equity
            r_capped = max(r, -1.0)
            new_eq = base_equity * (1 + r_capped)
            # Ensure equity never goes below a small positive value
            eq.append(max(new_eq, 0.01))
        return np.array(eq)
    else:
        # Standard calculation with fixed starting capital
        eq = [starting_capital]
        for r in rets:
            # Cap returns at -1.0 to prevent negative equity
            # This handles cases where losses exceed the margin requirement
            r_capped = max(r, -1.0)
            new_eq = eq[-1] * (1 + r_capped)
            # Ensure equity never goes below a small positive value
            eq.append(max(new_eq, 0.01))
        return np.array(eq)

def max_drawdown(equity):
    if len(equity) == 0:
        return 0.0
    # Ensure all values are positive
    equity = np.maximum(equity, 0.01)
    peak = np.maximum.accumulate(equity)
    # Avoid division by zero
    peak = np.maximum(peak, 0.01)
    dd = equity / peak - 1.0
    return dd.min()

def annualized_return(df, equity):
    if df.empty:
        return 0.0
    days = (df["Date Opened"].iloc[-1] - df["Date Opened"].iloc[0]).days
    if days <= 0:
        return 0.0
    years = days / 365.0
    if years <= 0:
        return 0.0
    # Handle edge cases where equity goes to zero or becomes invalid
    if equity[0] <= 0 or equity[-1] <= 0:
        return -1.0  # -100% return if equity is depleted
    ratio = equity[-1] / equity[0]
    if ratio <= 0:
        return -1.0
    try:
        cagr = ratio ** (1 / years) - 1
        # Handle NaN or infinite values
        if not np.isfinite(cagr):
            return -1.0
        return cagr
    except (ValueError, ZeroDivisionError):
        return -1.0

def stats_block(name, df, starting_equity=RISK_CAPITAL, use_adjusted=True):
    if df.empty:
        print(f"{name}: no trades")
        return None
    
    # Determine which P/L column to use for stats
    # When using allocation, always use adjusted_pl (scaled based on contracts_to_trade)
    if use_adjusted and "adjusted_pl" in df.columns:
        pl_col = "adjusted_pl"
    else:
        pl_col = "P/L"
    
    wins = (df[pl_col] > 0).sum()
    losses = (df[pl_col] <= 0).sum()
    win_rate = wins / len(df) if len(df) else 0
    avg_pl = df[pl_col].mean()
    
    # Calculate equity curve with proper allocation logic
    if ALLOCATION_TYPE == "percent" and "adjusted_pl" in df.columns:
        # Recalculate with running equity for percentage-based allocation
        eq_values = [starting_equity]
        rets = []
        for i in range(len(df)):
            current_eq = eq_values[-1]
            # Recalculate contracts for this trade based on current equity
            row = df.iloc[i:i+1].copy()
            row = calculate_contracts_and_pl(row, current_eq, ALLOCATION_TYPE, ALLOCATION_VALUE)
            # Ensure contracts_to_trade is integer (should already be from calculate_contracts_and_pl, but double-check)
            row["contracts_to_trade"] = row["contracts_to_trade"].astype(int)
            
            # Calculate return based on adjusted P/L and current equity
            adjusted_pl = row["adjusted_pl"].iloc[0]
            ret = adjusted_pl / current_eq if current_eq > 0 else 0
            ret = max(ret, -1.0)  # Cap at -100%
            rets.append(ret)
            
            new_eq = current_eq * (1 + ret)
            eq_values.append(max(new_eq, 0.01))
        eq = np.array(eq_values)
    else:
        # Use pre-calculated returns or calculate from adjusted P/L
        if "trade_ret" in df.columns:
            eq = equity_curve(df["trade_ret"].values, starting_equity)
        elif "adjusted_pl" in df.columns:
            # Calculate returns from adjusted P/L
            if "adjusted_margin" in df.columns:
                rets = (df["adjusted_pl"] / df["adjusted_margin"]).fillna(0).clip(lower=-1.0)
            else:
                rets = (df["adjusted_pl"] / starting_equity).fillna(0).clip(lower=-1.0)
            eq = equity_curve(rets.values, starting_equity)
        else:
            eq = equity_curve(df["trade_ret"].values, starting_equity)
    
    mdd = max_drawdown(eq)
    cagr = annualized_return(df, eq)
    
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Trades: {len(df)}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Avg P/L per trade: {avg_pl:.2f}")
    if use_adjusted and "contracts_to_trade" in df.columns:
        avg_contracts = df["contracts_to_trade"].mean()
        print(f"Avg contracts per trade: {avg_contracts:.2f}")
    print(f"CAGR (approx): {cagr:.2%}")
    print(f"Max drawdown (on unit capital): {mdd:.2%}")
    print(f"Starting equity: ${eq[0]:,.2f}")
    print(f"Ending equity: ${eq[-1]:,.2f}")
    print(f"Total return: {(eq[-1] / eq[0] - 1):.2%}")
    
    # Return equity curve for further analysis
    return eq

def calculate_metrics(df, eq, starting_equity=RISK_CAPITAL):
    """
    Calculate key metrics for a dataset.
    Returns a dictionary with metrics.
    """
    if df.empty or len(eq) == 0:
        return None
    
    # Determine which P/L column to use
    if "adjusted_pl" in df.columns:
        pl_col = "adjusted_pl"
    else:
        pl_col = "P/L"
    
    wins = (df[pl_col] > 0).sum()
    total_trades = len(df)
    win_rate = wins / total_trades if total_trades > 0 else 0
    avg_pl = df[pl_col].mean()
    
    mdd = max_drawdown(eq)
    cagr = annualized_return(df, eq)
    total_return = (eq[-1] / eq[0] - 1) if eq[0] > 0 else 0
    
    # Calculate MAR ratio (CAGR / abs(Max Drawdown %))
    mar = cagr / abs(mdd) if mdd < 0 else float('inf') if cagr > 0 else 0
    
    # Calculate P/L variance (standard deviation of trade P/L)
    pl_std = df[pl_col].std() if total_trades > 1 else 0
    pl_variance = df[pl_col].var() if total_trades > 1 else 0
    
    return {
        'win_rate': win_rate,
        'avg_pl': avg_pl,
        'cagr': cagr,
        'mdd': mdd,
        'total_return': total_return,
        'mar': mar,
        'starting_equity': eq[0],
        'ending_equity': eq[-1],
        'total_trades': total_trades,
        'pl_std': pl_std,
        'pl_variance': pl_variance
    }

def validate_oos_performance(is_metrics, oos_metrics, oos_name="OOS"):
    """
    Validate OOS performance against IS metrics using ratio-based comparisons.
    Returns a verdict dictionary with color-coded messages.
    """
    if is_metrics is None or oos_metrics is None:
        return None
    
    verdicts = []
    overall_status = "GOOD"  # GOOD, WARNING, FAIL
    interpretation = "UNKNOWN"
    
    # Check 1: Still profitable
    is_profitable = oos_metrics['total_return'] > 0
    if not is_profitable:
        verdicts.append((Colors.RED + "FAIL: OOS is not profitable" + Colors.RESET, "FAIL"))
        overall_status = "FAIL"
        interpretation = "❌ Strategy fails - not profitable OOS"
    else:
        verdicts.append((Colors.GREEN + f"✓ OOS is profitable ({oos_metrics['total_return']:.2%})" + Colors.RESET, "GOOD"))
    
    # Check 2: CAGR ratio comparison (compare ratios, not absolute values)
    # Good: OOS CAGR within ±50% of IS CAGR (ratio between 0.5 and 1.5)
    if is_metrics['cagr'] != 0:
        cagr_ratio = oos_metrics['cagr'] / is_metrics['cagr'] if is_metrics['cagr'] > 0 else oos_metrics['cagr'] / abs(is_metrics['cagr']) if is_metrics['cagr'] < 0 else 0
    else:
        cagr_ratio = float('inf') if oos_metrics['cagr'] > 0 else 0
    
    if cagr_ratio < 0.2:  # OOS CAGR is less than 20% of IS (exploded downward by 5x)
        verdicts.append((Colors.RED + f"FAIL: OOS CAGR ratio ({cagr_ratio:.2f}x) - collapsed (IS: {is_metrics['cagr']:.2%}, OOS: {oos_metrics['cagr']:.2%})" + Colors.RESET, "FAIL"))
        overall_status = "FAIL"
        interpretation = "❌ Strategy fails - edge collapsed"
    elif cagr_ratio < 0.5:  # OOS CAGR is less than 50% of IS
        verdicts.append((Colors.YELLOW + f"WARNING: OOS CAGR ratio ({cagr_ratio:.2f}x) - significantly lower (IS: {is_metrics['cagr']:.2%}, OOS: {oos_metrics['cagr']:.2%})" + Colors.RESET, "WARNING"))
        if overall_status == "GOOD":
            overall_status = "WARNING"
    elif cagr_ratio <= 1.5:  # Within ±50% range
        if cagr_ratio > 1.0:
            verdicts.append((Colors.GREEN + f"✓ OOS CAGR ratio ({cagr_ratio:.2f}x) - slightly better (IS: {is_metrics['cagr']:.2%}, OOS: {oos_metrics['cagr']:.2%})" + Colors.RESET, "GOOD"))
            if interpretation == "UNKNOWN":
                interpretation = "⭐ Strong evidence of real edge"
        else:
            verdicts.append((Colors.GREEN + f"✓ OOS CAGR ratio ({cagr_ratio:.2f}x) - within acceptable range (IS: {is_metrics['cagr']:.2%}, OOS: {oos_metrics['cagr']:.2%})" + Colors.RESET, "GOOD"))
    elif cagr_ratio <= 3.0:  # OOS CAGR is 1.5-3x IS (exploded upward)
        verdicts.append((Colors.YELLOW + f"WARNING: OOS CAGR ratio ({cagr_ratio:.2f}x) - much better (IS: {is_metrics['cagr']:.2%}, OOS: {oos_metrics['cagr']:.2%}) - edge may be conditional" + Colors.RESET, "WARNING"))
        if overall_status == "GOOD":
            overall_status = "WARNING"
        if interpretation == "UNKNOWN":
            interpretation = "⚠️ Edge may be conditional or OOS period unusually favorable"
    else:  # OOS CAGR is more than 3x IS (exploded upward by 3-5x+)
        verdicts.append((Colors.RED + f"FAIL: OOS CAGR ratio ({cagr_ratio:.2f}x) - exploded upward (IS: {is_metrics['cagr']:.2%}, OOS: {oos_metrics['cagr']:.2%}) - possible overfitting" + Colors.RESET, "FAIL"))
        overall_status = "FAIL"
        interpretation = "❌ Possible overfitting or weak edge"
    
    # Check 3: Max Drawdown ratio comparison (expect similar or slightly lower)
    if is_metrics['mdd'] != 0:
        mdd_ratio = abs(oos_metrics['mdd'] / is_metrics['mdd']) if is_metrics['mdd'] < 0 else abs(oos_metrics['mdd']) if oos_metrics['mdd'] < 0 else 1.0
    else:
        mdd_ratio = float('inf') if oos_metrics['mdd'] < 0 else 1.0
    
    if mdd_ratio > 3.0:  # OOS drawdown is more than 3x IS drawdown (exploded)
        verdicts.append((Colors.RED + f"FAIL: OOS Max Drawdown ratio ({mdd_ratio:.2f}x) - exploded (IS: {is_metrics['mdd']:.2%}, OOS: {oos_metrics['mdd']:.2%})" + Colors.RESET, "FAIL"))
        overall_status = "FAIL"
    elif mdd_ratio > 1.5:
        verdicts.append((Colors.YELLOW + f"WARNING: OOS Max Drawdown ratio ({mdd_ratio:.2f}x) - worse than IS (IS: {is_metrics['mdd']:.2%}, OOS: {oos_metrics['mdd']:.2%})" + Colors.RESET, "WARNING"))
        if overall_status == "GOOD":
            overall_status = "WARNING"
    elif mdd_ratio <= 1.0:  # OOS DD is similar or lower (good sign)
        verdicts.append((Colors.GREEN + f"✓ OOS Max Drawdown ratio ({mdd_ratio:.2f}x) - similar or lower (IS: {is_metrics['mdd']:.2%}, OOS: {oos_metrics['mdd']:.2%})" + Colors.RESET, "GOOD"))
    else:
        verdicts.append((Colors.GREEN + f"✓ OOS Max Drawdown ratio ({mdd_ratio:.2f}x) - acceptable (IS: {is_metrics['mdd']:.2%}, OOS: {oos_metrics['mdd']:.2%})" + Colors.RESET, "GOOD"))
    
    # Check 4: Win rate consistency (within ±5-10%)
    win_rate_diff = abs(oos_metrics['win_rate'] - is_metrics['win_rate'])
    win_rate_diff_pct = win_rate_diff * 100  # Convert to percentage points
    
    if win_rate_diff_pct > 10:  # More than 10 percentage points difference
        verdicts.append((Colors.RED + f"FAIL: OOS win rate differs by {win_rate_diff_pct:.1f}pp from IS (IS: {is_metrics['win_rate']:.2%}, OOS: {oos_metrics['win_rate']:.2%}) - exploded" + Colors.RESET, "FAIL"))
        overall_status = "FAIL"
    elif win_rate_diff_pct > 5:  # More than 5 percentage points difference
        verdicts.append((Colors.YELLOW + f"WARNING: OOS win rate differs by {win_rate_diff_pct:.1f}pp from IS (IS: {is_metrics['win_rate']:.2%}, OOS: {oos_metrics['win_rate']:.2%})" + Colors.RESET, "WARNING"))
        if overall_status == "GOOD":
            overall_status = "WARNING"
    else:
        verdicts.append((Colors.GREEN + f"✓ OOS win rate within ±5% of IS (IS: {is_metrics['win_rate']:.2%}, OOS: {oos_metrics['win_rate']:.2%}, diff: {win_rate_diff_pct:.1f}pp)" + Colors.RESET, "GOOD"))
    
    # Check 5: P/L variance consistency (check if variance shrinks or expands)
    if is_metrics['pl_variance'] > 0 and oos_metrics['pl_variance'] > 0:
        variance_ratio = oos_metrics['pl_variance'] / is_metrics['pl_variance']
        if variance_ratio > 5.0:  # Variance exploded (5x+)
            verdicts.append((Colors.RED + f"FAIL: OOS P/L variance ratio ({variance_ratio:.2f}x) - exploded - regime bias" + Colors.RESET, "FAIL"))
            overall_status = "FAIL"
        elif variance_ratio < 0.2:  # Variance collapsed (less than 20%)
            verdicts.append((Colors.YELLOW + f"WARNING: OOS P/L variance ratio ({variance_ratio:.2f}x) - collapsed - possible regime bias" + Colors.RESET, "WARNING"))
            if overall_status == "GOOD":
                overall_status = "WARNING"
        elif 0.5 <= variance_ratio <= 2.0:  # Variance is similar (within 2x)
            verdicts.append((Colors.GREEN + f"✓ OOS P/L variance ratio ({variance_ratio:.2f}x) - stable" + Colors.RESET, "GOOD"))
        else:
            verdicts.append((Colors.YELLOW + f"WARNING: OOS P/L variance ratio ({variance_ratio:.2f}x) - changed significantly" + Colors.RESET, "WARNING"))
            if overall_status == "GOOD":
                overall_status = "WARNING"
    
    # Check 6: MAR ratio comparison (for completeness)
    if is_metrics['mar'] > 0 and oos_metrics['mar'] > 0:
        mar_ratio = oos_metrics['mar'] / is_metrics['mar']
        if mar_ratio < 0.1:  # OOS MAR is less than 10% of IS MAR
            verdicts.append((Colors.RED + f"FAIL: OOS MAR ratio ({mar_ratio:.2f}x) - severe overfit (IS: {is_metrics['mar']:.2f}, OOS: {oos_metrics['mar']:.2f})" + Colors.RESET, "FAIL"))
            overall_status = "FAIL"
        elif mar_ratio < 0.4:  # OOS MAR is less than 40% of IS MAR
            verdicts.append((Colors.YELLOW + f"WARNING: OOS MAR ratio ({mar_ratio:.2f}x) - possible overfit (IS: {is_metrics['mar']:.2f}, OOS: {oos_metrics['mar']:.2f})" + Colors.RESET, "WARNING"))
            if overall_status == "GOOD":
                overall_status = "WARNING"
        else:
            verdicts.append((Colors.GREEN + f"✓ OOS MAR ratio ({mar_ratio:.2f}x) - acceptable (IS: {is_metrics['mar']:.2f}, OOS: {oos_metrics['mar']:.2f})" + Colors.RESET, "GOOD"))
    
    return {
        'verdicts': verdicts,
        'overall_status': overall_status,
        'interpretation': interpretation,
        'is_metrics': is_metrics,
        'oos_metrics': oos_metrics,
        'ratios': {
            'cagr_ratio': cagr_ratio if is_metrics['cagr'] != 0 else 0,
            'mdd_ratio': mdd_ratio,
            'mar_ratio': mar_ratio if is_metrics['mar'] > 0 and oos_metrics['mar'] > 0 else 0,
            'variance_ratio': variance_ratio if is_metrics['pl_variance'] > 0 and oos_metrics['pl_variance'] > 0 else 0
        }
    }

def print_verdict(validation_result, oos_name):
    """
    Print color-coded validation verdict.
    """
    if validation_result is None:
        return
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}Validation Verdict: {oos_name}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
    
    # Print comparison metrics
    is_m = validation_result['is_metrics']
    oos_m = validation_result['oos_metrics']
    
    print(f"\n{Colors.BOLD}Metrics Comparison (Ratio-based):{Colors.RESET}")
    print(f"  {'Metric':<20} {'IS':<15} {'OOS':<15} {'Ratio (OOS/IS)':<15} {'Status':<15}")
    print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    
    ratios = validation_result.get('ratios', {})
    cagr_ratio = ratios.get('cagr_ratio', 0)
    cagr_color = Colors.GREEN if 0.5 <= cagr_ratio <= 1.5 else Colors.YELLOW if 0.2 <= cagr_ratio < 0.5 or 1.5 < cagr_ratio <= 3.0 else Colors.RED
    cagr_status = "✓ Good" if 0.5 <= cagr_ratio <= 1.5 else "⚠ Warning" if 0.2 <= cagr_ratio < 0.5 or 1.5 < cagr_ratio <= 3.0 else "✗ Fail"
    print(f"  {'CAGR':<20} {is_m['cagr']:>14.2%} {oos_m['cagr']:>14.2%} {cagr_color}{cagr_ratio:>14.2f}x{cagr_color} {cagr_status}{Colors.RESET}")
    
    mdd_ratio = ratios.get('mdd_ratio', 0)
    mdd_color = Colors.GREEN if mdd_ratio <= 1.0 else Colors.YELLOW if mdd_ratio <= 1.5 else Colors.RED
    mdd_status = "✓ Good" if mdd_ratio <= 1.0 else "⚠ Warning" if mdd_ratio <= 1.5 else "✗ Fail"
    print(f"  {'Max Drawdown':<20} {is_m['mdd']:>14.2%} {oos_m['mdd']:>14.2%} {mdd_color}{mdd_ratio:>14.2f}x{mdd_color} {mdd_status}{Colors.RESET}")
    
    mar_ratio = ratios.get('mar_ratio', 0)
    mar_color = Colors.GREEN if mar_ratio >= 0.4 else Colors.YELLOW if mar_ratio >= 0.1 else Colors.RED
    mar_status = "✓ Good" if mar_ratio >= 0.4 else "⚠ Warning" if mar_ratio >= 0.1 else "✗ Fail"
    print(f"  {'MAR Ratio':<20} {is_m['mar']:>14.2f} {oos_m['mar']:>14.2f} {mar_color}{mar_ratio:>14.2f}x{mar_color} {mar_status}{Colors.RESET}")
    
    wr_diff_pct = abs(oos_m['win_rate'] - is_m['win_rate']) * 100
    wr_color = Colors.GREEN if wr_diff_pct <= 5 else Colors.YELLOW if wr_diff_pct <= 10 else Colors.RED
    wr_status = "✓ Good" if wr_diff_pct <= 5 else "⚠ Warning" if wr_diff_pct <= 10 else "✗ Fail"
    print(f"  {'Win Rate':<20} {is_m['win_rate']:>14.2%} {oos_m['win_rate']:>14.2%} {wr_color}{wr_diff_pct:>13.1f}pp{wr_color} {wr_status}{Colors.RESET}")
    
    variance_ratio = ratios.get('variance_ratio', 0)
    if variance_ratio > 0:
        var_color = Colors.GREEN if 0.5 <= variance_ratio <= 2.0 else Colors.YELLOW if 0.2 <= variance_ratio < 0.5 or 2.0 < variance_ratio <= 5.0 else Colors.RED
        var_status = "✓ Stable" if 0.5 <= variance_ratio <= 2.0 else "⚠ Changed" if 0.2 <= variance_ratio < 0.5 or 2.0 < variance_ratio <= 5.0 else "✗ Exploded"
        print(f"  {'P/L Variance':<20} {is_m['pl_variance']:>14.2f} {oos_m['pl_variance']:>14.2f} {var_color}{variance_ratio:>14.2f}x{var_color} {var_status}{Colors.RESET}")
    
    print(f"\n{Colors.BOLD}Validation Checks:{Colors.RESET}")
    for verdict, status in validation_result['verdicts']:
        print(f"  {verdict}")
    
    # Overall verdict with interpretation
    print(f"\n{Colors.BOLD}Overall Verdict:{Colors.RESET}")
    interpretation = validation_result.get('interpretation', 'UNKNOWN')
    
    if validation_result['overall_status'] == "GOOD":
        print(f"  {Colors.GREEN}{Colors.BOLD}✓ PASS - Strategy looks good for {oos_name}{Colors.RESET}")
        if interpretation != "UNKNOWN":
            print(f"  {Colors.GREEN}{interpretation}{Colors.RESET}")
    elif validation_result['overall_status'] == "WARNING":
        print(f"  {Colors.YELLOW}{Colors.BOLD}⚠ WARNING - Strategy shows some concerns for {oos_name}{Colors.RESET}")
        if interpretation != "UNKNOWN":
            print(f"  {Colors.YELLOW}{interpretation}{Colors.RESET}")
    else:
        print(f"  {Colors.RED}{Colors.BOLD}✗ FAIL - Strategy fails validation for {oos_name}{Colors.RESET}")
        if interpretation != "UNKNOWN":
            print(f"  {Colors.RED}{interpretation}{Colors.RESET}")
        else:
            print(f"  {Colors.RED}Consider: Strategy may be overfit or edge has collapsed{Colors.RESET}")

def calculate_full_equity_curve(df, starting_equity=RISK_CAPITAL):
    """
    Calculate full equity curve with dates for the entire backtest.
    Returns a DataFrame with Date and Equity columns.
    """
    if df.empty:
        return pd.DataFrame(columns=['Date', 'Equity', 'Period'])
    
    # Calculate equity curve with proper allocation logic
    if ALLOCATION_TYPE == "percent" and "adjusted_pl" in df.columns:
        eq_values = [starting_equity]
        dates = [df['Date Opened'].iloc[0]]
        periods = ['IS']
        
        # Determine split points
        is_end = int(len(df) * IS_FRAC)
        val_end = int(len(df) * (IS_FRAC + VAL_FRAC))
        
        for i in range(len(df)):
            current_eq = eq_values[-1]
            row = df.iloc[i:i+1].copy()
            row = calculate_contracts_and_pl(row, current_eq, ALLOCATION_TYPE, ALLOCATION_VALUE)
            row["contracts_to_trade"] = row["contracts_to_trade"].astype(int)
            
            adjusted_pl = row["adjusted_pl"].iloc[0]
            ret = adjusted_pl / current_eq if current_eq > 0 else 0
            ret = max(ret, -1.0)
            
            new_eq = current_eq * (1 + ret)
            eq_values.append(max(new_eq, 0.01))
            dates.append(df['Date Opened'].iloc[i])
            
            # Determine period
            if i < is_end:
                periods.append('IS')
            elif i < val_end:
                periods.append('Validation OOS')
            else:
                periods.append('Final OOS')
        
        return pd.DataFrame({
            'Date': dates,
            'Equity': eq_values,
            'Period': periods
        })
    else:
        # Use pre-calculated returns
        if "trade_ret" in df.columns:
            eq = equity_curve(df["trade_ret"].values, starting_equity)
        else:
            eq = equity_curve([0] * len(df), starting_equity)
        
        dates = [df['Date Opened'].iloc[0]] + df['Date Opened'].tolist()
        is_end = int(len(df) * IS_FRAC)
        val_end = int(len(df) * (IS_FRAC + VAL_FRAC))
        periods = ['IS'] + ['IS' if i < is_end else 'Validation OOS' if i < val_end else 'Final OOS' 
                            for i in range(len(df))]
        
        return pd.DataFrame({
            'Date': dates,
            'Equity': eq,
            'Period': periods
        })

def calculate_mc_equity_curves(oos_df, mc_data):
    """
    Calculate equity curves for best, median, and worst MC paths.
    Returns a dict with 'best', 'median', 'worst' equity curves.
    Note: This recalculates paths, so results may differ slightly from original MC.
    For exact reproduction, we'd need to store all paths, which is memory-intensive.
    """
    if mc_data is None or oos_df.empty:
        return None
    
    final_arr = mc_data['final_arr']
    
    # Find indices for best, median, worst
    best_idx = np.argmax(final_arr)
    worst_idx = np.argmin(final_arr)
    sorted_indices = np.argsort(final_arr)
    median_idx = sorted_indices[len(sorted_indices) // 2]
    
    # Recalculate equity curves for these specific paths
    # We need to recreate the exact same random sequence used in monte_carlo_on_oos
    n = len(oos_df)
    n_paths = mc_data.get('n_paths', 100)
    
    curves = {}
    rng = np.random.default_rng(seed=42)
    
    # Generate all paths and store the ones we need
    # This matches the exact sequence from monte_carlo_on_oos
    for path_idx in range(n_paths):
        sample_indices = rng.choice(len(oos_df), size=n, replace=True)
        
        if path_idx in [best_idx, median_idx, worst_idx]:
            sample_df = oos_df.iloc[sample_indices].copy().reset_index(drop=True)
            
            if ALLOCATION_TYPE == "percent" and "adjusted_pl" in sample_df.columns:
                eq_values = [RISK_CAPITAL]
                for i in range(len(sample_df)):
                    current_eq = eq_values[-1]
                    row = sample_df.iloc[i:i+1].copy()
                    row = calculate_contracts_and_pl(row, current_eq, ALLOCATION_TYPE, ALLOCATION_VALUE)
                    row["contracts_to_trade"] = row["contracts_to_trade"].astype(int)
                    
                    adjusted_pl = row["adjusted_pl"].iloc[0]
                    ret = adjusted_pl / current_eq if current_eq > 0 else 0
                    ret = max(ret, -1.0)
                    
                    new_eq = current_eq * (1 + ret)
                    eq_values.append(max(new_eq, 0.01))
                
                if path_idx == best_idx:
                    curves['best'] = eq_values
                elif path_idx == median_idx:
                    curves['median'] = eq_values
                elif path_idx == worst_idx:
                    curves['worst'] = eq_values
            else:
                sample_rets = sample_df["trade_ret"].values
                eq = equity_curve(sample_rets, RISK_CAPITAL)
                if path_idx == best_idx:
                    curves['best'] = eq.tolist()
                elif path_idx == median_idx:
                    curves['median'] = eq.tolist()
                elif path_idx == worst_idx:
                    curves['worst'] = eq.tolist()
        
        # Stop once we have all three
        if len(curves) == 3:
            break
    
    return curves

def validate_mc_performance(mc_data, oos_metrics):
    """
    Validate Monte Carlo results against OOS metrics using industry-grade checks.
    Returns a verdict dictionary with color-coded messages.
    """
    if mc_data is None or oos_metrics is None:
        return None
    
    verdicts = []
    overall_status = "GOOD"  # GOOD, WARNING, FAIL
    
    mdd_arr = mc_data['mdd_arr']
    final_arr = mc_data['final_arr']
    win_rates_arr = mc_data['win_rates_arr']
    oos_mdd = oos_metrics['mdd']
    
    # Check 1: Expected Max Drawdown vs. Observed (OOS) Drawdown
    # This is the single most important check
    # For drawdowns (negative values), more negative = worse
    # Percentiles: 5th = worst 5% of paths, 50th = median, 95th = best 5% of paths
    mdd_5 = np.percentile(mdd_arr, 5)   # Worst 5% of MC paths
    mdd_50 = mc_data['mdd_50']          # Median (50th percentile)
    mdd_80 = np.percentile(mdd_arr, 80) # 80th percentile (worse than 80% of paths)
    mdd_95 = np.percentile(mdd_arr, 95) # Best 5% of MC paths (95th percentile)
    
    # Compare OOS MDD to MC percentiles
    # If OOS MDD is worse (more negative) than worst MC paths, fail
    if oos_mdd < mdd_5:  # OOS is worse than worst 5% of MC paths
        verdicts.append((Colors.RED + f"FAIL: OOS MDD ({oos_mdd:.2%}) worse than 5th percentile (worst) MC MDD ({mdd_5:.2%}) - strategy is too fragile or overfit" + Colors.RESET, "FAIL"))
        overall_status = "FAIL"
    elif oos_mdd < mdd_50:  # OOS is worse than median
        verdicts.append((Colors.YELLOW + f"WARNING: OOS MDD ({oos_mdd:.2%}) worse than median MC MDD ({mdd_50:.2%}) - risky but possible" + Colors.RESET, "WARNING"))
        if overall_status == "GOOD":
            overall_status = "WARNING"
    elif oos_mdd < mdd_80:  # OOS is worse than 80th percentile
        verdicts.append((Colors.GREEN + f"✓ OOS MDD ({oos_mdd:.2%}) between median and 80th percentile MC MDD ({mdd_50:.2%}-{mdd_80:.2%}) - normal" + Colors.RESET, "GOOD"))
    else:  # OOS is better (less negative) than 80th percentile
        verdicts.append((Colors.GREEN + f"✓ OOS MDD ({oos_mdd:.2%}) better than 80th percentile MC MDD ({mdd_80:.2%}) - excellent robustness" + Colors.RESET, "GOOD"))
    
    # Check 2: Probability of Positive Return (PoPR)
    popr = (final_arr > RISK_CAPITAL).sum() / len(final_arr) * 100  # Percentage of paths with positive return
    
    if popr < 50:
        verdicts.append((Colors.RED + f"FAIL: PoPR ({popr:.1f}%) < 50% - strategy is not statistically reliable" + Colors.RESET, "FAIL"))
        overall_status = "FAIL"
    elif popr < 65:
        verdicts.append((Colors.YELLOW + f"WARNING: PoPR ({popr:.1f}%) between 50-65% - borderline, needs strong economic reasoning" + Colors.RESET, "WARNING"))
        if overall_status == "GOOD":
            overall_status = "WARNING"
    elif popr < 80:
        verdicts.append((Colors.GREEN + f"✓ PoPR ({popr:.1f}%) between 65-80% - good" + Colors.RESET, "GOOD"))
    else:
        verdicts.append((Colors.GREEN + f"✓ PoPR ({popr:.1f}%) > 80% - very strong consistency" + Colors.RESET, "GOOD"))
    
    # Check 3: Left-Tail Severity (5th and 1st percentiles)
    ret_5 = mc_data['ret_5']
    ret_1 = mc_data['ret_1']
    
    # 5th percentile check
    if ret_5 > -0.10:
        verdicts.append((Colors.GREEN + f"✓ 5th percentile return ({ret_5:.2%}) > -10% - good" + Colors.RESET, "GOOD"))
    elif ret_5 > -0.25:
        verdicts.append((Colors.YELLOW + f"WARNING: 5th percentile return ({ret_5:.2%}) between -10% to -25% - borderline" + Colors.RESET, "WARNING"))
        if overall_status == "GOOD":
            overall_status = "WARNING"
    else:
        verdicts.append((Colors.RED + f"FAIL: 5th percentile return ({ret_5:.2%}) < -25% - reject" + Colors.RESET, "FAIL"))
        overall_status = "FAIL"
    
    # 1st percentile check
    if ret_1 > -0.20:
        verdicts.append((Colors.GREEN + f"✓ 1st percentile return ({ret_1:.2%}) > -20% - good" + Colors.RESET, "GOOD"))
    elif ret_1 > -0.40:
        verdicts.append((Colors.YELLOW + f"WARNING: 1st percentile return ({ret_1:.2%}) between -20% to -40% - borderline" + Colors.RESET, "WARNING"))
        if overall_status == "GOOD":
            overall_status = "WARNING"
    else:
        verdicts.append((Colors.RED + f"FAIL: 1st percentile return ({ret_1:.2%}) < -40% - not safe for production" + Colors.RESET, "FAIL"))
        overall_status = "FAIL"
    
    # Check 4: Variance of Pathwise Outcomes (Dispersion)
    ret_std = np.std(final_arr / RISK_CAPITAL - 1)  # Standard deviation of total returns
    ret_median = mc_data['ret_median']
    ret_25 = mc_data['ret_25']
    ret_75 = mc_data['ret_75']
    
    # Interquartile range
    iqr = ret_75 - ret_25
    # Dispersion ratio: (75th percentile - 25th percentile) / median
    # Use absolute value of median to handle negative medians
    if abs(ret_median) > 0.001:  # Avoid division by very small numbers
        dispersion_ratio = abs(iqr / ret_median)
    else:
        # If median is very small, use IQR as absolute measure
        # Normalize by a small value to get meaningful ratio
        dispersion_ratio = abs(iqr) / 0.01 if abs(iqr) > 0 else 0
    
    if dispersion_ratio < 0.5:
        verdicts.append((Colors.GREEN + f"✓ Dispersion ratio ({dispersion_ratio:.2f}) < 0.5 - very stable" + Colors.RESET, "GOOD"))
    elif dispersion_ratio < 1.0:
        verdicts.append((Colors.GREEN + f"✓ Dispersion ratio ({dispersion_ratio:.2f}) between 0.5-1.0 - normal" + Colors.RESET, "GOOD"))
    elif dispersion_ratio < 2.0:
        verdicts.append((Colors.YELLOW + f"WARNING: Dispersion ratio ({dispersion_ratio:.2f}) between 1.0-2.0 - high uncertainty" + Colors.RESET, "WARNING"))
        if overall_status == "GOOD":
            overall_status = "WARNING"
    else:
        verdicts.append((Colors.RED + f"FAIL: Dispersion ratio ({dispersion_ratio:.2f}) > 2.0 - too unstable for trading" + Colors.RESET, "FAIL"))
        overall_status = "FAIL"
    
    # Check 5: Drawdown–Return Efficiency (MC MAR-like Test)
    # MC_Efficiency = Median_Return / Median_MaxDD
    mdd_median = mc_data['mdd_median']
    if abs(mdd_median) > 0.001:  # Avoid division by zero
        mc_efficiency = abs(ret_median / mdd_median)
    else:
        mc_efficiency = float('inf') if ret_median > 0 else 0
    
    if mc_efficiency > 0.5:
        verdicts.append((Colors.GREEN + f"✓ MC Efficiency ({mc_efficiency:.2f}) > 0.5 - strong (robust strategy)" + Colors.RESET, "GOOD"))
    elif mc_efficiency > 0.25:
        verdicts.append((Colors.GREEN + f"✓ MC Efficiency ({mc_efficiency:.2f}) between 0.25-0.5 - acceptable" + Colors.RESET, "GOOD"))
    elif mc_efficiency > 0.1:
        verdicts.append((Colors.YELLOW + f"WARNING: MC Efficiency ({mc_efficiency:.2f}) between 0.1-0.25 - weak" + Colors.RESET, "WARNING"))
        if overall_status == "GOOD":
            overall_status = "WARNING"
    else:
        verdicts.append((Colors.RED + f"FAIL: MC Efficiency ({mc_efficiency:.2f}) < 0.1 - reject" + Colors.RESET, "FAIL"))
        overall_status = "FAIL"
    
    # Check 6: Catastrophic Failure Probability
    catastrophic_mdd = (mdd_arr < -0.50).sum() / len(mdd_arr) * 100  # % of paths with MDD > 50%
    catastrophic_return = (final_arr < RISK_CAPITAL * 0.70).sum() / len(final_arr) * 100  # % of paths with return < -30%
    catastrophic_total = max(catastrophic_mdd, catastrophic_return)
    
    if catastrophic_total < 1.0:
        verdicts.append((Colors.GREEN + f"✓ Catastrophic failure probability ({catastrophic_total:.1f}%) < 1% - excellent" + Colors.RESET, "GOOD"))
    elif catastrophic_total < 5.0:
        verdicts.append((Colors.GREEN + f"✓ Catastrophic failure probability ({catastrophic_total:.1f}%) between 1-5% - acceptable with risk control" + Colors.RESET, "GOOD"))
    elif catastrophic_total < 10.0:
        verdicts.append((Colors.YELLOW + f"WARNING: Catastrophic failure probability ({catastrophic_total:.1f}%) between 5-10% - high risk" + Colors.RESET, "WARNING"))
        if overall_status == "GOOD":
            overall_status = "WARNING"
    else:
        verdicts.append((Colors.RED + f"FAIL: Catastrophic failure probability ({catastrophic_total:.1f}%) > 10% - reject" + Colors.RESET, "FAIL"))
        overall_status = "FAIL"
    
    # Check 7: Stability of Trade/Return Sequence
    original_win_rate = mc_data['original_win_rate']
    bad_win_rate_paths = (win_rates_arr < (original_win_rate - 0.10)).sum() / len(win_rates_arr) * 100
    
    if bad_win_rate_paths < 5.0:
        verdicts.append((Colors.GREEN + f"✓ Bad win rate paths ({bad_win_rate_paths:.1f}%) < 5% - stable" + Colors.RESET, "GOOD"))
    elif bad_win_rate_paths < 15.0:
        verdicts.append((Colors.YELLOW + f"WARNING: Bad win rate paths ({bad_win_rate_paths:.1f}%) between 5-15% - moderate" + Colors.RESET, "WARNING"))
        if overall_status == "GOOD":
            overall_status = "WARNING"
    else:
        verdicts.append((Colors.RED + f"FAIL: Bad win rate paths ({bad_win_rate_paths:.1f}%) > 15% - fragile" + Colors.RESET, "FAIL"))
        overall_status = "FAIL"
    
    # Check 8: "Noise-to-Signal Ratio" (NSR)
    # NSR = StdDev(final equity paths) / Median(final equity)
    final_std = np.std(final_arr)
    final_median = np.median(final_arr)
    if final_median > 0:
        nsr = final_std / final_median
    else:
        nsr = float('inf')
    
    if nsr < 0.5:
        verdicts.append((Colors.GREEN + f"✓ NSR ({nsr:.4f}) < 0.5 - excellent" + Colors.RESET, "GOOD"))
    elif nsr < 1.0:
        verdicts.append((Colors.GREEN + f"✓ NSR ({nsr:.4f}) between 0.5-1.0 - normal" + Colors.RESET, "GOOD"))
    elif nsr < 1.5:
        verdicts.append((Colors.YELLOW + f"WARNING: NSR ({nsr:.4f}) between 1.0-1.5 - high uncertainty" + Colors.RESET, "WARNING"))
        if overall_status == "GOOD":
            overall_status = "WARNING"
    else:
        verdicts.append((Colors.RED + f"FAIL: NSR ({nsr:.4f}) > 1.5 - reject" + Colors.RESET, "FAIL"))
        overall_status = "FAIL"
    
    return {
        'verdicts': verdicts,
        'overall_status': overall_status,
        'mc_data': mc_data,
        'oos_metrics': oos_metrics,
        'metrics': {
            'popr': popr,
            'ret_5': ret_5,
            'ret_1': ret_1,
            'dispersion_ratio': dispersion_ratio,
            'mc_efficiency': mc_efficiency,
            'catastrophic_total': catastrophic_total,
            'bad_win_rate_paths': bad_win_rate_paths,
            'nsr': nsr,
            'oos_mdd_percentile': _get_percentile(oos_mdd, mdd_arr)
        }
    }

def _get_percentile(value, arr):
    """
    Calculate what percentile a value falls into for an array.
    For drawdowns (negative values), higher percentile = worse (more negative).
    Returns the percentage of values that are better (less negative) than the given value.
    """
    if len(arr) == 0:
        return 0.0
    # For drawdowns, more negative is worse
    # Count how many MC values are better (less negative, i.e., greater) than the OOS value
    # This gives us the percentile where worse values fall
    better_count = (arr > value).sum()  # Values that are less negative (better)
    percentile = better_count / len(arr) * 100
    return percentile

def print_mc_verdict(mc_verdict):
    """
    Print color-coded Monte Carlo validation verdict.
    """
    if mc_verdict is None:
        return
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}Monte Carlo Validation Verdict{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
    
    mc_data = mc_verdict['mc_data']
    oos_metrics = mc_verdict['oos_metrics']
    metrics = mc_verdict['metrics']
    
    print(f"\n{Colors.BOLD}Key Metrics:{Colors.RESET}")
    print(f"  Monte Carlo Simulations: {mc_data.get('n_paths', 0):,}")
    print(f"  Trades per Path: {mc_data.get('n_trades', 0)}")
    print(f"  OOS Max Drawdown: {oos_metrics['mdd']:.2%}")
    print(f"  OOS MDD Percentile in MC: {metrics['oos_mdd_percentile']:.1f}th percentile")
    print(f"  Probability of Positive Return (PoPR): {metrics['popr']:.1f}%")
    print(f"  5th percentile return: {metrics['ret_5']:.2%}")
    print(f"  1st percentile return: {metrics['ret_1']:.2%}")
    print(f"  Dispersion ratio: {metrics['dispersion_ratio']:.2f}")
    print(f"  MC Efficiency: {metrics['mc_efficiency']:.2f}")
    print(f"  Catastrophic failure probability: {metrics['catastrophic_total']:.1f}%")
    print(f"  Bad win rate paths: {metrics['bad_win_rate_paths']:.1f}%")
    print(f"  Noise-to-Signal Ratio (NSR): {metrics['nsr']:.4f}")
    
    print(f"\n{Colors.BOLD}Validation Checks:{Colors.RESET}")
    for verdict, status in mc_verdict['verdicts']:
        print(f"  {verdict}")
    
    # Overall verdict
    print(f"\n{Colors.BOLD}Overall Verdict:{Colors.RESET}")
    if mc_verdict['overall_status'] == "GOOD":
        print(f"  {Colors.GREEN}{Colors.BOLD}✓ PASS - Monte Carlo validation passed{Colors.RESET}")
    elif mc_verdict['overall_status'] == "WARNING":
        print(f"  {Colors.YELLOW}{Colors.BOLD}⚠ WARNING - Monte Carlo shows some concerns{Colors.RESET}")
    else:
        print(f"  {Colors.RED}{Colors.BOLD}✗ FAIL - Monte Carlo validation failed{Colors.RESET}")
        print(f"  {Colors.RED}Strategy may be too fragile, overfit, or have high catastrophic risk{Colors.RESET}")

def generate_html_report(df, is_df, val_df, test_df, is_metrics, val_metrics, test_metrics, 
                         val_verdict, test_verdict, mc_verdict, is_eq, val_eq, test_eq,
                         csv_path, full_eq_curve=None, mc_equity_curves=None):
    """
    Generate a color-coded HTML report with all validation results.
    """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IS/OOS Validation Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #ecf0f1;
        }}
        h3 {{
            color: #555;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .summary-box {{
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th {{
            background: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .status-good {{
            color: #27ae60;
            font-weight: bold;
        }}
        .status-warning {{
            color: #f39c12;
            font-weight: bold;
        }}
        .status-fail {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .verdict-box {{
            padding: 20px;
            border-radius: 6px;
            margin: 20px 0;
        }}
        .verdict-good {{
            background: #d4edda;
            border-left: 4px solid #27ae60;
        }}
        .verdict-warning {{
            background: #fff3cd;
            border-left: 4px solid #f39c12;
        }}
        .verdict-fail {{
            background: #f8d7da;
            border-left: 4px solid #e74c3c;
        }}
        .check-item {{
            padding: 8px 0;
            border-bottom: 1px solid #ecf0f1;
        }}
        .check-item:last-child {{
            border-bottom: none;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 30px;
            text-align: right;
        }}
        .section {{
            margin: 30px 0;
        }}
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 20px 0;
            background: white;
            padding: 15px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body>
    <div class="container">
        <h1>📊 IS/OOS Validation Report</h1>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        
        <div class="section">
            <h2>📈 Overall Summary</h2>
"""
    
    # Add equity curve visualization
    if full_eq_curve is not None and not full_eq_curve.empty:
        # Prepare data for chart - use simple index-based x-axis
        dates = full_eq_curve['Date'].dt.strftime('%Y-%m-%d').tolist()
        equity = full_eq_curve['Equity'].tolist()
        periods = full_eq_curve['Period'].tolist()
        
        # Calculate boundary indices based on IS_FRAC and VAL_FRAC
        # The equity curve has n+1 points (initial equity + one per trade)
        # So boundaries are at: initial point (0) + trade indices
        # IS ends after IS_FRAC of trades, Validation OOS ends after (IS_FRAC + VAL_FRAC) of trades
        num_trades = len(df)
        is_end_trade_idx = int(num_trades * IS_FRAC)  # Last trade index in IS
        val_end_trade_idx = int(num_trades * (IS_FRAC + VAL_FRAC))  # Last trade index in Validation OOS
        
        # In equity curve: index 0 is initial equity, index 1+ corresponds to after each trade
        # So IS/Validation boundary is after trade is_end_trade_idx, which is at equity index is_end_trade_idx + 1
        # But we want to mark at the point where Validation OOS starts, which is after the IS period ends
        is_end_idx = is_end_trade_idx + 1  # Point after last IS trade
        val_end_idx = val_end_trade_idx + 1  # Point after last Validation OOS trade
        
        # Get boundary dates for labels
        is_boundary_date = dates[is_end_idx] if is_end_idx < len(dates) else None
        val_boundary_date = dates[val_end_idx] if val_end_idx < len(dates) else None
        
        # Prepare P/L per contract data for histogram
        pl_per_contract_data = df['pl_per_contract'].dropna().tolist() if 'pl_per_contract' in df.columns else []
        
        html += f"""
        <div class="section">
            <h2>📊 Cumulative Equity Curve</h2>
            <div class="chart-container">
                <canvas id="equityCurveChart"></canvas>
            </div>
            <script>
                const ctx1 = document.getElementById('equityCurveChart').getContext('2d');
                const dates = {dates};
                const equity = {equity};
                const isEndIdx = {is_end_idx};
                const valEndIdx = {val_end_idx};
                const isBoundaryDate = {f"'{is_boundary_date}'" if is_boundary_date else 'null'};
                const valBoundaryDate = {f"'{val_boundary_date}'" if val_boundary_date else 'null'};
                
                // Find min and max equity values for vertical line range
                const minEquity = Math.min(...equity);
                const maxEquity = Math.max(...equity);
                const equityRange = maxEquity - minEquity;
                const lineTop = maxEquity + equityRange * 0.05;
                const lineBottom = minEquity - equityRange * 0.05;
                
                // Create single continuous line with all equity points
                const equityDataPoints = equity.map((val, idx) => ({{x: idx, y: val}}));
                
                // Single dataset for the equity curve
                const datasets = [{{
                    label: 'Equity',
                    data: equityDataPoints,
                    borderColor: 'rgb(52, 152, 219)',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    fill: false,
                    tension: 0,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    spanGaps: false
                }}];
                
                // Add vertical lines at boundaries using line datasets
                // These will appear as vertical lines since x is constant and y varies
                if (isEndIdx !== null && isEndIdx >= 0 && isEndIdx < equity.length) {{
                    datasets.push({{
                        label: 'IS / Validation OOS' + (isBoundaryDate ? ' (' + isBoundaryDate + ')' : ''),
                        data: [
                            {{x: isEndIdx, y: lineBottom}},
                            {{x: isEndIdx, y: lineTop}}
                        ],
                        borderColor: 'rgb(241, 196, 15)',
                        backgroundColor: 'transparent',
                        borderWidth: 3,
                        borderDash: [8, 4],
                        fill: false,
                        tension: 0,
                        pointRadius: 0,
                        pointHoverRadius: 0,
                        showLine: true
                    }});
                }}
                if (valEndIdx !== null && valEndIdx >= 0 && valEndIdx < equity.length) {{
                    datasets.push({{
                        label: 'Validation OOS / Final OOS' + (valBoundaryDate ? ' (' + valBoundaryDate + ')' : ''),
                        data: [
                            {{x: valEndIdx, y: lineBottom}},
                            {{x: valEndIdx, y: lineTop}}
                        ],
                        borderColor: 'rgb(231, 76, 60)',
                        backgroundColor: 'transparent',
                        borderWidth: 3,
                        borderDash: [8, 4],
                        fill: false,
                        tension: 0,
                        pointRadius: 0,
                        pointHoverRadius: 0,
                        showLine: true
                    }});
                }}
                
                const equityData = {{
                    labels: dates,
                    datasets: datasets
                }};
                new Chart(ctx1, {{
                    type: 'line',
                    data: equityData,
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {{
                            intersect: false,
                            mode: 'index'
                        }},
                        plugins: {{
                            title: {{
                                display: true,
                                text: 'Cumulative Equity Curve'
                            }},
                            legend: {{
                                display: true,
                                position: 'top'
                            }},
                            tooltip: {{
                                mode: 'index',
                                intersect: false,
                                callbacks: {{
                                    title: function(context) {{
                                        return dates[context[0].dataIndex];
                                    }},
                                    label: function(context) {{
                                        if (context.datasetIndex === 0) {{
                                            return 'Equity: $' + context.parsed.y.toLocaleString('en-US', {{maximumFractionDigits: 2}});
                                        }}
                                        return context.dataset.label;
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                type: 'linear',
                                title: {{
                                    display: true,
                                    text: 'Trade Number'
                                }},
                                position: 'bottom'
                            }},
                            y: {{
                                title: {{
                                    display: true,
                                    text: 'Equity ($)'
                                }},
                                ticks: {{
                                    callback: function(value) {{
                                        return '$' + value.toLocaleString('en-US');
                                    }}
                                }}
                            }}
                        }}
                    }}
                }});
            </script>
        </div>
"""
        
        # Add P/L per contract histogram
        if pl_per_contract_data:
            html += f"""
        <div class="section">
            <h2>📊 P/L Per Contract Distribution</h2>
            <div class="chart-container">
                <canvas id="plHistogramChart"></canvas>
            </div>
            <script>
                const ctx3 = document.getElementById('plHistogramChart').getContext('2d');
                const plData = {pl_per_contract_data};
                
                // Calculate histogram bins with $100 jumps
                const min = Math.min(...plData);
                const max = Math.max(...plData);
                const binWidth = 500;  // $500 jumps
                const binStart = Math.floor(min / binWidth) * binWidth;  // Round down to nearest $100
                const binEnd = Math.ceil(max / binWidth) * binWidth;     // Round up to nearest $100
                const binCount = Math.ceil((binEnd - binStart) / binWidth);
                const bins = Array(binCount).fill(0);
                const binLabels = [];
                
                // Create bin labels
                for (let i = 0; i < binCount; i++) {{
                    const binLower = binStart + i * binWidth;
                    const binUpper = binLower + binWidth;
                    binLabels.push(`$${{binLower}} - ${{binUpper}}`);
                }}
                
                // Count values in each bin
                plData.forEach(value => {{
                    const binIndex = Math.min(Math.floor((value - binStart) / binWidth), binCount - 1);
                    if (binIndex >= 0 && binIndex < binCount) {{
                        bins[binIndex]++;
                    }}
                }});
                
                const histogramData = {{
                    labels: binLabels,
                    datasets: [{{
                        label: 'Frequency',
                        data: bins,
                        backgroundColor: 'rgba(52, 152, 219, 0.6)',
                        borderColor: 'rgb(52, 152, 219)',
                        borderWidth: 1
                    }}]
                }};
                
                new Chart(ctx3, {{
                    type: 'bar',
                    data: histogramData,
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            title: {{
                                display: true,
                                text: 'P/L Per Contract Distribution'
                            }},
                            legend: {{
                                display: false
                            }},
                            tooltip: {{
                                callbacks: {{
                                    title: function(context) {{
                                        const binIndex = context[0].dataIndex;
                                        return binLabels[binIndex];
                                    }},
                                    label: function(context) {{
                                        return 'Count: ' + context.parsed.y;
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                title: {{
                                    display: true,
                                    text: 'P/L Per Contract ($)'
                                }},
                                ticks: {{
                                    maxRotation: 45,
                                    minRotation: 45
                                }}
                            }},
                            y: {{
                                title: {{
                                    display: true,
                                    text: 'Frequency'
                                }},
                                beginAtZero: true
                            }}
                        }}
                    }}
                }});
            </script>
        </div>
"""
    
    html += f"""
        <div class="section">
            <h2>📈 Overall Summary</h2>
            <div class="summary-box">
                <p><strong>Data Source:</strong> {csv_path}</p>
                <p><strong>Total Trades:</strong> {len(df)}</p>
                <p><strong>Period:</strong> {df['Date Opened'].min().date()} → {df['Date Opened'].max().date()}</p>
                <p><strong>Allocation Strategy:</strong> {ALLOCATION_TYPE}</p>
                <p><strong>Allocation Value:</strong> {ALLOCATION_VALUE*100 if ALLOCATION_TYPE == 'percent' else ALLOCATION_VALUE}{'% of equity' if ALLOCATION_TYPE == 'percent' else ' contracts'}</p>
                <p><strong>Initial Equity:</strong> ${RISK_CAPITAL:,.2f}</p>
            </div>
        </div>
"""
    
    # Add period metrics
    for name, metrics, eq in [("In-Sample (Train)", is_metrics, is_eq), 
                               ("Validation OOS (Dev)", val_metrics, val_eq),
                               ("Final OOS (Test)", test_metrics, test_eq)]:
        if metrics:
            html += f"""
        <div class="section">
            <h2>{name}</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">CAGR</div>
                    <div class="metric-value">{metrics['cagr']:.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value">{metrics['mdd']:.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value">{metrics['win_rate']:.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value">{metrics['total_return']:.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">MAR Ratio</div>
                    <div class="metric-value">{metrics['mar']:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Starting Equity</div>
                    <div class="metric-value">${metrics['starting_equity']:,.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Ending Equity</div>
                    <div class="metric-value">${metrics['ending_equity']:,.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Trades</div>
                    <div class="metric-value">{metrics['total_trades']}</div>
                </div>
            </div>
        </div>
"""
    
    # Add validation verdicts
    for verdict, name in [(val_verdict, "Validation OOS (Dev)"), (test_verdict, "Final OOS (Test)")]:
        if verdict:
            status_class = "good" if verdict['overall_status'] == "GOOD" else "warning" if verdict['overall_status'] == "WARNING" else "fail"
            html += f"""
        <div class="section">
            <h2>🔍 {name} Validation</h2>
            <div class="verdict-box verdict-{status_class}">
                <h3>Overall Verdict: <span class="status-{status_class}">{verdict['overall_status']}</span></h3>
                <p>{verdict.get('interpretation', '')}</p>
            </div>
            
            <h3>Metrics Comparison</h3>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>IS</th>
                        <th>OOS</th>
                        <th>Ratio (OOS/IS)</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
"""
            ratios = verdict.get('ratios', {})
            is_m = is_metrics if is_metrics else {}
            oos_m = val_metrics if "Validation" in name else test_metrics
            
            # CAGR
            cagr_ratio = ratios.get('cagr_ratio', 0)
            cagr_status = "good" if 0.5 <= cagr_ratio <= 1.5 else "warning" if 0.2 <= cagr_ratio <= 2.0 else "fail"
            html += f"""
                    <tr>
                        <td>CAGR</td>
                        <td>{is_m.get('cagr', 0):.2%}</td>
                        <td>{oos_m.get('cagr', 0):.2%}</td>
                        <td>{cagr_ratio:.2f}x</td>
                        <td><span class="status-{cagr_status}">{'✓ Good' if cagr_status == 'good' else '⚠ Warning' if cagr_status == 'warning' else '✗ Fail'}</span></td>
                    </tr>
"""
            
            # Max Drawdown
            mdd_ratio = ratios.get('mdd_ratio', 0)
            mdd_status = "good" if 0.5 <= mdd_ratio <= 1.5 else "warning" if 0.2 <= mdd_ratio <= 2.0 else "fail"
            html += f"""
                    <tr>
                        <td>Max Drawdown</td>
                        <td>{is_m.get('mdd', 0):.2%}</td>
                        <td>{oos_m.get('mdd', 0):.2%}</td>
                        <td>{mdd_ratio:.2f}x</td>
                        <td><span class="status-{mdd_status}">{'✓ Good' if mdd_status == 'good' else '⚠ Warning' if mdd_status == 'warning' else '✗ Fail'}</span></td>
                    </tr>
"""
            
            # MAR Ratio
            mar_ratio = ratios.get('mar_ratio', 0)
            mar_status = "good" if mar_ratio >= 0.4 else "warning" if mar_ratio >= 0.1 else "fail"
            html += f"""
                    <tr>
                        <td>MAR Ratio</td>
                        <td>{is_m.get('mar', 0):.2f}</td>
                        <td>{oos_m.get('mar', 0):.2f}</td>
                        <td>{mar_ratio:.2f}x</td>
                        <td><span class="status-{mar_status}">{'✓ Good' if mar_status == 'good' else '⚠ Warning' if mar_status == 'warning' else '✗ Fail'}</span></td>
                    </tr>
"""
            
            # Win Rate
            wr_diff = abs(oos_m.get('win_rate', 0) - is_m.get('win_rate', 0)) * 100
            wr_status = "good" if wr_diff <= 5 else "warning" if wr_diff <= 10 else "fail"
            html += f"""
                    <tr>
                        <td>Win Rate</td>
                        <td>{is_m.get('win_rate', 0):.2%}</td>
                        <td>{oos_m.get('win_rate', 0):.2%}</td>
                        <td>{wr_diff:.1f}pp</td>
                        <td><span class="status-{wr_status}">{'✓ Good' if wr_status == 'good' else '⚠ Warning' if wr_status == 'warning' else '✗ Fail'}</span></td>
                    </tr>
"""
            
            html += """
                </tbody>
            </table>
            
            <h3>Validation Checks</h3>
            <div class="verdict-box">
"""
            for verdict_text, status in verdict.get('verdicts', []):
                # Remove ANSI codes and extract status
                clean_text = verdict_text.replace(Colors.GREEN, '').replace(Colors.YELLOW, '').replace(Colors.RED, '').replace(Colors.RESET, '').replace(Colors.BOLD, '')
                status_class = "good" if status == "GOOD" else "warning" if status == "WARNING" else "fail"
                html += f'                <div class="check-item"><span class="status-{status_class}">{clean_text}</span></div>\n'
            
            html += """
            </div>
        </div>
"""
    
    # Add Monte Carlo verdict
    if mc_verdict:
        status_class = "good" if mc_verdict['overall_status'] == "GOOD" else "warning" if mc_verdict['overall_status'] == "WARNING" else "fail"
        metrics = mc_verdict.get('metrics', {})
        html += f"""
        <div class="section">
            <h2>🎲 Monte Carlo Validation</h2>
            <div class="verdict-box verdict-{status_class}">
                <h3>Overall Verdict: <span class="status-{status_class}">{mc_verdict['overall_status']}</span></h3>
            </div>
            
            <div class="summary-box">
                <p><strong>Monte Carlo Simulations:</strong> {mc_verdict['mc_data'].get('n_paths', 0):,}</p>
                <p><strong>Trades per Path:</strong> {mc_verdict['mc_data'].get('n_trades', 0)}</p>
            </div>
            
            <h3>Monte Carlo Equity Curves</h3>
"""
        
        # Add MC equity curve visualization
        if mc_equity_curves:
            best_curve = mc_equity_curves.get('best', [])
            median_curve = mc_equity_curves.get('median', [])
            worst_curve = mc_equity_curves.get('worst', [])
            
            if best_curve and median_curve and worst_curve:
                # Convert numpy types to native Python types for JSON serialization
                best_clean = [float(x) if isinstance(x, (np.integer, np.floating)) else float(x) for x in best_curve]
                median_clean = [float(x) if isinstance(x, (np.integer, np.floating)) else float(x) for x in median_curve]
                worst_clean = [float(x) if isinstance(x, (np.integer, np.floating)) else float(x) for x in worst_curve]
                
                best_json = json.dumps(best_clean)
                median_json = json.dumps(median_clean)
                worst_json = json.dumps(worst_clean)
                n_trades = len(best_clean)
                
                html += f"""
            <div class="chart-container">
                <canvas id="mcEquityChart"></canvas>
            </div>
            <script>
                const ctx2 = document.getElementById('mcEquityChart').getContext('2d');
                const bestData = {best_json};
                const medianData = {median_json};
                const worstData = {worst_json};
                const nTrades = {n_trades};
                
                const mcData = {{
                    labels: Array.from({{length: nTrades}}, (_, i) => i),
                    datasets: [
                        {{
                            label: 'Best Case',
                            data: bestData.map((val, idx) => ({{x: idx, y: val}})),
                            borderColor: 'rgb(39, 174, 96)',
                            backgroundColor: 'rgba(39, 174, 96, 0.1)',
                            borderWidth: 2,
                            fill: false,
                            tension: 0,
                            pointRadius: 0,
                            pointHoverRadius: 4
                        }},
                        {{
                            label: 'Median Case',
                            data: medianData.map((val, idx) => ({{x: idx, y: val}})),
                            borderColor: 'rgb(52, 152, 219)',
                            backgroundColor: 'rgba(52, 152, 219, 0.1)',
                            borderWidth: 2,
                            fill: false,
                            tension: 0,
                            pointRadius: 0,
                            pointHoverRadius: 4
                        }},
                        {{
                            label: 'Worst Case',
                            data: worstData.map((val, idx) => ({{x: idx, y: val}})),
                            borderColor: 'rgb(231, 76, 60)',
                            backgroundColor: 'rgba(231, 76, 60, 0.1)',
                            borderWidth: 2,
                            fill: false,
                            tension: 0,
                            pointRadius: 0,
                            pointHoverRadius: 4
                        }}
                    ]
                }};
                new Chart(ctx2, {{
                    type: 'line',
                    data: mcData,
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {{
                            intersect: false,
                            mode: 'index'
                        }},
                        plugins: {{
                            title: {{
                                display: true,
                                text: 'Monte Carlo Equity Curves (Best, Median, Worst)'
                            }},
                            legend: {{
                                display: true,
                                position: 'top'
                            }},
                            tooltip: {{
                                mode: 'index',
                                intersect: false,
                                callbacks: {{
                                    label: function(context) {{
                                        return context.dataset.label + ': $' + context.parsed.y.toLocaleString('en-US', {{maximumFractionDigits: 2}});
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                title: {{
                                    display: true,
                                    text: 'Trade Number'
                                }}
                            }},
                            y: {{
                                title: {{
                                    display: true,
                                    text: 'Equity ($)'
                                }},
                                ticks: {{
                                    callback: function(value) {{
                                        return '$' + value.toLocaleString('en-US');
                                    }}
                                }}
                            }}
                        }}
                    }}
                }});
            </script>
"""
        
        html += f"""
            <h3>Key Metrics</h3>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">OOS Max Drawdown</div>
                    <div class="metric-value">{test_metrics.get('mdd', 0):.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">OOS MDD Percentile in MC</div>
                    <div class="metric-value">{metrics.get('oos_mdd_percentile', 0):.1f}th</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Probability of Positive Return</div>
                    <div class="metric-value">{metrics.get('popr', 0):.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">5th Percentile Return</div>
                    <div class="metric-value">{metrics.get('ret_5', 0):.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">1st Percentile Return</div>
                    <div class="metric-value">{metrics.get('ret_1', 0):.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Dispersion Ratio</div>
                    <div class="metric-value">{metrics.get('dispersion_ratio', 0):.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">MC Efficiency</div>
                    <div class="metric-value">{metrics.get('mc_efficiency', 0):.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Catastrophic Failure Prob</div>
                    <div class="metric-value">{metrics.get('catastrophic_total', 0):.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Bad Win Rate Paths</div>
                    <div class="metric-value">{metrics.get('bad_win_rate_paths', 0):.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Noise-to-Signal Ratio</div>
                    <div class="metric-value">{metrics.get('nsr', 0):.4f}</div>
                </div>
            </div>
            
            <h3>Validation Checks</h3>
            <div class="verdict-box">
"""
        for verdict_text, status in mc_verdict.get('verdicts', []):
            clean_text = verdict_text.replace(Colors.GREEN, '').replace(Colors.YELLOW, '').replace(Colors.RED, '').replace(Colors.RESET, '').replace(Colors.BOLD, '')
            status_class = "good" if status == "GOOD" else "warning" if status == "WARNING" else "fail"
            html += f'                <div class="check-item"><span class="status-{status_class}">{clean_text}</span></div>\n'
        
        html += """
            </div>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    return html

def print_progress_bar(current, total, bar_length=40):
    """
    Print an ASCII progress bar.
    
    Args:
        current: Current iteration number (0-indexed or 1-indexed)
        total: Total number of iterations
        bar_length: Length of the progress bar in characters
    """
    if total == 0:
        return
    
    # Ensure current is within bounds
    current = min(max(current, 0), total)
    
    # Calculate percentage
    percent = (current / total) * 100
    
    # Calculate filled length
    filled_length = int(bar_length * current // total)
    
    # Create the bar with green color for filled portion
    if filled_length >= bar_length:
        # Fully complete
        filled_bar = Colors.GREEN + '=' * bar_length + Colors.RESET
        bar = filled_bar
    elif filled_length == 0:
        # Not started
        bar = '-' * bar_length
    else:
        # In progress - green for filled, default for remaining
        filled_bar = Colors.GREEN + '=' * filled_length + '>' + Colors.RESET
        remaining_bar = '-' * (bar_length - filled_length - 1)
        bar = filled_bar + remaining_bar
    
    # Print the progress bar (using \r to overwrite the same line)
    print(f'\rMonte Carlo Progress: [{bar}] {percent:.1f}% ({current}/{total})', end='', flush=True)
    
    # If complete, print a newline
    if current >= total:
        print()

def monte_carlo_on_oos(oos_df, n_paths=N_MC_PATHS, return_data=False):
    """
    Run Monte Carlo simulation on OOS data.
    Returns MC data if return_data=True, otherwise prints statistics.
    """
    rets = oos_df["trade_ret"].values
    if len(rets) == 0:
        print("\nNo OOS trades for Monte Carlo.")
        return None
    
    mdd_list = []
    final_list = []
    win_rates = []
    n = len(rets)
    rng = np.random.default_rng(seed=42)
    
    # Determine which P/L column to use for win rate calculation
    if "adjusted_pl" in oos_df.columns:
        pl_col = "adjusted_pl"
    else:
        pl_col = "P/L"
    
    original_win_rate = (oos_df[pl_col] > 0).sum() / len(oos_df) if len(oos_df) > 0 else 0
    
    # Print initial progress message
    print(f"\nRunning Monte Carlo simulation ({n_paths} paths)...")
    
    # For Monte Carlo, we need to use the same dynamic allocation logic as the actual OOS
    # So we'll bootstrap the actual trades (rows) rather than just returns
    for path_idx in range(n_paths):
        # Sample rows (trades) with replacement, not just returns
        # This preserves the relationship between P/L, margin, and contracts
        sample_indices = rng.choice(len(oos_df), size=n, replace=True)
        sample_df = oos_df.iloc[sample_indices].copy().reset_index(drop=True)
        
        # Recalculate equity curve with dynamic allocation (same as stats_block)
        if ALLOCATION_TYPE == "percent" and "adjusted_pl" in sample_df.columns:
            # Use dynamic allocation logic
            eq_values = [RISK_CAPITAL]
            rets_path = []
            for i in range(len(sample_df)):
                current_eq = eq_values[-1]
                # Recalculate contracts for this trade based on current equity
                row = sample_df.iloc[i:i+1].copy()
                row = calculate_contracts_and_pl(row, current_eq, ALLOCATION_TYPE, ALLOCATION_VALUE)
                row["contracts_to_trade"] = row["contracts_to_trade"].astype(int)
                
                # Calculate return based on adjusted P/L and current equity
                adjusted_pl = row["adjusted_pl"].iloc[0]
                ret = adjusted_pl / current_eq if current_eq > 0 else 0
                ret = max(ret, -1.0)  # Cap at -100%
                rets_path.append(ret)
                
                new_eq = current_eq * (1 + ret)
                eq_values.append(max(new_eq, 0.01))
            eq = np.array(eq_values)
        else:
            # Use pre-calculated returns
            sample_rets = sample_df["trade_ret"].values
            eq = equity_curve(sample_rets, RISK_CAPITAL)
        
        mdd_list.append(max_drawdown(eq))
        final_list.append(eq[-1])
        
        # Calculate win rate for this path
        if "adjusted_pl" in sample_df.columns:
            path_pl_col = "adjusted_pl"
        else:
            path_pl_col = "P/L"
        positive_trades = (sample_df[path_pl_col] > 0).sum()
        win_rates.append(positive_trades / n if n > 0 else 0)
        
        # Update progress bar
        print_progress_bar(path_idx + 1, n_paths)
    
    mdd_arr = np.array(mdd_list)
    final_arr = np.array(final_list)
    win_rates_arr = np.array(win_rates)
    
    # Calculate returns as (final_equity / starting_capital) - 1
    ret_median = (np.median(final_arr) / RISK_CAPITAL) - 1
    ret_worst = (np.min(final_arr) / RISK_CAPITAL) - 1
    ret_best = (np.max(final_arr) / RISK_CAPITAL) - 1
    ret_5 = (np.percentile(final_arr, 5) / RISK_CAPITAL) - 1
    ret_1 = (np.percentile(final_arr, 1) / RISK_CAPITAL) - 1
    ret_25 = (np.percentile(final_arr, 25) / RISK_CAPITAL) - 1
    ret_75 = (np.percentile(final_arr, 75) / RISK_CAPITAL) - 1
    ret_95 = (np.percentile(final_arr, 95) / RISK_CAPITAL) - 1
    ret_99 = (np.percentile(final_arr, 99) / RISK_CAPITAL) - 1
    
    # Max drawdown statistics
    # For drawdowns (negative values), more negative = worse
    mdd_worst = np.min(mdd_arr)  # Most negative (worst)
    mdd_median = np.median(mdd_arr)
    mdd_best = np.max(mdd_arr)   # Least negative (best)
    mdd_5 = np.percentile(mdd_arr, 5)   # Worst 5% of paths
    mdd_50 = np.percentile(mdd_arr, 50)  # Median
    mdd_80 = np.percentile(mdd_arr, 80)  # Worse than 80% of paths
    mdd_95 = np.percentile(mdd_arr, 95)  # Best 5% of paths
    mdd_99 = np.percentile(mdd_arr, 99)  # Best 1% of paths
    
    if not return_data:
        print("\nMonte Carlo on final OOS")
        print("------------------------")
        print(f"Paths: {n_paths}, trades per path: {n}")
        print(f"\nTotal Return Statistics:")
        print(f"  Worst case: {ret_worst:.2%} (${np.min(final_arr):,.2f})")
        print(f"  1st percentile: {ret_1:.2%} (${np.percentile(final_arr, 1):,.2f})")
        print(f"  5th percentile: {ret_5:.2%} (${np.percentile(final_arr, 5):,.2f})")
        print(f"  Median: {ret_median:.2%} (${np.median(final_arr):,.2f})")
        print(f"  95th percentile: {ret_95:.2%} (${np.percentile(final_arr, 95):,.2f})")
        print(f"  99th percentile: {ret_99:.2%} (${np.percentile(final_arr, 99):,.2f})")
        print(f"  Best case: {ret_best:.2%} (${np.max(final_arr):,.2f})")
        print(f"\nMax Drawdown Statistics:")
        print(f"  Worst case: {mdd_worst:.2%}")
        print(f"  99th percentile: {mdd_99:.2%}")
        print(f"  95th percentile: {mdd_95:.2%}")
        print(f"  Median: {mdd_median:.2%}")
        print(f"  Best case: {mdd_best:.2%}")
    
    # Return MC data for validation
    return {
        'mdd_arr': mdd_arr,
        'final_arr': final_arr,
        'win_rates_arr': win_rates_arr,
        'ret_median': ret_median,
        'ret_worst': ret_worst,
        'ret_best': ret_best,
        'ret_5': ret_5,
        'ret_1': ret_1,
        'ret_25': ret_25,
        'ret_75': ret_75,
        'ret_95': ret_95,
        'ret_99': ret_99,
        'mdd_worst': mdd_worst,
        'mdd_median': mdd_median,
        'mdd_best': mdd_best,
        'mdd_5': mdd_5,
        'mdd_50': mdd_50,
        'mdd_80': mdd_80,
        'mdd_95': mdd_95,
        'mdd_99': mdd_99,
        'original_win_rate': original_win_rate,
        'n_paths': n_paths,
        'n_trades': n
    }

def main(csv_path):
    path = Path(csv_path)
    if not path.exists():
        raise SystemExit(f"CSV not found at {path.absolute()}")
    
    # Load and prepare data
    df = load_data(path)
    df = add_per_contract_pl(df)
    
    # Calculate contracts and adjusted P/L based on allocation strategy
    # For the first calculation, use initial equity
    df = calculate_contracts_and_pl(df, RISK_CAPITAL, ALLOCATION_TYPE, ALLOCATION_VALUE)
    
    # Add returns based on adjusted P/L
    df = add_returns(df, use_adjusted=True)
    
    # Split into IS/OOS
    is_df, val_df, test_df = split_is_oos(df)
    
    print("Overall")
    print("-------")
    print(f"Total trades: {len(df)}")
    print(f"Period: {df['Date Opened'].min().date()} -> {df['Date Opened'].max().date()}")
    print(f"\nAllocation Strategy:")
    print(f"  Type: {ALLOCATION_TYPE}")
    if ALLOCATION_TYPE == "percent":
        print(f"  Value: {ALLOCATION_VALUE*100:.1f}% of equity per trade")
        print(f"  Initial equity: ${RISK_CAPITAL:,.2f}")
    else:
        print(f"  Value: {ALLOCATION_VALUE} contracts per trade")
    print(f"\nPer-contract calculations:")
    print(f"  Premium per contract: Calculated from 'Premium' / 'No. of Contracts'")
    print(f"  Closing cost per contract: Calculated from 'Avg. Closing Cost' / 'No. of Contracts'")
    if "P/L" in df.columns:
        print(f"  P/L per contract: Calculated from 'P/L' / 'No. of Contracts' (most accurate)")
    else:
        print(f"  P/L per contract: premium_per_contract - closing_cost_per_contract")
    print(f"\nNote: P/L per trade is scaled based on allocation (contracts_to_trade)")
    
    # Calculate and display statistics with equity curves
    is_eq = stats_block("In-sample (train)", is_df, RISK_CAPITAL, use_adjusted=True)
    val_eq = stats_block("Validation OOS (dev)", val_df, RISK_CAPITAL, use_adjusted=True)
    test_eq = stats_block("Final OOS (test)", test_df, RISK_CAPITAL, use_adjusted=True)
    
    # Calculate metrics for validation
    is_metrics = calculate_metrics(is_df, is_eq, RISK_CAPITAL)
    val_metrics = calculate_metrics(val_df, val_eq, RISK_CAPITAL)
    test_metrics = calculate_metrics(test_df, test_eq, RISK_CAPITAL)
    
    # Calculate full equity curve with dates for visualization
    full_eq_curve = calculate_full_equity_curve(df, RISK_CAPITAL)
    
    # Run Monte Carlo and get data for validation
    mc_data = monte_carlo_on_oos(test_df, return_data=False)
    mc_data = monte_carlo_on_oos(test_df, return_data=True)  # Get data for validation
    
    # Calculate MC equity curves for best, median, worst cases
    mc_equity_curves = calculate_mc_equity_curves(test_df, mc_data)
    
    # Print validation verdicts
    val_verdict = None
    test_verdict = None
    if is_metrics and val_metrics:
        val_verdict = validate_oos_performance(is_metrics, val_metrics, "Validation OOS (dev)")
        print_verdict(val_verdict, "Validation OOS (dev)")
    
    if is_metrics and test_metrics:
        test_verdict = validate_oos_performance(is_metrics, test_metrics, "Final OOS (test)")
        print_verdict(test_verdict, "Final OOS (test)")
    
    # Print Monte Carlo verdict
    mc_verdict = None
    if mc_data and test_metrics:
        mc_verdict = validate_mc_performance(mc_data, test_metrics)
        print_mc_verdict(mc_verdict)
    
    # Generate HTML report
    html_report = generate_html_report(
        df, is_df, val_df, test_df,
        is_metrics, val_metrics, test_metrics,
        val_verdict, test_verdict, mc_verdict,
        is_eq, val_eq, test_eq,
        csv_path, full_eq_curve, mc_equity_curves
    )
    
    # Save HTML report
    csv_filename = Path(csv_path).stem  # Get filename without extension
    report_path = Path(f"is_oos_report_{csv_filename}.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}✓ HTML report saved to: {report_path.absolute()}{Colors.RESET}")
    if full_eq_curve is not None and not full_eq_curve.empty:
        print(f"{Colors.GREEN}  - Equity curve chart included ({len(full_eq_curve)} data points){Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}  - Warning: Equity curve data not available for chart{Colors.RESET}")
    if mc_equity_curves:
        print(f"{Colors.GREEN}  - Monte Carlo equity curves included{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}  - Warning: MC equity curves not available{Colors.RESET}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IS/OOS validation tool for trading strategy backtests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py strategy_backtest_logs/Monday-2-4-DC.csv
  python main.py trade-log-rics.csv
  python main.py /path/to/your/trade_log.csv
        """
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=DEFAULT_CSV_PATH,
        help=f"Path to the CSV file containing trade log data (default: {DEFAULT_CSV_PATH})"
    )
    args = parser.parse_args()
    main(args.csv_path)
