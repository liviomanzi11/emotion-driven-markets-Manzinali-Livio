"""
figures_company.py
==================
Generates visualizations comparing ML + Sentiment strategies across 6 companies.

Creates 9 professional outputs:
- 3 Technical figures (returns, drawdown, winners)
- 3 Sentiment figures (returns, drawdown, winners)
- 1 Equity curves comparison
- 1 Overall summary table (CSV + PNG)
- 1 Overall performance comparison

Outputs saved to results/figures_company/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Project structure
BASE = Path(__file__).resolve().parents[2]
RESULTS = BASE / "results"
OUT = RESULTS / "figures_company"
OUT.mkdir(parents=True, exist_ok=True)
EQUITY_DIR = BASE / "results" / "equity_curves"
EQUITY_DIR.mkdir(parents=True, exist_ok=True)

# Features directory (to load date/time index for synthetic curves)
FEATURES_DIR = BASE / "data" / "processed" / "features_company"

# Backtest results location
BACKTEST_FILE = OUT / "strategy" / "backtest_company_results.txt"

# List of tickers
TICKERS = ["AAPL", "GOOG", "GOOGL", "AMZN", "TSLA", "MSFT"]

# Model mapping for parsing
MODEL_MAPPING = {
    "Logistic Regression (Technical)": ("tech", "LR"),
    "Logistic Regression (Sentiment)": ("sent", "LR"),
    "Random Forest (Technical)": ("tech", "RF"),
    "Random Forest (Sentiment)": ("sent", "RF"),
    "XGBoost (Technical)": ("tech", "XGB"),
    "XGBoost (Sentiment)": ("sent", "XGB"),
    "LSTM (Technical)": ("tech", "LSTM"),
    "LSTM (Sentiment)": ("sent", "LSTM"),
    "Buy & Hold": ("baseline", "BH")
}

# Color mappings (consistent across all plots)
MODEL_COLORS = {
    "LR": "#377eb8",   # blue
    "RF": "#ff7f00",   # orange
    "XGB": "#4daf4a",  # green
    "LSTM": "#e41a1c"  # red
}

TYPE_COLORS = {
    "Technical": "steelblue",
    "Sentiment": "coral",
    "Buy & Hold": "lightgreen"
}


def parse_backtest_results():
    """
    Parse backtest results file with 11 strategies per ticker.
    
    Returns:
        dict: {
            "tech": {"LR": {ticker: {...}}, "RF": {...}, ...},
            "sent": {"LR": {ticker: {...}}, "RF": {...}, ...},
            "buyhold": {ticker: {...}}
        }
    """
    if not BACKTEST_FILE.exists():
        print(f"   [WARNING] Backtest file not found: {BACKTEST_FILE}")
        return None
    
    with open(BACKTEST_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    results = {
        "tech": {model: {ticker: {} for ticker in TICKERS} for model in ["LR", "RF", "XGB", "LSTM"]},
        "sent": {model: {ticker: {} for ticker in TICKERS} for model in ["LR", "RF", "XGB", "LSTM"]},
        "buyhold": {ticker: {} for ticker in TICKERS}
    }
    
    current_ticker = None
    current_model = None
    current_category = None
    
    for line in lines:
        line = line.strip()
        
        # Parser state machine: ticker line resets context for following model lines
        if line in TICKERS:
            current_ticker = line
            continue
        
        # Model name determines where to store following metrics (tech/sent/buyhold)
        for model_name, (category, model_code) in MODEL_MAPPING.items():
            if line.startswith(model_name + ":"):
                current_model = model_code
                current_category = category
                break
        
        # Metrics belong to the current ticker+model context established by previous lines
        if current_ticker and current_model and current_category:
            if "Total Return:" in line:
                val = line.split("Total Return:")[1].strip().replace("%", "")
                try:
                    val = float(val)
                except:
                    val = 0.0
                
                if current_category == "baseline":
                    results["buyhold"][current_ticker]["return"] = val
                else:
                    results[current_category][current_model][current_ticker]["return"] = val
                    
            elif "Sharpe Ratio:" in line:
                val = line.split("Sharpe Ratio:")[1].strip()
                try:
                    val = float(val)
                except:
                    val = 0.0
                
                if current_category == "baseline":
                    results["buyhold"][current_ticker]["sharpe"] = val
                else:
                    results[current_category][current_model][current_ticker]["sharpe"] = val
                    
            elif "Max Drawdown:" in line:
                val = line.split("Max Drawdown:")[1].strip().replace("%", "")
                try:
                    val = float(val)
                except:
                    val = 0.0
                
                if current_category == "baseline":
                    results["buyhold"][current_ticker]["drawdown"] = val
                    current_model = None
                    current_category = None
                else:
                    results[current_category][current_model][current_ticker]["drawdown"] = val
                    current_model = None
                    current_category = None
    
    return results




def figure_technical_returns():
    """Figure A1: Technical Returns Bar Chart"""
    print("  > Technical Returns")
    
    results = parse_backtest_results()
    if results is None:
        return
    
    models = ["LR", "RF", "XGB", "LSTM"]
    x = np.arange(len(TICKERS))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Fixed positions: LR=-2, RF=-1, XGB=0, LSTM=+1, Buy&Hold=+2
    positions = [-2, -1, 0, 1]
    
    for i, model in enumerate(models):
        returns = [results["tech"][model][t].get("return", 0) for t in TICKERS]
        offset = positions[i] * width
        bars = ax.bar(x + offset, returns, width, label=model, color=MODEL_COLORS.get(model, None), alpha=0.85)
        
        # Add value labels at the end of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8, fontweight='bold')
    
    # Buy & Hold at position +2
    bh_returns = [results["buyhold"][t].get("return", 0) for t in TICKERS]
    bars = ax.bar(x + 2*width, bh_returns, width, label="Buy & Hold", color=TYPE_COLORS["Buy & Hold"], alpha=0.85)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=8, fontweight='bold')
    
    ax.set_xlabel("Company", fontsize=13, fontweight="bold")
    ax.set_ylabel("Total Return (%)", fontsize=13, fontweight="bold")
    ax.set_title("Technical Features: Total Returns by Model", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(TICKERS, fontsize=12)
    ax.legend(fontsize=11, ncol=5)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT / "returns_technical.png", dpi=300, bbox_inches="tight")
    plt.close()


def figure_technical_drawdown():
    """Figure A2: Technical Drawdown Bar Chart"""
    print("  > Technical Drawdown")
    results = parse_backtest_results()
    if results is None:
        return
    models = ["LR", "RF", "XGB", "LSTM"]
    x = np.arange(len(TICKERS))
    width = 0.15
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Fixed positions: LR=-2, RF=-1, XGB=0, LSTM=+1, Buy&Hold=+2
    positions = [-2, -1, 0, 1]
    
    for i, model in enumerate(models):
        drawdowns = [results["tech"][model][t].get("drawdown", 0) for t in TICKERS]
        offset = positions[i] * width
        bars = ax.bar(x + offset, drawdowns, width, label=model, color=MODEL_COLORS.get(model, None), alpha=0.85)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='top' if height < 0 else 'bottom',
                    fontsize=8, fontweight='bold')
    
    # Buy & Hold at position +2
    bh_dd = [results["buyhold"][t].get("drawdown", 0) for t in TICKERS]
    bars = ax.bar(x + 2*width, bh_dd, width, label="Buy & Hold", color=TYPE_COLORS["Buy & Hold"], alpha=0.85)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='top' if height < 0 else 'bottom',
                fontsize=8, fontweight='bold')
    ax.set_xlabel("Company", fontsize=13, fontweight="bold")
    ax.set_ylabel("Max Drawdown (%)", fontsize=13, fontweight="bold")
    ax.set_title("Technical Features: Maximum Drawdown by Model", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(TICKERS, fontsize=12)
    ax.legend(fontsize=11, ncol=5)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "drawdown_technical.png", dpi=300, bbox_inches="tight")
    plt.close()


def figure_technical_winners():
    """Figure A3: Technical Winners Analysis"""
    print("  > Technical Winners")
    results = parse_backtest_results()
    if results is None:
        return
    models = ["LR", "RF", "XGB", "LSTM"]
    model_wins = {m: 0 for m in models}
    best_per_ticker = []
    for ticker in TICKERS:
        returns = {m: results["tech"][m][ticker].get("return", 0) for m in models}
        best_model = max(returns, key=returns.get)
        model_wins[best_model] += 1
        best_per_ticker.append((ticker, best_model, returns[best_model]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    tickers = [b[0] for b in best_per_ticker]
    returns = [b[2] for b in best_per_ticker]
    colors = [MODEL_COLORS.get(b[1], f"C{models.index(b[1])}") for b in best_per_ticker]
    bars = ax1.bar(tickers, returns, color=colors, alpha=0.8, edgecolor="black")
    ax1.set_xlabel("Company", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Best Return (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Best Technical Model per Company", fontsize=13, fontweight="bold")
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.grid(axis="y", alpha=0.3)
    for bar, best in zip(bars, best_per_ticker):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f"{best[1]}\n{best[2]:.1f}%",
                ha="center", va="bottom" if height >= 0 else "top", fontsize=9, fontweight="bold")
    non_zero_wins = {k: v for k, v in model_wins.items() if v > 0}
    if non_zero_wins:
        wedges, texts, autotexts = ax2.pie(non_zero_wins.values(), labels=non_zero_wins.keys(),
                autopct=lambda pct: f'{pct:.0f}%' if pct > 5 else '',
                startangle=90, colors=[MODEL_COLORS.get(k, f"C{list(model_wins.keys()).index(k)}") for k in non_zero_wins.keys()],
                textprops={'fontsize': 11, 'fontweight': 'bold'})
        for autotext in autotexts:
            autotext.set_color('white')
    ax2.set_title("Technical Model Win Rate", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "winners_technical.png", dpi=300, bbox_inches="tight")
    plt.close()


def figure_sentiment_returns():
    """Figure B1: Sentiment Returns Bar Chart"""
    print("  > Sentiment Returns")
    
    results = parse_backtest_results()
    if results is None:
        return
    
    models = ["LR", "RF", "XGB", "LSTM"]
    x = np.arange(len(TICKERS))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Fixed positions: LR=-2, RF=-1, XGB=0, LSTM=+1, Buy&Hold=+2
    positions = [-2, -1, 0, 1]
    
    for i, model in enumerate(models):
        returns = [results["sent"][model][t].get("return", 0) for t in TICKERS]
        offset = positions[i] * width
        bars = ax.bar(x + offset, returns, width, label=model, color=MODEL_COLORS.get(model, None), alpha=0.85)
        
        # Add value labels at the end of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8, fontweight='bold')
    
    # Buy & Hold at position +2
    bh_returns = [results["buyhold"][t].get("return", 0) for t in TICKERS]
    bars = ax.bar(x + 2*width, bh_returns, width, label="Buy & Hold", color=TYPE_COLORS["Buy & Hold"], alpha=0.85)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=8, fontweight='bold')
    
    ax.set_xlabel("Company", fontsize=13, fontweight="bold")
    ax.set_ylabel("Total Return (%)", fontsize=13, fontweight="bold")
    ax.set_title("Sentiment Features: Total Returns by Model", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(TICKERS, fontsize=12)
    ax.legend(fontsize=11, ncol=5)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT / "returns_sentiment.png", dpi=300, bbox_inches="tight")
    plt.close()


def figure_sentiment_drawdown():
    """Figure B2: Sentiment Drawdown Bar Chart"""
    print("  > Sentiment Drawdown")
    results = parse_backtest_results()
    if results is None:
        return
    models = ["LR", "RF", "XGB", "LSTM"]
    x = np.arange(len(TICKERS))
    width = 0.15
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Fixed positions: LR=-2, RF=-1, XGB=0, LSTM=+1, Buy&Hold=+2
    positions = [-2, -1, 0, 1]
    
    for i, model in enumerate(models):
        drawdowns = [results["sent"][model][t].get("drawdown", 0) for t in TICKERS]
        offset = positions[i] * width
        bars = ax.bar(x + offset, drawdowns, width, label=model, color=MODEL_COLORS.get(model, None), alpha=0.85)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='top' if height < 0 else 'bottom',
                    fontsize=8, fontweight='bold')
    
    # Buy & Hold at position +2
    bh_dd = [results["buyhold"][t].get("drawdown", 0) for t in TICKERS]
    bars = ax.bar(x + 2*width, bh_dd, width, label="Buy & Hold", color=TYPE_COLORS["Buy & Hold"], alpha=0.85)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='top' if height < 0 else 'bottom',
                fontsize=8, fontweight='bold')
    ax.set_xlabel("Company", fontsize=13, fontweight="bold")
    ax.set_ylabel("Max Drawdown (%)", fontsize=13, fontweight="bold")
    ax.set_title("Sentiment Features: Maximum Drawdown by Model", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(TICKERS, fontsize=12)
    ax.legend(fontsize=11, ncol=5)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "drawdown_sentiment.png", dpi=300, bbox_inches="tight")
    plt.close()


def figure_sentiment_winners():
    """Figure B3: Sentiment Winners Analysis"""
    print("  > Sentiment Winners")
    
    results = parse_backtest_results()
    if results is None:
        return
    
    models = ["LR", "RF", "XGB", "LSTM"]
    model_wins = {m: 0 for m in models}
    best_per_ticker = []
    
    for ticker in TICKERS:
        returns = {m: results["sent"][m][ticker].get("return", 0) for m in models}
        best_model = max(returns, key=returns.get)
        model_wins[best_model] += 1
        best_per_ticker.append((ticker, best_model, returns[best_model]))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Best model per ticker
    tickers = [b[0] for b in best_per_ticker]
    returns = [b[2] for b in best_per_ticker]
    colors = [MODEL_COLORS.get(b[1], f"C{models.index(b[1])}") for b in best_per_ticker]
    
    bars = ax1.bar(tickers, returns, color=colors, alpha=0.8, edgecolor="black")
    ax1.set_xlabel("Company", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Best Return (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Best Sentiment Model per Company", fontsize=13, fontweight="bold")
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.grid(axis="y", alpha=0.3)
    
    for bar, best in zip(bars, best_per_ticker):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f"{best[1]}\n{best[2]:.1f}%",
                ha="center", va="bottom" if height >= 0 else "top", fontsize=9, fontweight="bold")
    
    # Right: Win distribution pie chart
    non_zero_wins = {k: v for k, v in model_wins.items() if v > 0}
    if non_zero_wins:
        wedges, texts, autotexts = ax2.pie(non_zero_wins.values(), labels=non_zero_wins.keys(),
                autopct=lambda pct: f'{pct:.0f}%' if pct > 5 else '',
                startangle=90, colors=[MODEL_COLORS.get(k, f"C{list(model_wins.keys()).index(k)}") for k in non_zero_wins.keys()],
                textprops={'fontsize': 11, 'fontweight': 'bold'})
        for autotext in autotexts:
            autotext.set_color('white')
    ax2.set_title("Sentiment Model Win Rate", fontsize=13, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(OUT / "winners_sentiment.png", dpi=300, bbox_inches="tight")
    plt.close()



def figure_equity_curves():
    """Figure C: Equity Curves - Best Tech vs Best Sent vs Buy & Hold per ticker"""
    print("  → Equity Curves")
    
    results = parse_backtest_results()
    if results is None:
        return
    
    models = ["LR", "RF", "XGB", "LSTM"]

    # Use top-level synthetic_equity_from_total helper
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, ticker in enumerate(TICKERS):
        ax = axes[idx]
        
        # Get best tech and best sent
        tech_returns = {m: results["tech"][m][ticker].get("return", 0) for m in models}
        sent_returns = {m: results["sent"][m][ticker].get("return", 0) for m in models}
        
        best_tech = max(tech_returns, key=tech_returns.get)
        best_sent = max(sent_returns, key=sent_returns.get)
        bh_return = results["buyhold"][ticker].get("return", 0)
        
        # Time-series visualization requires daily equity values, not just summary metrics
        def model_code_to_slug(code, features_type):
            mapping = {"LR": "lr", "RF": "rf", "XGB": "xgb", "LSTM": "lstm"}
            return f"{mapping.get(code, code.lower())}_{features_type}"

        try:
            tech_slug = model_code_to_slug(best_tech, "technical")
            sent_slug = model_code_to_slug(best_sent, "sentiment")
            bh_slug = "buyhold"
            df_eq_tech = None
            df_eq_sent = None
            df_eq_bh = None
            ftech = EQUITY_DIR / f"{ticker}_{tech_slug}.csv"
            fsent = EQUITY_DIR / f"{ticker}_{sent_slug}.csv"
            fbh = EQUITY_DIR / f"{ticker}_{bh_slug}.csv"
            import pandas as _pd
            if ftech.exists():
                df_eq_tech = _pd.read_csv(ftech, parse_dates=["date"]).sort_values("date")
            if fsent.exists():
                df_eq_sent = _pd.read_csv(fsent, parse_dates=["date"]).sort_values("date")
            if fbh.exists():
                df_eq_bh = _pd.read_csv(fbh, parse_dates=["date"]).sort_values("date")
        except Exception:
            df_eq_tech = df_eq_sent = df_eq_bh = None

        # If we have dates, plot time-series; otherwise fall back to bars
        if df_eq_tech is not None and df_eq_sent is not None and df_eq_bh is not None:
            # Normalize to 100 and plot realistic equity curves
            df_eq_tech["equity_norm"] = df_eq_tech["equity"] / df_eq_tech["equity"].iloc[0] * 100
            df_eq_sent["equity_norm"] = df_eq_sent["equity"] / df_eq_sent["equity"].iloc[0] * 100
            df_eq_bh["equity_norm"] = df_eq_bh["equity"] / df_eq_bh["equity"].iloc[0] * 100
            ax.plot(df_eq_tech["date"], df_eq_tech["equity_norm"], label=f"Tech ({best_tech})", color=TYPE_COLORS["Technical"])
            ax.plot(df_eq_sent["date"], df_eq_sent["equity_norm"], label=f"Sent ({best_sent})", color=TYPE_COLORS["Sentiment"])
            ax.plot(df_eq_bh["date"], df_eq_bh["equity_norm"], label="Buy & Hold", color=TYPE_COLORS["Buy & Hold"])
            # Choose three ticks: start, middle, end
            ticks = [df_eq_tech['date'].iloc[0], df_eq_tech['date'].iloc[len(df_eq_tech)//2], df_eq_tech['date'].iloc[-1]]
            ax.set_xticks(ticks)
            ax.set_xticklabels([pd.to_datetime(d).strftime("%Y-%m-%d") for d in ticks], fontsize=9)
            ax.set_ylabel("Equity (base 100)", fontsize=11)
            ax.legend(fontsize=9)
            ax.set_title(f"{ticker}", fontsize=13, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
        else:
            # Fallback: show final returns as bars (original behavior)
            strategies = [f"Tech ({best_tech})", f"Sent ({best_sent})", "Buy & Hold"]
            returns = [tech_returns[best_tech], sent_returns[best_sent], bh_return]
            colors = [TYPE_COLORS["Technical"], TYPE_COLORS["Sentiment"], TYPE_COLORS["Buy & Hold"]]
            bars = ax.bar(strategies, returns, color=colors, alpha=0.8, edgecolor="black")
            ax.set_title(f"{ticker}", fontsize=13, fontweight="bold")
            ax.set_ylabel("Return (%)", fontsize=11)
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
            ax.grid(axis="y", alpha=0.3)
        
        if 'bars' in locals():
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f"{height:.1f}%",
                        ha="center", va="bottom" if height >= 0 else "top", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUT / "equity_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def table_overall_summary():
    """Table D: Overall Summary CSV + PNG"""
    print("  > Overall Summary Table")
    
    results = parse_backtest_results()
    if results is None:
        return
    
    models = ["LR", "RF", "XGB", "LSTM"]
    
    summary_data = []
    
    for ticker in TICKERS:
        # Best technical
        tech_returns = {m: results["tech"][m][ticker].get("return", 0) for m in models}
        best_tech_model = max(tech_returns, key=tech_returns.get)
        best_tech_return = tech_returns[best_tech_model]
        
        # Best sentiment
        sent_returns = {m: results["sent"][m][ticker].get("return", 0) for m in models}
        best_sent_model = max(sent_returns, key=sent_returns.get)
        best_sent_return = sent_returns[best_sent_model]
        
        # Buy & Hold
        bh_return = results["buyhold"][ticker].get("return", 0)
        
        # Global winner
        max_return = max(best_tech_return, best_sent_return, bh_return)
        if max_return == best_tech_return:
            global_winner = f"Tech ({best_tech_model})"
        elif max_return == best_sent_return:
            global_winner = f"Sent ({best_sent_model})"
        else:
            global_winner = "Buy & Hold"
        
        summary_data.append({
            "Ticker": ticker,
            "Best Technical": best_tech_model,
            "Tech Return (%)": f"{best_tech_return:.2f}",
            "Best Sentiment": best_sent_model,
            "Sent Return (%)": f"{best_sent_return:.2f}",
            "B&H Return (%)": f"{bh_return:.2f}",
            "Global Winner": global_winner
        })
    
    # Save CSV
    df = pd.DataFrame(summary_data)
    df.to_csv(OUT / "overall_summary.csv", index=False)
    
    # Create table figure
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("tight")
    ax.axis("off")
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc="center", loc="center",
                     colWidths=[0.12, 0.15, 0.15, 0.15, 0.15, 0.13, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    plt.title("Overall Performance Summary", fontsize=14, fontweight="bold", pad=20)
    plt.savefig(OUT / "overall_summary_table.png", dpi=300, bbox_inches="tight")
    plt.close()


def figure_overall_performance_comparison():
    """Figure E: Overall Performance Comparison - Mean Best Tech vs Mean Best Sent vs B&H"""
    print("  > Overall Performance Comparison")
    
    results = parse_backtest_results()
    if results is None:
        return
    
    models = ["LR", "RF", "XGB", "LSTM"]
    
    # Calculate mean of best models
    best_tech_returns = []
    best_tech_sharpe = []
    best_tech_dd = []
    
    best_sent_returns = []
    best_sent_sharpe = []
    best_sent_dd = []
    
    bh_returns = []
    bh_sharpe = []
    bh_dd = []
    
    for ticker in TICKERS:
        # Best tech
        tech_ret = {m: results["tech"][m][ticker].get("return", 0) for m in models}
        best_tech_model = max(tech_ret, key=tech_ret.get)
        best_tech_returns.append(tech_ret[best_tech_model])
        best_tech_sharpe.append(results["tech"][best_tech_model][ticker].get("sharpe", 0))
        best_tech_dd.append(results["tech"][best_tech_model][ticker].get("drawdown", 0))
        
        # Best sent
        sent_ret = {m: results["sent"][m][ticker].get("return", 0) for m in models}
        best_sent_model = max(sent_ret, key=sent_ret.get)
        best_sent_returns.append(sent_ret[best_sent_model])
        best_sent_sharpe.append(results["sent"][best_sent_model][ticker].get("sharpe", 0))
        best_sent_dd.append(results["sent"][best_sent_model][ticker].get("drawdown", 0))
        
        # Buy & Hold
        bh_returns.append(results["buyhold"][ticker].get("return", 0))
        bh_sharpe.append(results["buyhold"][ticker].get("sharpe", 0))
        bh_dd.append(results["buyhold"][ticker].get("drawdown", 0))
    
    # Means
    mean_tech = [np.mean(best_tech_returns), np.mean(best_tech_sharpe), np.mean(best_tech_dd)]
    mean_sent = [np.mean(best_sent_returns), np.mean(best_sent_sharpe), np.mean(best_sent_dd)]
    mean_bh = [np.mean(bh_returns), np.mean(bh_sharpe), np.mean(bh_dd)]
    
    metrics = ["Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)"]
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width, mean_tech, width, label="Best Technical", color=TYPE_COLORS["Technical"], alpha=0.85)
    bars2 = ax.bar(x, mean_sent, width, label="Best Sentiment", color=TYPE_COLORS["Sentiment"], alpha=0.85)
    bars3 = ax.bar(x + width, mean_bh, width, label="Buy & Hold", color=TYPE_COLORS["Buy & Hold"], alpha=0.85)
    
    ax.set_ylabel("Value", fontsize=13, fontweight="bold")
    ax.set_title("Overall Performance: Mean Best Technical vs Mean Best Sentiment vs Buy & Hold",
                fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=12)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f"{height:.2f}",
                    ha="center", va="bottom" if height >= 0 else "top", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(OUT / "overall_performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def figure_average_equity_comparison():
    """Plot average equity (normalized) across tickers for Best Technical, Best Sentiment, and Buy & Hold.
    The function constructs synthetic daily equity curves (from total returns) aligned by an intersected date range
    and plots the mean equity for each type.
    """
    print("  > Average Equity Comparison (Best Tech vs Best Sent vs Buy & Hold)")
    results = parse_backtest_results()
    if results is None:
        return

    models = ["LR", "RF", "XGB", "LSTM"]
    tech_series = []
    sent_series = []
    bh_series = []
    # capture start and end dates to compute the intersection
    start_dates = []
    end_dates = []
    for ticker in TICKERS:
        # model names
        tech_ret_map = {m: results["tech"][m][ticker].get("return", 0) for m in models}
        sent_ret_map = {m: results["sent"][m][ticker].get("return", 0) for m in models}
        best_tech = max(tech_ret_map, key=tech_ret_map.get)
        best_sent = max(sent_ret_map, key=sent_ret_map.get)
        bh_return = results["buyhold"][ticker].get("return", 0)
        # read dates from features
        file = FEATURES_DIR / f"{ticker}_features_test.csv"
        if not file.exists():
            continue
        df_dates = pd.read_csv(file, parse_dates=["date"], usecols=["date"])  # only dates
        dates = pd.to_datetime(df_dates["date"])  # pandas Series
        n = len(dates)
        if n <= 1:
            continue
        # Load equity CSVs for the best tech, best sent, and buyhold
        def model_code_to_slug(code, features_type):
            mapping = {"LR": "lr", "RF": "rf", "XGB": "xgb", "LSTM": "lstm"}
            return f"{mapping.get(code, code.lower())}_{features_type}"

        tech_slug = model_code_to_slug(best_tech, "technical")
        sent_slug = model_code_to_slug(best_sent, "sentiment")
        ftech = EQUITY_DIR / f"{ticker}_{tech_slug}.csv"
        fsent = EQUITY_DIR / f"{ticker}_{sent_slug}.csv"
        fbh = EQUITY_DIR / f"{ticker}_buyhold.csv"
        import pandas as _pd
        df_eq_tech = _pd.read_csv(ftech, parse_dates=["date"]).sort_values("date") if ftech.exists() else None
        df_eq_sent = _pd.read_csv(fsent, parse_dates=["date"]).sort_values("date") if fsent.exists() else None
        df_eq_bh = _pd.read_csv(fbh, parse_dates=["date"]).sort_values("date") if fbh.exists() else None
        # Normalize per-ticker series to base 100 for mean aggregation
        if df_eq_tech is not None:
            df_eq_tech["equity_norm"] = df_eq_tech["equity"] / df_eq_tech["equity"].iloc[0] * 100
        if df_eq_sent is not None:
            df_eq_sent["equity_norm"] = df_eq_sent["equity"] / df_eq_sent["equity"].iloc[0] * 100
        if df_eq_bh is not None:
            df_eq_bh["equity_norm"] = df_eq_bh["equity"] / df_eq_bh["equity"].iloc[0] * 100
        # Create pd.Series with date index from loaded CSVs
        if df_eq_tech is not None:
            s_tech = pd.Series(df_eq_tech["equity_norm"].values, index=pd.to_datetime(df_eq_tech["date"]))
            tech_series.append(s_tech)
        if df_eq_sent is not None:
            s_sent = pd.Series(df_eq_sent["equity_norm"].values, index=pd.to_datetime(df_eq_sent["date"]))
            sent_series.append(s_sent)
        if df_eq_bh is not None:
            s_bh = pd.Series(df_eq_bh["equity_norm"].values, index=pd.to_datetime(df_eq_bh["date"]))
            bh_series.append(s_bh)
        start_dates.append(dates.min())
        end_dates.append(dates.max())

    if not tech_series and not sent_series and not bh_series:
        print("   [WARNING] No features dates found for tickers; cannot build average equity comparison.")
        return

    # Determine intersection date range
    overall_start = max(start_dates)
    overall_end = min(end_dates)
    if overall_start > overall_end:
        # fallback: use overlapping by picking min length and aligning from the end
        overall_start = max(start_dates)
        overall_end = min(end_dates)

    # Reindex each series to the intersection and compute mean
    def mean_series(series_list, start, end):
        trimmed = []
        for s in series_list:
            # Use .loc with pandas Timestamps for robust slicing
            s2 = s.loc[start:end]
            # If index doesn't align, reindex to a daily date range
            trimmed.append(s2)
        # Align by intersection index
        df = pd.concat(trimmed, axis=1, join="inner")
        df.columns = range(df.shape[1])
        return df.mean(axis=1)

    mean_tech = mean_series(tech_series, overall_start, overall_end)
    mean_sent = mean_series(sent_series, overall_start, overall_end)
    mean_bh = mean_series(bh_series, overall_start, overall_end)

    # Normalize to base 100 for visualization
    mean_tech_norm = mean_tech / mean_tech.iloc[0] * 100
    mean_sent_norm = mean_sent / mean_sent.iloc[0] * 100
    mean_bh_norm = mean_bh / mean_bh.iloc[0] * 100

    # Combine into DataFrame
    df_mean = pd.DataFrame({"Best Technical": mean_tech_norm, "Best Sentiment": mean_sent_norm, "Buy & Hold": mean_bh_norm})
    df_mean.to_csv(OUT / "avg_equity_comparison.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_mean.index, df_mean["Best Technical"], label="Best Technical", color=TYPE_COLORS["Technical"])
    ax.plot(df_mean.index, df_mean["Best Sentiment"], label="Best Sentiment", color=TYPE_COLORS["Sentiment"])
    ax.plot(df_mean.index, df_mean["Buy & Hold"], label="Buy & Hold", color=TYPE_COLORS["Buy & Hold"])
    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("Index (base 100)", fontsize=12, fontweight="bold")
    ax.set_title("Average Equity: Best Technical vs Best Sentiment vs Buy & Hold (normalized)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "avg_equity_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()



def run_figures_company():
    """Generate all visualization outputs for company backtesting"""
    print("\n=== GENERATING COMPANY BACKTEST FIGURES ===\n")
    
    print("[1/3] Technical Figures")
    figure_technical_returns()
    figure_technical_drawdown()
    figure_technical_winners()
    
    print("\n[2/3] Sentiment Figures")
    figure_sentiment_returns()
    figure_sentiment_drawdown()
    figure_sentiment_winners()
    
    print("\n[3/3] Comparative Figures")
    figure_equity_curves()
    # Average equity comparison across tickers
    figure_average_equity_comparison()
    table_overall_summary()
    figure_overall_performance_comparison()
    figure_overall_drawdown_comparison()
    figure_technical_vs_sentiment_winners()
    
    print("\n✓ All figures saved to: results/figures_company/")
    print("Outputs generated:")
    print("  Technical   > returns_technical.png, drawdown_technical.png, winners_technical.png")
    print("  Sentiment   > returns_sentiment.png, drawdown_sentiment.png, winners_sentiment.png")
    print("  Comparative > equity_curves.png, overall_summary.csv, overall_summary_table.png, overall_performance_comparison.png")


def figure_overall_drawdown_comparison():
    """Figure: Overall Drawdown Comparison - Mean drawdown of ALL models"""
    print("  > Overall Drawdown Comparison")
    
    results = parse_backtest_results()
    if results is None:
        return
    
    models = ["LR", "RF", "XGB", "LSTM"]
    
    all_tech_dd = []
    all_sent_dd = []
    bh_dd = []
    
    for ticker in TICKERS:
        # Collect ALL technical drawdowns (4 models per ticker)
        for model in models:
            all_tech_dd.append(results["tech"][model][ticker].get("drawdown", 0))
        
        # Collect ALL sentiment drawdowns (4 models per ticker)
        for model in models:
            all_sent_dd.append(results["sent"][model][ticker].get("drawdown", 0))
        
        # Buy & Hold drawdown (once per ticker)
        bh_dd.append(results["buyhold"][ticker].get("drawdown", 0))
    
    # Compute means over all models
    mean_tech_dd = np.mean(all_tech_dd)  # 6 tickers × 4 models = 24 values
    mean_sent_dd = np.mean(all_sent_dd)  # 6 tickers × 4 models = 24 values
    mean_bh_dd = np.mean(bh_dd)          # 6 tickers = 6 values
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 7))
    
    categories = ["All Technical Models", "All Sentiment Models", "Buy & Hold"]
    values = [mean_tech_dd, mean_sent_dd, mean_bh_dd]
    colors = [TYPE_COLORS["Technical"], TYPE_COLORS["Sentiment"], TYPE_COLORS["Buy & Hold"]]
    
    bars = ax.bar(categories, values, color=colors, alpha=0.85, edgecolor="black", linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='top' if height < 0 else 'bottom',
                fontsize=12, fontweight='bold')
    
    ax.set_ylabel("Mean Max Drawdown (%)", fontsize=13, fontweight="bold")
    ax.set_title("Overall Drawdown Comparison: All Models vs Buy & Hold", fontsize=15, fontweight="bold")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT / "overall_drawdown_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def figure_best_overall_model():
    """For each ticker, show the best model (sentiment, technical, or Buy & Hold) and its type (colored bar chart)."""
    print("  > Best Overall Model per Company (Tech vs Sentiment vs Buy & Hold)")
    results = parse_backtest_results()
    if results is None:
        return
    models = ["LR", "RF", "XGB", "LSTM"]
    best_labels = []
    best_returns = []
    best_types = []
    tickers = []
    for ticker in TICKERS:
        tech_ret = {m: results["tech"][m][ticker].get("return", 0) for m in models}
        sent_ret = {m: results["sent"][m][ticker].get("return", 0) for m in models}
        best_tech = max(tech_ret, key=tech_ret.get)
        best_sent = max(sent_ret, key=sent_ret.get)
        best_tech_val = tech_ret[best_tech]
        best_sent_val = sent_ret[best_sent]
        bh_val = results["buyhold"][ticker].get("return", 0)
        vals = {f"Technical ({best_tech})": best_tech_val, f"Sentiment ({best_sent})": best_sent_val, "Buy & Hold": bh_val}
        best_label = max(vals, key=vals.get)
        best_return = vals[best_label]
        if "Technical" in best_label:
            best_type = "Technical"
        elif "Sentiment" in best_label:
            best_type = "Sentiment"
        else:
            best_type = "Buy & Hold"
        best_labels.append(best_label)
        best_returns.append(best_return)
        best_types.append(best_type)
        tickers.append(ticker)
    # Bar chart: best model per company
    x = np.arange(len(tickers))
    color_map = TYPE_COLORS
    colors = [color_map[t] for t in best_types]
    fig, ax = plt.subplots(figsize=(16, 7))
    bars = ax.bar(x, best_returns, color=colors, edgecolor="black", alpha=0.85)
    ax.set_xlabel("Company", fontsize=13, fontweight="bold")
    ax.set_ylabel("Best Return (%)", fontsize=13, fontweight="bold")
    ax.set_title("Best Model per Company (Technical, Sentiment, or Buy & Hold)", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, fontsize=12)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.grid(axis="y", alpha=0.3)
    # Add value labels and model type
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{best_labels[i]}\n{height:.1f}%",
                ha="center", va="bottom" if height >= 0 else "top", fontsize=10, fontweight="bold")
    # Legend for model type
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[t], edgecolor='black', label=t) for t in color_map]
    ax.legend(handles=legend_elements, fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "best_overall_model.png", dpi=300, bbox_inches="tight")
    plt.close()
    # Pie chart: distribution of best type
    fig, ax = plt.subplots(figsize=(7, 7))
    from collections import Counter
    counts = Counter(best_types)
    wedges, texts, autotexts = ax.pie([counts.get("Technical",0), counts.get("Sentiment",0), counts.get("Buy & Hold",0)],
                                      labels=["Technical", "Sentiment", "Buy & Hold"],
                                      autopct=lambda pct: f'{pct:.0f}%' if pct > 0 else '',
                                      startangle=90, colors=[color_map[t] for t in color_map],
                                      textprops={'fontsize': 13, 'fontweight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('white')
    ax.set_title("Best Model Type Distribution (per company)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "best_overall_model_pie.png", dpi=300, bbox_inches="tight")
    plt.close()


def figure_technical_vs_sentiment_winners():
    """Figure: Compare Best Technical vs Best Sentiment for each company"""
    print("  > Technical vs Sentiment Winners")
    
    results = parse_backtest_results()
    if results is None:
        return
    
    models = ["LR", "RF", "XGB", "LSTM"]
    
    # For each ticker, get best technical and best sentiment
    best_tech_returns = []
    best_sent_returns = []
    best_tech_models = []
    best_sent_models = []
    
    for ticker in TICKERS:
        # Best technical
        tech_ret = {m: results["tech"][m][ticker].get("return", 0) for m in models}
        best_tech = max(tech_ret, key=tech_ret.get)
        best_tech_returns.append(tech_ret[best_tech])
        best_tech_models.append(best_tech)
        
        # Best sentiment
        sent_ret = {m: results["sent"][m][ticker].get("return", 0) for m in models}
        best_sent = max(sent_ret, key=sent_ret.get)
        best_sent_returns.append(sent_ret[best_sent])
        best_sent_models.append(best_sent)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Bar chart comparing returns
    x = np.arange(len(TICKERS))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, best_tech_returns, width, label="Best Technical", 
                    color=TYPE_COLORS["Technical"], alpha=0.85, edgecolor="black")
    bars2 = ax1.bar(x + width/2, best_sent_returns, width, label="Best Sentiment", 
                    color=TYPE_COLORS["Sentiment"], alpha=0.85, edgecolor="black")
    
    ax1.set_xlabel("Company", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Best Return (%)", fontsize=13, fontweight="bold")
    ax1.set_title("Best Technical vs Best Sentiment per Company", fontsize=15, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(TICKERS, fontsize=12)
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Add value labels and model names
    for i, (bar, model) in enumerate(zip(bars1, best_tech_models)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f"{model}\n{height:.1f}%",
                ha="center", va="bottom" if height >= 0 else "top", fontsize=9, fontweight="bold")
    
    for i, (bar, model) in enumerate(zip(bars2, best_sent_models)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f"{model}\n{height:.1f}%",
                ha="center", va="bottom" if height >= 0 else "top", fontsize=9, fontweight="bold")
    
    # Right: Pie chart showing who wins more often
    tech_wins = sum(1 for i in range(len(TICKERS)) if best_tech_returns[i] > best_sent_returns[i])
    sent_wins = sum(1 for i in range(len(TICKERS)) if best_sent_returns[i] > best_tech_returns[i])
    ties = sum(1 for i in range(len(TICKERS)) if best_tech_returns[i] == best_sent_returns[i])
    
    if tech_wins + sent_wins + ties > 0:
        labels = []
        sizes = []
        colors = []
        
        if tech_wins > 0:
            labels.append(f"Technical Wins ({tech_wins})")
            sizes.append(tech_wins)
            colors.append(TYPE_COLORS["Technical"])
        
        if sent_wins > 0:
            labels.append(f"Sentiment Wins ({sent_wins})")
            sizes.append(sent_wins)
            colors.append(TYPE_COLORS["Sentiment"])
        
        if ties > 0:
            labels.append(f"Ties ({ties})")
            sizes.append(ties)
            colors.append("gray")
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels,
                autopct=lambda pct: f'{pct:.0f}%' if pct > 5 else '',
                startangle=90, colors=colors,
                textprops={'fontsize': 12, 'fontweight': 'bold'})
        for autotext in autotexts:
            autotext.set_color('white')
    
    ax2.set_title("Technical vs Sentiment Win Rate", fontsize=15, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(OUT / "winners_technical_vs_sentiment.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_figures_company()
