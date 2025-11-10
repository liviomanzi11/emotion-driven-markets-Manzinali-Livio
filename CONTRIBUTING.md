# Contributing to Emotion-Driven Markets

This document describes the development workflow, structure, and coding guidelines for the project **"Emotion-Driven Markets: Social Media Sentiment and NASDAQ Movements"**, conducted as part of the *Data Science and Advanced Programming (DSAP 2025)* course at HEC Lausanne.

Although this is an **individual academic project**, the following guidelines ensure clarity, reproducibility, and consistency throughout the codebase.

## 1. Development Workflow

### Repository Setup
To reproduce or extend this project:

```bash
git clone https://github.com/your-username/emotion-driven-markets.git
cd emotion-driven-markets
pip install -r requirements.txt
```

The recommended environment is **Python 3.10+** with **Visual Studio Code** as IDE.

### Project Structure
```text
emotion-driven-markets/
│
├── data/                # Raw and processed datasets
├── notebooks/           # Jupyter/VS Code notebooks for exploration and analysis
├── src/                 # Core scripts: preprocessing, sentiment analysis, correlations
├── reports/             # Generated figures and final report
├── tests/               # Basic validation tests
├── PROJECT_SPECIFICATION.md   # Project overview and objectives
├── CONTRIBUTING.md            # This document
└── README.md                  # Main documentation
```

## 2. Coding Guidelines

- **Language:** Python 3.10+  
- **Naming conventions:** Use clear, descriptive snake_case for variables and functions.  
- **Documentation:** Every function should include a concise docstring following the Google style.  
- **Imports:** Group standard, third-party, and local imports separately.  
- **Line length:** ≤ 100 characters recommended.  
- **Comments:** Explain reasoning, not just operations.

### Example
```python
def clean_tweets(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize tweet text.
    
    Removes URLs, mentions, hashtags, and special characters.
    Converts text to lowercase and trims extra spaces.
    
    Args:
        df: Raw tweet dataset.
    Returns:
        DataFrame with cleaned tweet text.
    """
    ...
```

## 3. Data Management

- Raw datasets are stored under `data/`.  
- Processed or intermediate files should be saved under `data/processed/` or `reports/`.  
- Do **not** commit large raw data files to GitHub; use `.gitignore` for protection.  
- All scripts should be deterministic and reproducible (no random seeds left uncontrolled).

---

## 4. Version Control and Reproducibility

- Use clear and atomic commit messages (e.g., `feat: add sentiment aggregation` or `fix: clean null timestamps`).  
- Keep the repository organized and avoid committing unnecessary files.  
- Periodically push commits to GitHub to maintain backup and version history.  
- Code execution should produce the same results when rerun on the same data.

## 5. Testing and Validation

Although this project does not require industrial-level test coverage, basic checks are encouraged:

- Verify data loading and cleaning functions behave as expected.  
- Confirm sentiment scoring with FinBERT returns valid labels.  
- Validate that data merging preserves date alignment.  

Example:
```python
def test_sentiment_labels():
    result = analyze_sentiment("The company is performing well.")
    assert result in ["positive", "neutral", "negative"]
```

## 6. Style and Quality Checks

Before committing code:

```bash
# Format and lint code
black src/
ruff src/

# Run quick validation
pytest tests/
```

Optional but recommended tools:  
- **Black** – automatic code formatting  
- **Ruff** – linting for code quality  
- **pytest** – lightweight testing

## 7. Acknowledgements

This repository is developed as part of the **MSc in Finance – Data Science and Advanced Programming (DSAP 2025)** course at **HEC Lausanne**, under the supervision of *Prof. Simon Scheidegger* and *Dr. Anna Smirnova*.

> **Note:**  
> These guidelines are designed for academic clarity and reproducibility rather than open-source collaboration.  
> All work is individual and evaluated within the context of the DSAP 2025 course.
