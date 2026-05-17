# Data Mining Techniques

## Setup

1. Clone this repository and open a terminal in the project root.
2. Create a virtual environment:
   - Windows: `python -m venv .venv`
   - macOS/Linux: `python3 -m venv .venv`
3. Activate the virtual environment:
   - Windows (Command Prompt): `.venv\Scripts\activate`
   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
   - macOS/Linux: `source .venv/bin/activate`
4. Install dependencies from `requirements.txt`:
   - Windows/macOS/Linux: `pip install -r requirements.txt`

## Run Scripts

From the project root, run the Assignment 2 pipeline:

- `python assignment2_pipeline.py`

This regenerates the EDA figures, metrics file, `submission_final.csv`, and `submission_mitigated.csv`.

## Optional

When finished, deactivate the virtual environment:

- `deactivate`
