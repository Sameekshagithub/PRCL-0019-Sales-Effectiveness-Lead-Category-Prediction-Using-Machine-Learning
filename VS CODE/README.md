# PRCL-0019 — Sales Effectiveness (Lead Category Predictor)

A Flask web app that predicts whether a FicZon lead is **High Potential** or
**Low Potential**, based on a Random-Forest-family model trained on FicZon's
lead data.

## Project structure

```
PRCL-0019-Flask-App/
├── train_model.py         # connects to MySQL, cleans data, trains & saves the model
├── app.py                 # Flask web app that serves predictions
├── templates/
│   └── index.html         # the prediction form page
├── model_artifacts/        # created by train_model.py (model, scaler, etc.)
├── requirements.txt
└── README.md
```

## Setup (VS Code)

1. Open this folder in VS Code (`File > Open Folder...`).
2. Open a terminal in VS Code (`` Ctrl+` ``) and create a virtual environment:
   ```bash
   python -m venv venv
   ```
   Activate it:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Step 1 — Train the model

```bash
python train_model.py
```

You'll be prompted for the database password (from your project spec). This
script:
- connects to the MySQL database and loads the `data` table,
- cleans missing values,
- builds the `Lead_Category` target from `Status`,
- trains and compares five models plus a tuned Random Forest,
- saves the best model and all supporting files into `model_artifacts/`.

**Important:** open `train_model.py` and check the printed `Status` values
against the `status_to_category` dictionary near the top of the "build the
target" section — adjust it to match your real data before trusting the
results.

## Step 2 — Run the web app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. Fill in the form and
submit to see the predicted Lead Category.

## Notes

- `model_artifacts/` is regenerated every time you rerun `train_model.py` —
  rerun it if you update the data or the Status mapping.
- The database password is never hard-coded or saved to disk; it's requested
  securely at runtime via `getpass`.
- To stop the Flask server, press `Ctrl+C` in the terminal.
