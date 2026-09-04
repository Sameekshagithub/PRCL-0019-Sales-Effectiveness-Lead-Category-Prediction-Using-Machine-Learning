# """

# PRCL-0019 : Sales Effectiveness — Model Training Script
# ---------------------------------------------------------
# Connects to the FicZon MySQL database, cleans the lead data, builds the
# Lead_Category (High Potential / Low Potential) target from Status, trains
# and compares several classification models, and saves the best model plus
# all supporting artifacts that app.py needs to serve predictions.

# Run this once (or whenever you want to retrain) from a terminal:

#     python train_model.py
# DM!$Team&27@9!20!
# It will prompt for the database password securely (it is never stored in
# this file or printed to the screen).
# """
import os
from getpass import getpass

import joblib
import pandas as pd
import mysql.connector

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ===============================================================
# 1. DATABASE CONNECTION
# ===============================================================

DB_HOST = "18.136.157.135"
DB_PORT = 3306
DB_NAME = "project_sales"
DB_USER = "dm_team2"

print("=" * 70)
print("SALES EFFECTIVENESS - MODEL TRAINING")
print("=" * 70)

print("\nConnecting to database...")

db_password = getpass("Enter database password: ")

try:
    connection = mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=db_password,
        database=DB_NAME
    )

    print("Database connected successfully!")

except mysql.connector.Error as e:
    print("\nERROR: Database connection failed!")
    print(e)
    raise SystemExit


# ===============================================================
# 2. LOAD DATA
# ===============================================================

try:
    df = pd.read_sql("SELECT * FROM data", connection)
    connection.close()

except Exception as e:
    print("\nERROR: Could not load data!")
    print(e)
    connection.close()
    raise SystemExit


print(f"\nLoaded {df.shape[0]} rows, {df.shape[1]} columns.")

print("\nColumns available:")
print(df.columns.tolist())


# ===============================================================
# 3. BASIC DATA CLEANING
# ===============================================================

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)

# Make sure expected columns exist
required_columns = [
    "Source",
    "Sales_Agent",
    "Location",
    "Mobile",
    "Product_ID",
    "Status",
    "Created",
    "EMAIL",
    "Delivery_Mode"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("\nERROR: Missing columns:")
    print(missing_columns)
    raise SystemExit


# Fill categorical columns
for col in ["Source", "Sales_Agent", "Location", "Delivery_Mode"]:
    df[col] = df[col].fillna("Unknown").astype(str).str.strip()

# Mobile
df["Mobile"] = df["Mobile"].fillna("Unknown")

# Product ID
df["Product_ID"] = pd.to_numeric(
    df["Product_ID"],
    errors="coerce"
)

df["Product_ID"] = df["Product_ID"].fillna(
    df["Product_ID"].median()
)

# Remove duplicate rows
before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

print(f"Duplicate rows removed: {before_duplicates - after_duplicates}")


# ===============================================================
# 4. CREATE TARGET VARIABLE
#    Status -> Lead_Category
# ===============================================================

print("\n" + "=" * 70)
print("CREATING LEAD CATEGORY")
print("=" * 70)

# Normalize Status values
df["Status"] = (
    df["Status"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

print("\nUnique Status values found:")

print(sorted(df["Status"].unique()))


# ---------------------------------------------------------------
# IMPORTANT
# Mapping based on the actual values in your database
# ---------------------------------------------------------------

status_to_category = {

    # HIGH POTENTIAL
    "converted": "High Potential",
    "potential": "High Potential",
    "in progress positive": "High Potential",
    "long term": "High Potential",

    # LOW POTENTIAL
    "open": "Low Potential",
    "not responding": "Low Potential",
    "just enquiry": "Low Potential",
    "junk lead": "Low Potential",
    "in progress negative": "Low Potential",
    "lost": "Low Potential"
}


# Apply mapping
df["Lead_Category"] = df["Status"].map(status_to_category)


# Check unmapped values
unmapped = df.loc[
    df["Lead_Category"].isna(),
    "Status"
].unique()


if len(unmapped) > 0:

    print("\nWARNING: Unmapped Status values found:")

    for status in unmapped:
        print("-", status)

    print("\nThese rows will be removed from training.")


# Remove rows where target is missing
df = df.dropna(
    subset=["Lead_Category"]
)


# Show target distribution
print("\nLead_Category distribution:")

print(
    df["Lead_Category"].value_counts()
)


# ---------------------------------------------------------------
# VERY IMPORTANT CHECK
# ---------------------------------------------------------------

if df["Lead_Category"].nunique() < 2:

    print("\n" + "=" * 70)
    print("ERROR: ONLY ONE TARGET CLASS FOUND")
    print("=" * 70)

    print(
        "\nThe model needs at least two Lead_Category classes."
    )

    print(
        "\nCurrent classes:"
    )

    print(
        df["Lead_Category"].unique()
    )

    raise SystemExit


print(
    f"\nNumber of target classes: "
    f"{df['Lead_Category'].nunique()}"
)


# ===============================================================
# 5. FEATURE ENGINEERING
# ===============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)


# Convert Created to datetime
df["Created"] = pd.to_datetime(
    df["Created"],
    errors="coerce",
    dayfirst=True
)


# Extract date features
df["Created_Year"] = df["Created"].dt.year
df["Created_Month"] = df["Created"].dt.month
df["Created_Day"] = df["Created"].dt.day


# Fill missing date values
df["Created_Year"] = df["Created_Year"].fillna(0)
df["Created_Month"] = df["Created_Month"].fillna(0)
df["Created_Day"] = df["Created_Day"].fillna(0)


# Mobile availability
df["Has_Mobile"] = df["Mobile"].apply(
    lambda x:
        0
        if pd.isna(x)
        or str(x).strip().lower() in ["", "unknown", "none", "nan"]
        else 1
)


# Email availability
df["Has_Email"] = df["EMAIL"].apply(
    lambda x:
        0
        if pd.isna(x)
        or str(x).strip().lower() in ["", "unknown", "none", "nan"]
        else 1
)


# ===============================================================
# 6. PREPARE MODEL DATA
# ===============================================================

print("\nPreparing model dataset...")


df_model = df.drop(
    columns=[
        "Created",
        "Mobile",
        "EMAIL",
        "Status"
    ],
    errors="ignore"
)


# ===============================================================
# 7. ENCODE TARGET
# ===============================================================

le = LabelEncoder()

df_model["Lead_Category_Encoded"] = (
    le.fit_transform(
        df_model["Lead_Category"]
    )
)


X = df_model.drop(
    columns=[
        "Lead_Category",
        "Lead_Category_Encoded"
    ]
)

y = df_model["Lead_Category_Encoded"]


print("\nTarget classes:")

for number, category in enumerate(le.classes_):
    print(f"{number} = {category}")


# ===============================================================
# 8. ENCODE CATEGORICAL FEATURES
# ===============================================================

categorical_columns = [
    "Source",
    "Sales_Agent",
    "Location",
    "Delivery_Mode"
]


# Only encode columns that actually exist
categorical_columns = [
    col
    for col in categorical_columns
    if col in X.columns
]


X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)


# ===============================================================
# 9. HANDLE MISSING VALUES
# ===============================================================

# Convert boolean columns to integers
bool_columns = X.select_dtypes(
    include=["bool"]
).columns

X[bool_columns] = X[bool_columns].astype(int)


# Convert everything to numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# Fill any remaining missing values
X = X.fillna(0)


print("\nFinal feature shape:")
print(X.shape)


print("\nNumber of features:")
print(len(X.columns))


# Save feature names
feature_columns = list(X.columns)


# ===============================================================
# 10. TRAIN / TEST SPLIT
# ===============================================================

print("\nSplitting data into training and testing sets...")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ===============================================================
# 11. STANDARD SCALING
# ===============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ===============================================================
# 12. MODEL TRAINING
# ===============================================================

print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)


results = {}

trained_models = {}


scaled_models = {
    "Logistic Regression",
    "KNN"
}


def evaluate_model(
    name,
    model,
    predictions
):

    results[name] = {

        "Accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "Precision":
            precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),

        "F1-Score":
            f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )
    }

    trained_models[name] = model

    print(
        f"\n{name}"
    )

    print(
        results[name]
    )


# ---------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------

print("\nTraining Logistic Regression...")

log_reg = LogisticRegression(
    max_iter=1000,
    random_state=42
)

log_reg.fit(
    X_train_scaled,
    y_train
)

log_predictions = log_reg.predict(
    X_test_scaled
)

evaluate_model(
    "Logistic Regression",
    log_reg,
    log_predictions
)


# ---------------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------------

print("\nTraining Decision Tree...")

dt_clf = DecisionTreeClassifier(
    random_state=42
)

dt_clf.fit(
    X_train,
    y_train
)

dt_predictions = dt_clf.predict(
    X_test
)

evaluate_model(
    "Decision Tree",
    dt_clf,
    dt_predictions
)


# ---------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------

print("\nTraining Random Forest...")

rf_clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_clf.fit(
    X_train,
    y_train
)

rf_predictions = rf_clf.predict(
    X_test
)

evaluate_model(
    "Random Forest",
    rf_clf,
    rf_predictions
)


# ---------------------------------------------------------------
# KNN
# ---------------------------------------------------------------

print("\nTraining KNN...")

# Make sure KNN has enough neighbors for the training set
n_neighbors = min(
    7,
    len(X_train)
)

knn_clf = KNeighborsClassifier(
    n_neighbors=n_neighbors
)

knn_clf.fit(
    X_train_scaled,
    y_train
)

knn_predictions = knn_clf.predict(
    X_test_scaled
)

evaluate_model(
    "KNN",
    knn_clf,
    knn_predictions
)


# ---------------------------------------------------------------
# Gradient Boosting
# ---------------------------------------------------------------

print("\nTraining Gradient Boosting...")

gb_clf = GradientBoostingClassifier(
    random_state=42
)

gb_clf.fit(
    X_train,
    y_train
)

gb_predictions = gb_clf.predict(
    X_test
)

evaluate_model(
    "Gradient Boosting",
    gb_clf,
    gb_predictions
)


# ===============================================================
# 13. RANDOM FOREST HYPERPARAMETER TUNING
# ===============================================================

print("\n" + "=" * 70)
print("TUNING RANDOM FOREST")
print("=" * 70)


param_grid = {

    "n_estimators": [
        100,
        200,
        300
    ],

    "max_depth": [
        None,
        8,
        12,
        16
    ],

    "min_samples_split": [
        2,
        5,
        10
    ]
}


grid_search = GridSearchCV(

    estimator=RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ),

    param_grid=param_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1
)


grid_search.fit(
    X_train,
    y_train
)


best_rf = grid_search.best_estimator_


print(
    "\nBest Random Forest parameters:"
)

print(
    grid_search.best_params_
)


best_rf_predictions = best_rf.predict(
    X_test
)


evaluate_model(
    "Random Forest (Tuned)",
    best_rf,
    best_rf_predictions
)


# ===============================================================
# 14. MODEL COMPARISON
# ===============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


results_df = (
    pd.DataFrame(results)
    .T
    .sort_values(
        "Accuracy",
        ascending=False
    )
)


print(
    results_df.round(4)
)


# ===============================================================
# 15. SELECT BEST MODEL
# ===============================================================

best_model_name = (
    results_df[
        "Accuracy"
    ].idxmax()
)


best_model = (
    trained_models[
        best_model_name
    ]
)


print(
    f"\nBest model: {best_model_name}"
)

print(
    f"Best Accuracy: "
    f"{results_df.loc[best_model_name, 'Accuracy']:.4f}"
)


# ===============================================================
# 16. SAVE MODEL ARTIFACTS
# ===============================================================

print("\n" + "=" * 70)
print("SAVING MODEL ARTIFACTS")
print("=" * 70)


artifact_folder = "model_artifacts"

os.makedirs(
    artifact_folder,
    exist_ok=True
)


# Model
joblib.dump(
    best_model,
    os.path.join(
        artifact_folder,
        "lead_model.pkl"
    )
)


# Scaler
joblib.dump(
    scaler,
    os.path.join(
        artifact_folder,
        "scaler.pkl"
    )
)


# Label Encoder
joblib.dump(
    le,
    os.path.join(
        artifact_folder,
        "label_encoder.pkl"
    )
)


# Feature columns
joblib.dump(
    feature_columns,
    os.path.join(
        artifact_folder,
        "feature_columns.pkl"
    )
)


# Whether model requires scaling
joblib.dump(
    best_model_name in scaled_models,
    os.path.join(
        artifact_folder,
        "needs_scaling.pkl"
    )
)


# ===============================================================
# 17. SAVE FORM OPTIONS
# ===============================================================

form_options = {

    "Source":
        sorted(
            df["Source"]
            .dropna()
            .unique()
            .tolist()
        ),

    "Sales_Agent":
        sorted(
            df["Sales_Agent"]
            .dropna()
            .unique()
            .tolist()
        ),

    "Location":
        sorted(
            df["Location"]
            .dropna()
            .unique()
            .tolist()
        ),

    "Delivery_Mode":
        sorted(
            df["Delivery_Mode"]
            .dropna()
            .unique()
            .tolist()
        )
}


joblib.dump(
    form_options,
    os.path.join(
        artifact_folder,
        "form_options.pkl"
    )
)


# ===============================================================
# 18. VERIFY ARTIFACTS
# ===============================================================

print("\nSaved artifacts:")

for filename in sorted(
    os.listdir(artifact_folder)
):

    print(
        "✓",
        filename
    )


# ===============================================================
# 19. FINAL MESSAGE
# ===============================================================

print("\n" + "=" * 70)

print(
    "MODEL TRAINING COMPLETED SUCCESSFULLY!"
)

print("=" * 70)

print(
    f"\nBest model: {best_model_name}"
)

print(
    f"Accuracy: "
    f"{results_df.loc[best_model_name, 'Accuracy']:.4f}"
)

print(
    "\nModel artifacts are ready."
)

print(
    "\nNext step:"
)

print(
    "python app.py"
)


# import os
# from getpass import getpass

# import joblib
# import pandas as pd
# import mysql.connector

# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # ---------------------------------------------------------------
# # 1. Connect to the database and load the data
# # ---------------------------------------------------------------
# DB_HOST = "18.136.157.135"
# DB_PORT = 3306
# DB_NAME = "project_sales"
# DB_USER = "dm_team2"

# print("Connecting to database...")
# db_password = getpass("Enter database password: ")

# connection = mysql.connector.connect(
#     host=DB_HOST,
#     port=DB_PORT,
#     user=DB_USER,
#     password=db_password,
#     database=DB_NAME,
# )
# print("Database connected successfully!")

# df = pd.read_sql("SELECT * FROM data", connection)
# connection.close()
# print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")

# # ---------------------------------------------------------------
# # 2. Clean the data
# # ---------------------------------------------------------------
# for col in ["Source", "Sales_Agent", "Location", "Mobile"]:
#     df[col] = df[col].fillna("Unknown")

# df["Product_ID"] = pd.to_numeric(df["Product_ID"], errors="coerce")
# df["Product_ID"] = df["Product_ID"].fillna(df["Product_ID"].median())

# df = df.drop_duplicates()

# # ---------------------------------------------------------------
# # 3. Build the target variable: Status -> Lead_Category
# # ---------------------------------------------------------------
# # IMPORTANT: check df['Status'].unique() against your real data and adjust
# # this mapping before trusting the trained model's predictions.
# print("\nUnique Status values found in the data:")
# print(df["Status"].unique())

# status_to_category = {
#     "Converted": "High Potential",
#     "Qualified": "High Potential",
#     "In Progress": "High Potential",
#     "Interested": "High Potential",
#     "Open": "Low Potential",
#     "Not Interested": "Low Potential",
#     "Junk Lead": "Low Potential",
#     "Lost": "Low Potential",
# }

# df["Lead_Category"] = df["Status"].map(status_to_category)

# unmapped = df[df["Lead_Category"].isnull()]["Status"].unique()
# if len(unmapped) > 0:
#     print("\nWARNING: these Status values are not covered by the mapping "
#           "and will be dropped from training:")
#     print(unmapped)

# df = df.dropna(subset=["Lead_Category"])
# print("\nLead_Category distribution:")
# print(df["Lead_Category"].value_counts())

# # ---------------------------------------------------------------
# # 4. Feature engineering
# # ---------------------------------------------------------------
# df["Created"] = pd.to_datetime(df["Created"], errors="coerce")
# df["Created_Year"] = df["Created"].dt.year
# df["Created_Month"] = df["Created"].dt.month
# df["Created_Day"] = df["Created"].dt.day

# df["Has_Mobile"] = df["Mobile"].apply(lambda x: 0 if x in ["Unknown", None] or pd.isnull(x) else 1)
# df["Has_Email"] = df["EMAIL"].notnull().astype(int)

# df_model = df.drop(columns=["Created", "Mobile", "EMAIL", "Status"])

# # ---------------------------------------------------------------
# # 5. Encode target and features
# # ---------------------------------------------------------------
# le = LabelEncoder()
# df_model["Lead_Category_Encoded"] = le.fit_transform(df_model["Lead_Category"])

# X = df_model.drop(columns=["Lead_Category", "Lead_Category_Encoded"])
# y = df_model["Lead_Category_Encoded"]

# X = pd.get_dummies(
#     X, columns=["Source", "Sales_Agent", "Location", "Delivery_Mode"], drop_first=True
# )

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # ---------------------------------------------------------------
# # 6. Train and compare models
# # ---------------------------------------------------------------
# results = {}
# trained_models = {}
# scaled_models = {"Logistic Regression", "KNN"}


# def evaluate(name, model, preds):
#     results[name] = {
#         "Accuracy": accuracy_score(y_test, preds),
#         "Precision": precision_score(y_test, preds, average="weighted"),
#         "Recall": recall_score(y_test, preds, average="weighted"),
#         "F1-Score": f1_score(y_test, preds, average="weighted"),
#     }
#     trained_models[name] = model
#     print(f"{name}: {results[name]}")


# print("\nTraining models...")

# log_reg = LogisticRegression(max_iter=1000, random_state=42)
# log_reg.fit(X_train_scaled, y_train)
# evaluate("Logistic Regression", log_reg, log_reg.predict(X_test_scaled))

# dt_clf = DecisionTreeClassifier(random_state=42)
# dt_clf.fit(X_train, y_train)
# evaluate("Decision Tree", dt_clf, dt_clf.predict(X_test))

# rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
# rf_clf.fit(X_train, y_train)
# evaluate("Random Forest", rf_clf, rf_clf.predict(X_test))

# knn_clf = KNeighborsClassifier(n_neighbors=7)
# knn_clf.fit(X_train_scaled, y_train)
# evaluate("KNN", knn_clf, knn_clf.predict(X_test_scaled))

# gb_clf = GradientBoostingClassifier(random_state=42)
# gb_clf.fit(X_train, y_train)
# evaluate("Gradient Boosting", gb_clf, gb_clf.predict(X_test))

# # ---------------------------------------------------------------
# # 7. Hyperparameter tuning on Random Forest
# # ---------------------------------------------------------------
# print("\nTuning Random Forest with GridSearchCV...")
# param_grid = {
#     "n_estimators": [100, 200, 300],
#     "max_depth": [None, 8, 12, 16],
#     "min_samples_split": [2, 5, 10],
# }

# grid_search = GridSearchCV(
#     RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="accuracy", n_jobs=-1
# )
# grid_search.fit(X_train, y_train)
# best_rf = grid_search.best_estimator_
# print("Best Random Forest params:", grid_search.best_params_)
# evaluate("Random Forest (Tuned)", best_rf, best_rf.predict(X_test))

# # ---------------------------------------------------------------
# # 8. Pick the best model
# # ---------------------------------------------------------------
# results_df = pd.DataFrame(results).T.sort_values("Accuracy", ascending=False)
# print("\nModel comparison:")
# print(results_df.round(4))

# best_model_name = results_df["Accuracy"].idxmax()
# best_model = trained_models[best_model_name]
# print(f"\nBest model: {best_model_name}")

# # ---------------------------------------------------------------
# # 9. Save all artifacts for the Flask app
# # ---------------------------------------------------------------
# os.makedirs("model_artifacts", exist_ok=True)

# joblib.dump(best_model, "model_artifacts/lead_model.pkl")
# joblib.dump(scaler, "model_artifacts/scaler.pkl")
# joblib.dump(le, "model_artifacts/label_encoder.pkl")
# joblib.dump(list(X.columns), "model_artifacts/feature_columns.pkl")
# joblib.dump(best_model_name in scaled_models, "model_artifacts/needs_scaling.pkl")

# form_options = {
#     "Source": sorted(df["Source"].unique().tolist()),
#     "Sales_Agent": sorted(df["Sales_Agent"].unique().tolist()),
#     "Location": sorted(df["Location"].unique().tolist()),
#     "Delivery_Mode": sorted(df["Delivery_Mode"].unique().tolist()),
# }
# joblib.dump(form_options, "model_artifacts/form_options.pkl")

# print("\nSaved artifacts to model_artifacts/:")
# print(os.listdir("model_artifacts"))
# print("\nDone. You can now run: python app.py")
