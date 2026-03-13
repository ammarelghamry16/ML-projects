### Laptop Price Prediction – ML Project

---

### Overview

This project builds and compares several regression models to predict laptop prices based on their specifications.  
The main work is implemented in the notebook `laptop_price_ml.ipynb`.

The workflow includes:

- **Data loading and cleaning**
- **Feature engineering and encoding**
- **Baseline linear regression**
- **Polynomial regression with degree search**
- **Regularized models**: Ridge, Lasso, Elastic Net
- **Regression trees**
- **K-fold cross‑validation and model comparison**
- **Hyperparameter tuning** with `GridSearchCV` / `RandomizedSearchCV`
- **Learning curve / epoch-wise loss** using `SGDRegressor`

---

### Data

- **Training data**: tabular laptop dataset (loaded in the notebook)
- **Test data**: `test_data.csv`
- **Target variable**: `Price` (continuous)
- **Identifier**: `id`  
  - Kept in `df_test` for submission
  - Dropped from feature matrices (`X`, `X_train`, `X_val`, `X_test`)

Example engineered / processed features:

- `CPU_Frequency (GHz)`
- `RAM (GB)`
- `Weight (kg)`
- `ppi` (pixels per inch, derived)
- `HDD`, `SSD`
- One‑hot encoded columns for brand, CPU family, and operating system

---

### Environment & Dependencies

Recommended (Python 3.10+):

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

Key libraries:

- `pandas`, `numpy` – data manipulation
- `matplotlib`, `seaborn` – visualizations
- `scikit-learn`
  - Models: `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `DecisionTreeRegressor`, `SGDRegressor`
  - Tools: `train_test_split`, `KFold`, `cross_val_score`, `GridSearchCV`, `RandomizedSearchCV`,
    `PolynomialFeatures`, `StandardScaler`, `Pipeline`, `learning_curve`
  - Metrics: `r2_score`, `mean_squared_error`, `mean_absolute_error`

---

### Notebook Structure (`laptop_price_ml.ipynb`)

1. **Imports & Data Loading**
2. **EDA and Preprocessing**
   - Handling missing values
   - Creating derived features (e.g., `ppi`)
   - One‑hot encoding categorical variables
3. **Train / Validation Split**
   - `X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)`
4. **Baseline Linear Regression**
   - Fit on `X_train`, evaluate on `X_val`
   - Compute R², MSE, MAE for train and validation
5. **Polynomial Regression**
   - Loop over polynomial degrees (e.g., 2–5)
   - Optionally evaluate with K‑fold cross‑validation
6. **Regularized Models**
   - Ridge, Lasso, Elastic Net
   - Each evaluated with **5-fold cross‑validation** (R², MSE, MAE)
7. **Tree‑based Regression**
   - Regression trees with K‑fold evaluation
8. **Model Selection & Fine‑Tuning**
   - Use `RandomizedSearchCV` / `GridSearchCV` on the best family (e.g., Ridge or Lasso)
   - Models wrapped in a `Pipeline(StandardScaler() -> Model)` to ensure proper scaling
9. **Final Model & Submission**
   - Refit `best_model` on all `X, y`
   - Build `X_test` aligned with `X.columns`
   - Predict on `X_test`
   - Attach predictions to `df_test` and save a submission CSV (e.g., `submission_full.csv`)
10. **Learning Curve / Epoch‑wise Loss**
    - `Pipeline(StandardScaler() -> SGDRegressor)` trained for multiple epochs
    - Plot **train** and **validation** MSE vs **epoch** to visualize convergence and generalization

---

### Running the Project

1. Open the project in your IDE / Jupyter environment.
2. Ensure the training and test CSV files are in the project directory.
3. Open `laptop_price_ml.ipynb`.
4. Run cells from top to bottom:
   - Preprocessing → Model training → Evaluation → Fine‑tuning → Final predictions.
5. Locate the submission cell:
   - It creates a CSV with `id` and the predicted price column ready for upload.

---

### Evaluation Metrics

For each model, the notebook typically reports:

- **R² (coefficient of determination)** – goodness of fit
- **MSE (Mean Squared Error)** – squared loss
- **MAE (Mean Absolute Error)** – absolute loss

Metrics are calculated for:

- Train / validation split
- K‑fold cross‑validation (mean and per‑fold scores)

---

### Learning Curve Requirement

To meet the project requirement of plotting the loss function over epochs:

- A ridge‑like model is implemented via `SGDRegressor` with:
  - `loss="squared_error"`, `penalty="l2"`
  - `max_iter=1` and `warm_start=True`
- In a loop over epochs:
  - Fit one epoch at a time
  - Compute train and validation MSE each epoch
  - Plot both curves against epoch count

This shows how the model’s training and validation loss evolve as learning proceeds.

---

### Possible Extensions

- Add more advanced models (Random Forests, Gradient Boosting, XGBoost).
-.Perform feature selection and outlier analysis for robustness.
- Package preprocessing and the final model into a reusable module or API.

