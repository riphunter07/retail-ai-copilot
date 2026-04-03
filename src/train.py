import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import joblib
import argparse
import os


# mlflow.set_tracking_uri("http://127.0.0.1:5000") needed if using local mlflow server not on azure
mlflow.set_experiment("retail-ai")

# Load data
parser = argparse.ArgumentParser()  
parser.add_argument("--data", type=str, default="data/sales.csv",required=True)
args = parser.parse_args()
df = pd.read_csv(args.data)

X = df[["store", "day_of_week", "promo"]]
y = df["sales"]

# Try multiple models
models = {
    "rf_100": RandomForestRegressor(n_estimators=100),
    "rf_200": RandomForestRegressor(n_estimators=200),
    "linear": LinearRegression()
}

# Ensure output directory exists for saving models on azure not needed for local
os.makedirs("outputs/models", exist_ok=True)   

for name, model in models.items():
    with mlflow.start_run(run_name=name):

        model.fit(X, y)
        preds = model.predict(X)

        rmse = mean_squared_error(y, preds)

        # Log metrics
        mlflow.log_metric("rmse", rmse)

        # Log params
        if hasattr(model, "n_estimators"):
            mlflow.log_param("n_estimators", model.n_estimators)

        # Log model
        mlflow.sklearn.log_model(model, name)

        print(f"{name} RMSE:", rmse)
       
        # Save model locally
        # joblib.dump(model, f"models/{name}.pkl")

        # Save model to outputs for Azure ML
        joblib.dump(model, f"outputs/models/{name}.pkl")

