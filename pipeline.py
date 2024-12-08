import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.functions import single_obj

class DataPipeline:
    def __init__(self, loan_path, payment_path):
        self.loan_path = loan_path
        self.payment_path = payment_path
        self.label_encoders = {}
        self.combined_df = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.model = None

    def load_and_merge_data(self):
        loan_df = pd.read_csv(self.loan_path)
        payments_df = pd.read_csv(self.payment_path)
        self.combined_df = pd.merge(loan_df, payments_df, how="inner", on="loanId")
        self.combined_df.drop(columns=["anon_ssn", "clarityFraudId", "paymentReturnCode"], inplace=True)
        del loan_df, payments_df  # Free memory

    def handle_missing_values(self):
        # Numeric columns
        self.numeric_columns = [
            col for col in self.combined_df.select_dtypes(include="number").columns
            if self.combined_df[col].isna().any()
        ]
        self.combined_df[self.numeric_columns] = self.combined_df[self.numeric_columns].fillna(
            self.combined_df[self.numeric_columns].mean()
        )

        # Categorical columns
        self.categorical_columns = [
            col for col in self.combined_df.select_dtypes(include="object").columns
            if self.combined_df[col].isna().any()
        ]
        self.combined_df[self.categorical_columns] = self.combined_df[self.categorical_columns].fillna("Unknown")

    def feature_engineering(self):
        self.combined_df["applicationDate"] = pd.to_datetime(self.combined_df["applicationDate"], errors="coerce")
        self.combined_df["paymentDate"] = pd.to_datetime(self.combined_df["paymentDate"], errors="coerce")
        self.combined_df["paymentDuration"] = (self.combined_df["paymentDate"] - self.combined_df["applicationDate"]).dt.days
        self.combined_df.drop(columns=["applicationDate", "paymentDate"], inplace=True)
        self.combined_df["paymentDuration"].fillna(self.combined_df["paymentDuration"].mean(), inplace=True)

    def remove_outliers(self):
        def handle_outliers(col):
            q1, q3 = col.quantile([0.25, 0.75])
            col = np.clip(col, q1, q3)
            return col

        for col in self.numeric_columns:
            if col not in ["hasCF", "isFunded", "nPaidOff"]:
                self.combined_df[col] = handle_outliers(self.combined_df[col])

    def encode_categorical_features(self):
        '''for col in self.categorical_columns:
            le = LabelEncoder()
            self.combined_df[col] = le.fit_transform(self.combined_df[col])
            self.label_encoders[col] = le
        self.combined_df.drop(columns=["loanId"], inplace=True)'''

        categorical_columns = self.combined_df.select_dtypes(include=['object']).columns
        label_encoders = {}
    
        for col in categorical_columns:
            le = LabelEncoder()
            self.combined_df[col] = le.fit_transform(self.combined_df[col].astype(str))  # Ensure consistent string type
            label_encoders[col] = le  # Save encoders for potential inverse transformations
        
        self.label_encoders = label_encoders  # Save encoders for use later if needed

    def balance_data(self):
        healthy_indices = self.combined_df[self.combined_df["hasCF"] == 1].index
        random_pos = np.random.choice(
            healthy_indices, size=len(self.combined_df[self.combined_df["hasCF"] == 0]), replace=False
        )
        healthy_samples = self.combined_df.loc[random_pos]
        self.combined_df = pd.concat(
            [self.combined_df[self.combined_df["hasCF"] == 0], healthy_samples], ignore_index=True
        )

    def split_data(self):
        X = self.combined_df.drop(columns=["hasCF"])
        Y = self.combined_df["hasCF"]
        return train_test_split(X, Y, test_size=0.2, stratify=Y)

    def optimize_hyperparameters(self, X, Y):
        def pso_objective_function(hyperparameters):
            num_leaves, max_depth, learning_rate, n_estimators = hyperparameters.T
            f1_scores = []
            for leaves, depth, lr, estimators in zip(num_leaves, max_depth, learning_rate, n_estimators):
                params = {
                    "num_leaves": int(leaves),
                    "max_depth": int(depth),
                    "learning_rate": lr,
                    "n_estimators": int(estimators),
                }
                model = lgb.LGBMClassifier(**params)
                x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                f1_scores.append(f1_score(y_test, y_pred, average="macro"))
            return -np.array(f1_scores)

        bounds = ([10, 3, 0.01, 50], [100, 20, 0.5, 300])
        optimizer = GlobalBestPSO(
            n_particles=10, dimensions=4, options={"c1": 1.5, "c2": 1.5, "w": 0.9}, bounds=bounds
        )
        best_cost, best_hyperparameters = optimizer.optimize(pso_objective_function, iters=30)
        return best_hyperparameters

    def train_final_model(self, X_train, y_train, best_hyperparameters):
        self.model = lgb.LGBMClassifier(
            num_leaves=int(best_hyperparameters[0]),
            max_depth=int(best_hyperparameters[1]),
            learning_rate=best_hyperparameters[2],
            n_estimators=int(best_hyperparameters[3]),
        )
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return f1_score(y_test, y_pred, average="macro")


# Initialize and run pipeline
pipeline = DataPipeline("data folder\data\loan.csv", "data folder\data\payment.csv")
pipeline.load_and_merge_data()
pipeline.handle_missing_values()
pipeline.feature_engineering()
pipeline.remove_outliers()
pipeline.encode_categorical_features()
pipeline.balance_data()

X_train, X_test, y_train, y_test = pipeline.split_data()
best_hyperparameters = pipeline.optimize_hyperparameters(X_train, y_train)
pipeline.train_final_model(X_train, y_train, best_hyperparameters)
f1_score = pipeline.evaluate_model(X_test, y_test)

print(f"Final Model F1 Score: {f1_score}")
