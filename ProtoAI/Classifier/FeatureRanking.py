import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def rank_from_pandas_dataframe( df: pd.DataFrame ) -> pd.DataFrame:

    # 1. Generate sample data
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, random_state=42)
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

    # 2. Initialize estimator and RFE wrapper
    estimator = LogisticRegression()
    # n_features_to_select=1 forces it to rank every single feature uniquely
    selector = RFE(estimator, n_features_to_select=1, step=1)
    selector.fit(X, y)

    # 3. Build ranking DataFrame
    ranking_df = pd.DataFrame({
        'Feature': feature_names,
        'Rank': selector.ranking_,
        'Selected': selector.support_
    }).sort_values(by='Rank')

    print(ranking_df)