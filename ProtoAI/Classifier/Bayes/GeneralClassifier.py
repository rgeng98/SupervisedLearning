from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal

from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from scipy.stats import chi2, poisson, expon
from typing import Dict


@dataclass
class FeatureConfig:
    continuous: Optional[list[str]]            = None
    discrete:   Optional[list[str]]            = None
    ignore:     list[str]                      = field(default_factory=list)
    continuous_threshold:      int             = 10
    max_discrete_cardinality:  int             = 50


class LikelihoodEstimator:
    """Abstract base — subclass to add new likelihood types."""

    def fit(self, values: pd.Series, **kwargs) -> "LikelihoodEstimator":
        raise NotImplementedError

    def log_likelihood(self, values: pd.Series) -> np.ndarray:
        """Return array of log P(x | this distribution) for each value."""
        raise NotImplementedError

    def summary(self) -> dict:
        """Human-readable description of fitted parameters."""
        raise NotImplementedError


class GaussianLikelihood(LikelihoodEstimator):

    def fit(self, values: pd.Series, **kwargs) -> "GaussianLikelihood":
        clean = values.dropna()
        self.mu_    = float(clean.mean())
        self.sigma_ = float(max(clean.std(ddof=1), 1e-6))
        self.fallback_ = self.mu_        # used to impute NaN at predict time
        return self

    def log_likelihood(self, values: pd.Series) -> np.ndarray:
        v = values.fillna(self.fallback_).values.astype(float)
        return norm.logpdf(v, self.mu_, self.sigma_)

    def summary(self) -> dict:
        return {"type": "gaussian", "mu": self.mu_, "sigma": self.sigma_}

class BernoulliLikelihood(LikelihoodEstimator):
    """P(x | class) = Bernoulli(p)"""

    def fit(self, values: pd.Series, **kwargs) -> "BernoulliLikelihood":
        clean = values.dropna()
        # Proportion of 1s (handle all-0 or all-1 gracefully)
        n = len(clean)
        self.p_ = float(clean.mean()) if n > 0 else 0.5
        self.p_ = max(min(self.p_, 1 - 1e-6), 1e-6)  # avoid log(0)
        self.fallback_ = 1 if self.p_ >= 0.5 else 0
        return self

    def log_likelihood(self, values: pd.Series) -> np.ndarray:
        v = values.fillna(self.fallback_).values.astype(float)
        # Use log(1-p) + x*log(p/(1-p)) for numerical stability
        return v * np.log(self.p_) + (1 - v) * np.log(1 - self.p_)

    def summary(self) -> dict:
        return {"type": "bernoulli", "p": self.p_}

class ChiSquaredLikelihood(LikelihoodEstimator):
    def fit(self, values: pd.Series, **kwargs) -> "ChiSquaredLikelihood":
        clean = values.dropna()
        clean = clean[clean > 0]  # chi2 support is x > 0
        
        if len(clean) == 0:
            self.df_ = 1.0
        else:
            # Method of moments: df ≈ mean (for chi2)
            self.df_ = float(max(clean.mean(), 0.1))
        
        self.fallback_ = self.df_  # reasonable fallback
        return self

    def log_likelihood(self, values: pd.Series) -> np.ndarray:
        v = values.fillna(self.fallback_).clip(lower=1e-8).values.astype(float)
        return chi2.logpdf(v, df=self.df_)

    def summary(self) -> dict:
        return {"type": "chisquared", "df": self.df_}
    
class PoissonLikelihood(LikelihoodEstimator):
    """P(x | class) = Poisson(λ)   (excellent for count / frequency features)"""

    def fit(self, values: pd.Series, **kwargs) -> "PoissonLikelihood":
        clean = values.dropna()
        self.lambda_ = float(max(clean.mean(), 1e-6))
        self.fallback_ = round(self.lambda_)
        return self

    def log_likelihood(self, values: pd.Series) -> np.ndarray:
        v = values.fillna(self.fallback_).values.astype(float)
        return poisson.logpmf(v.round().astype(int), mu=self.lambda_)

    def summary(self) -> dict:
        return {"type": "poisson", "lambda": self.lambda_}

class ExponentialLikelihood(LikelihoodEstimator):
    """P(x | class) = Exponential(λ)"""

    def fit(self, values: pd.Series, **kwargs) -> "ExponentialLikelihood":
        clean = values.dropna()
        clean = clean[clean > 0]
        if len(clean) == 0:
            self.scale_ = 1.0
        else:
            self.scale_ = float(clean.mean())  # scale = 1/λ
        self.fallback_ = self.scale_
        return self

    def log_likelihood(self, values: pd.Series) -> np.ndarray:
        v = values.fillna(self.fallback_).clip(lower=1e-8).values.astype(float)
        return expon.logpdf(v, scale=self.scale_)

    def summary(self) -> dict:
        return {"type": "exponential", "scale": self.scale_}


class CategoricalLikelihood(LikelihoodEstimator):
    """P(x | class) = Categorical(p)   (for discrete categories)"""

    def fit(self, values: pd.Series, **kwargs) -> "CategoricalLikelihood":
        clean = values.dropna()
        # Empirical frequencies with Laplace smoothing
        counts = clean.value_counts()
        categories = counts.index.tolist()
        probs = (counts + 1) / (len(clean) + len(counts))  # Laplace
        
        self.categories_ = categories
        self.probs_ = dict(zip(categories, probs))
        self.unknown_prob_ = 1.0 / (len(counts) + 1)  # for unseen values
        self.fallback_ = clean.mode().iloc[0] if not clean.empty else categories[0]
        return self

    def log_likelihood(self, values: pd.Series) -> np.ndarray:
        def get_logp(x):
            x = x if pd.notna(x) else self.fallback_
            return np.log(self.probs_.get(x, self.unknown_prob_))
        
        return values.apply(get_logp).values

    def summary(self) -> dict:
        return {
            "type": "categorical",
            "n_categories": len(self.probs_),
            "probs": self.probs_
        }
    
class MultivariateGaussianLikelihood(LikelihoodEstimator):
    """Full covariance multivariate Gaussian — captures feature couplings."""

    def fit(self, values: pd.DataFrame, reg_param: float = 1e-6) -> "MultivariateGaussianLikelihood":
        clean = values.dropna()
        if len(clean) < 2:
            # Fallback
            self.mean_ = np.zeros(values.shape[1])
            self.cov_ = np.eye(values.shape[1]) * 1e-4
        else:
            self.mean_ = clean.mean().values
            self.cov_ = clean.cov().values
            # Regularization to prevent singular covariance
            self.cov_ += np.eye(self.cov_.shape[0]) * reg_param

        self.fallback_ = self.mean_
        self.reg_param = reg_param
        return self

    def log_likelihood(self, values: pd.DataFrame) -> np.ndarray:
        v = values.fillna(0).values  # simple imputation
        if v.ndim == 1:
            v = v.reshape(1, -1)
        return multivariate_normal.logpdf(v, mean=self.mean_, cov=self.cov_)

    def summary(self) -> dict:
        return {
            "type": "multivariate_gaussian",
            "n_features": len(self.mean_),
            "cov_shape": self.cov_.shape,
            "reg_param": self.reg_param
        }

likelihoods = {
    'gaussian': GaussianLikelihood,
    'bernoulli': BernoulliLikelihood,
    'poisson': PoissonLikelihood,
    'chisquared': ChiSquaredLikelihood,
    'exponential': ExponentialLikelihood,
    'categorical': CategoricalLikelihood,
}


class BayesianClassifier:
    def __init__(
        self,
        target:                 str,
        feature_distributions:  Dict[str, str],
        config:        Optional[FeatureConfig] = None,
        laplace_alpha:          float = 1.0,
        threshold:              float = 0.5,
        use_multivariate:        bool = False
    ):
        self.use_multivariate      = use_multivariate
        self.target                = target
        self.config                = config or FeatureConfig()
        self.feature_distributions = feature_distributions or {}
        self.laplace_alpha         = laplace_alpha
        self.threshold             = threshold

        # Populated during fit()
        self.classes_:            list[Any]              = []
        self.log_prior_:          dict[Any, float]       = {}
        self.continuous_features_: list[str]             = []
        self.discrete_features_:   list[str]             = []

        # Nested: estimators_[feature][class] = LikelihoodEstimator
        self.estimators_: dict[str, dict[Any, LikelihoodEstimator]] = {}

        self._is_fitted = False


    def fit(self, df: pd.DataFrame) -> "BayesianClassifier":
        self._validate_target(df)
        y = df[self.target]
        self.classes_ = sorted(y.unique())

        # Resolve feature lists
        self.continuous_features_, self.discrete_features_ = \
            self._resolve_features(df)

        # Global vocabulary per discrete feature (shared across classes)
        vocabs = {
            feat: set(df[feat].dropna().unique())
            for feat in self.discrete_features_
        }

        
        # Fit per class
        for c in self.classes_:
            subset = df[df[self.target] == c]
            self.log_prior_[c] = np.log(len(subset) / len(df))
            if self.use_multivariate and self.continuous_features_:
                cont_df = subset[self.continuous_features_]
                # Fit one multivariate estimator for all continuous features
                mv_est = MultivariateGaussianLikelihood().fit(cont_df)
                self.estimators_['__multivariate_continuous__'] = {c: mv_est for c in self.classes_}  # store once
            else:
                for feat in self.continuous_features_:
                    dist_type = self.feature_distributions.get(feat, None)
                    estimator = self._get_likelihood_estimator(
                        feat, dist_type, subset[feat], vocabs.get(feat)
                    )
                    # est = GaussianLikelihood().fit(subset[feat])
                    self.estimators_.setdefault(feat, {})[c] = estimator

            for feat in self.discrete_features_:
                dist_type = self.feature_distributions.get(feat, None)
                estimator = self._get_likelihood_estimator(
                    feat, dist_type, subset[feat], vocabs.get(feat)
                )
                self.estimators_.setdefault(feat, {})[c] = estimator

        self._is_fitted = True
        self._print_fit_summary()
        return self

    def _get_likelihood_estimator(
        self, 
        feature: str, 
        dist_type: Optional[str], 
        values: pd.Series,
        vocabulary: Optional[set] = None
    ) -> "LikelihoodEstimator":
        """Factory to create the appropriate likelihood estimator."""

        if self.use_multivariate and feature in self.continuous_features_ and isinstance(values, pd.DataFrame):
            # Special handling for joint continuous
            return MultivariateGaussianLikelihood().fit(values)
        
        if dist_type is None:
            # Default logic
            if feature in self.discrete_features_:
                dist_type = "categorical"
            else:
                dist_type = "gaussian"

        dist_type = dist_type.lower()
        
        if dist_type == "gaussian":
            return GaussianLikelihood().fit(values)
        elif dist_type == "bernoulli":
            return BernoulliLikelihood().fit(values)
        elif dist_type == "chisquared":
            return ChiSquaredLikelihood().fit(values)
        elif dist_type == "poisson":
            return PoissonLikelihood().fit(values)
        elif dist_type == "exponential":
            return ExponentialLikelihood().fit(values)
        elif dist_type == "categorical":
            return CategoricalLikelihood().fit(
                values,
                alpha=self.laplace_alpha,
                vocabulary=vocabulary
            )
        else:
            warnings.warn(
                f"Unknown distribution '{dist_type}' for feature '{feature}'. "
                f"Falling back to Gaussian.", UserWarning
            )
            return GaussianLikelihood().fit(values)
        
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        N = len(df)
        log_ps = np.zeros((N, len(self.classes_)))

        for ci, c in enumerate(self.classes_):
            lp = np.full(N, self.log_prior_[c])

            # Multivariate continuous block
            if self.use_multivariate and '__multivariate_continuous__' in self.estimators_:
                cont_df = df[self.continuous_features_]
                mv_est = self.estimators_['__multivariate_continuous__'][c]
                lp += mv_est.log_likelihood(cont_df)
            else:
                # Original per-feature continuous
                for feat in self.continuous_features_:
                    if feat in self.estimators_:
                        lp += self.estimators_[feat][c].log_likelihood(df[feat])

            # Discrete features
            for feat in self.discrete_features_:
                if feat in self.estimators_:
                    lp += self.estimators_[feat][c].log_likelihood(df[feat])

            log_ps[:, ci] = lp

        # Log-sum-exp normalization
        log_ps -= log_ps.max(axis=1, keepdims=True)
        ps = np.exp(log_ps)
        ps /= ps.sum(axis=1, keepdims=True)
        return ps

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(df)
        if len(self.classes_) == 2:
            return np.where(proba[:, 1] >= self.threshold,
                            self.classes_[1], self.classes_[0])
        return np.array([self.classes_[i] for i in proba.argmax(axis=1)])

    def evaluate(self, df: pd.DataFrame, split: str = "Data") -> dict:
        self._check_fitted()
        y_true = df[self.target].values
        proba  = self.predict_proba(df)
        y_pred = self.predict(df)

        acc = accuracy_score(y_true, y_pred)
        cm  = confusion_matrix(y_true, y_pred, labels=self.classes_)

        print(f"\n{'='*55}")
        print(f"  {split}")
        print(f"{'='*55}")
        print(f"  Accuracy : {acc:.4f}")

        if len(self.classes_) == 2:
            auc = roc_auc_score(y_true, proba[:, 1])
            print(f"  ROC-AUC  : {auc:.4f}")
        else:
            auc = None

        print(f"\n  Confusion Matrix (rows=true, cols=pred):")
        header = "        " + "  ".join(f"{c!s:>6}" for c in self.classes_)
        print(header)
        for i, c in enumerate(self.classes_):
            row = "  ".join(f"{cm[i,j]:>6}" for j in range(len(self.classes_)))
            print(f"  {c!s:>5} | {row}")

        print(f"\n{classification_report(y_true, y_pred)}")

        return {
            "split":  split,
            "acc":    acc,
            "auc":    auc,
            "y_true": y_true,
            "y_prob": proba[:, -1],
            "y_pred": y_pred,
        }

    def cross_validate(self, df: pd.DataFrame,
                       n_splits: int = 5,
                       feature_config: Dict[str, str] = {
                            "age":        "gaussian",
                            "Fare":       "gaussian",
                            "FamilySize": "gaussian",
                            "Pclass":     "categorical", 
                            "Sex":        "categorical", 
                            "Embarked":   "categorical", 
                            "Title":      "categorical",
                            "IsAlone":    "categorical", 
                            "HasCabin":   "categorical", 
                            "SibSp":      "categorical", 
                            "Parch":      "categorical"
                        }) -> dict[str, float]:
        """Stratified k-fold CV. Returns mean accuracy and AUC."""
        skf   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accs, aucs = [], []
        for fold, (tr_idx, val_idx) in enumerate(
                skf.split(df, df[self.target]), 1):
            
            m = BayesianClassifier(
                target=self.target,
                feature_distributions=feature_config,
                config=self.config,
                laplace_alpha=self.laplace_alpha,
                threshold=self.threshold,
            ).fit(df.iloc[tr_idx])
            r = m.evaluate(df.iloc[val_idx], split=f"Fold {fold}")
            accs.append(r["acc"])
            if r["auc"] is not None:
                aucs.append(r["auc"])

        print(f"\n  CV Mean Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        if aucs:
            print(f"  CV Mean ROC-AUC  : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        return {"acc": float(np.mean(accs)),
                "auc": float(np.mean(aucs)) if aucs else None}

    def explain(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Per-row, per-feature log-odds contribution (binary classification only).

        log-odds(feature, row) = log P(feat | class=1) - log P(feat | class=0)

        Positive  → pushes toward the positive class.
        Negative  → pushes toward the negative class.

        Prints a formatted breakdown for each row and returns a DataFrame.
        """
        if not self.use_multivariate:
            self._check_fitted()
            if len(self.classes_) != 2:
                raise ValueError("explain() is only supported for binary classification.")

            c0, c1 = self.classes_[0], self.classes_[1]
            rows   = []

            for idx, row in df.iterrows():
                entry = {}
                for feat, class_ests in self.estimators_.items():
                    ll0 = float(class_ests[c0].log_likelihood(pd.Series([row[feat]]))[0])
                    ll1 = float(class_ests[c1].log_likelihood(pd.Series([row[feat]]))[0])
                    entry[feat] = ll1 - ll0
                rows.append(entry)

            lo_df = pd.DataFrame(rows, index=df.index)

            # Print summary for each row
            proba = self.predict_proba(df)
            for i, (idx, row_lo) in enumerate(lo_df.iterrows()):
                prob = proba[i, 1]
                print(f"\n  ── Row index: {idx}  │  "
                    f"P({self.classes_[1]}) = {prob:.4f}  │  "
                    f"Prediction: {self.classes_[1] if prob >= self.threshold else self.classes_[0]}")
                print(f"  {'Feature':<22} {'Log-odds':>9}  Direction")
                print(f"  {'-'*50}")
                for feat, lo in row_lo.sort_values(ascending=False).items():
                    bar  = "█" * min(int(abs(lo) * 5), 20)
                    sign = "▲ toward " + str(self.classes_[1]) if lo >= 0 \
                        else "▼ toward " + str(self.classes_[0])
                    print(f"  {feat:<22} {lo:>+9.3f}  {sign}  {bar}")

            return lo_df

    def feature_summary(self) -> pd.DataFrame:
        self._check_fitted()
        records = []
        
        for feat_key, class_ests in self.estimators_.items():
            if feat_key == '__multivariate_continuous__':
                for c, est in class_ests.items():
                    rec = {
                        "feature": "MULTIVARIATE_CONTINUOUS",
                        "class": c,
                        "distribution": "multivariate_gaussian"
                    }
                    rec.update(est.summary())
                    records.append(rec)
                continue
                
            for c, est in class_ests.items():
                rec = {
                    "feature": feat_key, 
                    "class": c, 
                    "distribution": est.summary().get("type","unknown")
                }
                rec.update(est.summary())
                records.append(rec)
                
        return pd.DataFrame(records)

    def plot_likelihoods(self) -> None:
        """Plot Gaussian curves and categorical bars for all fitted features."""
        self._check_fitted()
        self._plot_likelihoods()

    def plot_roc(self, datasets: list[dict]) -> None:
        """
        datasets: list of dicts with keys 'y_true', 'y_prob', 'split'.
        Typically the dicts returned by evaluate().
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        for d in datasets:
            if d.get("auc") is None:
                continue
            fpr, tpr, _ = roc_curve(d["y_true"], d["y_prob"])
            ax.plot(fpr, tpr, lw=2,
                    label=f"{d['split']}  AUC={d['auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight")
        plt.show()

    def plot_probability_distribution(self, df: pd.DataFrame,
                                       split: str = "Data") -> None:
        proba  = self.predict_proba(df)[:, 1]
        y_true = df[self.target].values
        c0, c1 = self.classes_[0], self.classes_[1]

        fig, ax = plt.subplots(figsize=(7, 4))
        for cls, color in [(c0, "#4C72B0"), (c1, "#DD8452")]:
            mask = y_true == cls
            ax.hist(proba[mask], bins=40, alpha=0.6, color=color,
                    label=str(cls), density=True, edgecolor="none")
        ax.axvline(self.threshold, color="red", linestyle="--",
                   lw=1.5, label=f"Threshold {self.threshold}")
        ax.set_xlabel(f"P({c1} | features)")
        ax.set_ylabel("Density")
        ax.set_title(f"Predicted Probability Distribution — {split}")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"prob_dist_{split.lower().replace(' ','_')}.png",
                    dpi=150, bbox_inches="tight")
        plt.show()

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BayesianClassifier":
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"  Model loaded from {path}")
        return model

    def _validate_target(self, df: pd.DataFrame) -> None:
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found. "
                             f"Columns: {list(df.columns)}")

    def _resolve_features(self, df: pd.DataFrame
                          ) -> tuple[list[str], list[str]]:
        """
        Return (continuous_features, discrete_features).
        Uses explicit config if provided; otherwise auto-detects.
        """
        cfg         = self.config
        all_cols    = set(df.columns) - {self.target} - set(cfg.ignore)

        if cfg.continuous is not None and cfg.discrete is not None:
            # Fully explicit — just validate
            cont = [c for c in cfg.continuous if c in df.columns]
            disc = [c for c in cfg.discrete   if c in df.columns]
        elif cfg.continuous is not None:
            # Continuous explicit, infer discrete from remainder
            cont = [c for c in cfg.continuous if c in df.columns]
            disc = [c for c in all_cols - set(cont)
                    if c in df.columns]
        elif cfg.discrete is not None:
            # Discrete explicit, infer continuous from remainder
            disc = [c for c in cfg.discrete if c in df.columns]
            cont = [c for c in all_cols - set(disc)
                    if c in df.columns]
        else:
            # Fully auto-detect
            cont, disc = [], []
            for col in all_cols:
                series = df[col]
                if pd.api.types.is_numeric_dtype(series):
                    n_unique = series.nunique()
                    if n_unique >= cfg.continuous_threshold:
                        cont.append(col)
                    else:
                        disc.append(col)
                else:
                    disc.append(col)

        # Drop high-cardinality discrete columns
        safe_disc = []
        for col in disc:
            n = df[col].nunique()
            if n > cfg.max_discrete_cardinality:
                warnings.warn(
                    f"Column '{col}' has {n} unique values "
                    f"(> max_discrete_cardinality={cfg.max_discrete_cardinality}). "
                    f"Dropping. Add it to config.ignore to silence this.",
                    UserWarning, stacklevel=3,
                )
            else:
                safe_disc.append(col)

        print(f"  Continuous features : {cont}")
        print(f"  Discrete features   : {safe_disc}")
        return cont, safe_disc

    def _print_fit_summary(self) -> None:
        print(f"\n  Fitted BayesianClassifier")
        print(f"  Target   : {self.target}")
        print(f"  Classes  : {self.classes_}")
        print(f"  Priors   : { {c: round(np.exp(v),4) for c,v in self.log_prior_.items()} }")

    def _plot_likelihoods(self) -> None:
        """Plot likelihood distributions for all features (now supports multivariate)."""
        self._check_fitted()
        
        # Collect all features to plot
        all_feats = self.discrete_features_ + self.continuous_features_
        
        if self.use_multivariate and self.continuous_features_:
            all_feats = [f for f in all_feats if f not in self.continuous_features_]
            all_feats.append('__multivariate_continuous__')

        if not all_feats:
            print("No features to plot.")
            return

        n_cols = min(3, len(all_feats))
        n_rows = (len(all_feats) + n_cols - 1) // n_cols
        
        cmap = plt.get_cmap("tab10")
        colors = {c: cmap(i / len(self.classes_)) 
                for i, c in enumerate(self.classes_)}

        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten()
        fig.suptitle("Likelihood Distributions  P(feature(s) | class)", 
                    fontsize=14, fontweight="bold")

        for ax_idx, feat in enumerate(all_feats):
            ax = axes[ax_idx]
            
            if feat == '__multivariate_continuous__':
                ax.set_title("Multivariate Continuous\n(full covariance)", 
                           fontsize=11, fontweight="bold")
                ax.text(0.5, 0.5, "Multivariate Gaussian\n(covariance shown in summary)",
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xlabel("Multiple continuous features")
                continue

            # Original per-feature plotting logic
            est_example = self.estimators_[feat][self.classes_[0]]
            dist_type = est_example.summary().get("type", "unknown")

            ax.set_title(f"{feat}\n({dist_type})", fontsize=11, fontweight="bold")
            ax.set_xlabel(feat)
            ax.grid(alpha=0.3)

            for c in self.classes_:
                est = self.estimators_[feat][c]
                color = colors[c]
                label = str(c)

                # ... (keep all your existing plotting code for gaussian, categorical, etc.)
                if dist_type == "gaussian":
                    mu, sigma = est.mu_, est.sigma_
                    xs = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
                    ys = norm.pdf(xs, mu, sigma)
                    ax.plot(xs, ys, lw=2.5, color=color, label=label)
                    ax.axvline(mu, lw=1.2, linestyle="--", color=color, alpha=0.6)
                    ax.fill_between(xs, ys, alpha=0.12, color=color)

                elif dist_type in ("exponential", "chisquared"):
                    if dist_type == "exponential":
                        scale = est.scale_
                        xmax = scale * 6
                        xs = np.linspace(1e-6, xmax, 400)
                        ys = expon.pdf(xs, scale=scale)
                    else:
                        df_ = est.df_
                        xmax = chi2.ppf(0.995, df_)
                        xs = np.linspace(1e-6, xmax, 400)
                        ys = chi2.pdf(xs, df_)
                    ax.plot(xs, ys, lw=2.5, color=color, label=label)
                    ax.fill_between(xs, ys, alpha=0.12, color=color)

                elif dist_type == "poisson":
                    lam = est.lambda_
                    x_max = int(lam + 6 * np.sqrt(lam)) + 1
                    x_vals = np.arange(0, x_max)
                    pmf = poisson.pmf(x_vals, mu=lam)
                    ax.stem(x_vals, pmf, linefmt=color, markerfmt='o', 
                           basefmt=" ", label=label)

                elif dist_type == "bernoulli":
                    p = est.p_
                    x_vals = [0, 1]
                    probs = [1 - p, p]
                    ax.bar(x_vals, probs, width=0.6, alpha=0.8, color=color, 
                          label=label, edgecolor='black')

                elif dist_type == "categorical":
                    cat_est = self.estimators_[feat][self.classes_[0]]
                    categories = list(cat_est.probs_.keys())
                    x = np.arange(len(categories))
                    probs = [cat_est.probs_.get(v, cat_est.unknown_prob_) for v in categories]
                    
                    offset = (list(self.classes_).index(c) - (len(self.classes_)-1)/2) * 0.3
                    ax.bar(x + offset, probs, width=0.25, alpha=0.8, 
                          color=color, label=label, edgecolor='white')

            # Finalize subplot
            if dist_type in ("categorical", "bernoulli"):
                if dist_type == "categorical":
                    ax.set_xticks(x)
                    ax.set_xticklabels([str(v) for v in categories], 
                                     rotation=45, ha="right", fontsize=8)
                ax.set_ylabel("Probability")
            else:
                ax.set_ylabel("Density / PMF")

            ax.legend(title=self.target, fontsize=8)

        # Hide unused subplots
        for ax in axes[len(all_feats):]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.savefig("likelihood_distributions.png", dpi=160, bbox_inches="tight")
        plt.show()

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call .fit(df) first.")
