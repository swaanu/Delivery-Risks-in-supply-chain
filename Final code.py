
import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# DISPLAY CONFIG
# --------------------------------------------------------------------------
SHOW_PLOTS = True

import matplotlib
if SHOW_PLOTS:
    _backend_found = False
    for _bk in ("TkAgg", "Qt5Agg", "Qt4Agg"):
        try:
            matplotlib.use(_bk)
            _backend_found = True
            break
        except Exception:
            continue
    if not _backend_found:
        print("No interactive display backend available — falling back to save-only mode.")
        SHOW_PLOTS = False
        matplotlib.use("Agg")
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    silhouette_samples, silhouette_score
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import PartialDependenceDisplay

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------
# 0. SETUP
# --------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "DataCoSupplyChainDataset.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
TABLES_DIR = os.path.join(OUT_DIR, "summary_tables")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

PLOT_COUNTER = [0]


def outp(filename):
    return os.path.join(OUT_DIR, filename)


def tabp(filename):
    return os.path.join(TABLES_DIR, filename)


if not os.path.exists(DATA_PATH):
    sys.exit(
        f"ERROR: could not find '{DATA_PATH}'.\n"
        "Put DataCoSupplyChainDataset.csv in the same folder as this script."
    )

sns.set_style("whitegrid")
PALETTE = ["#065A82", "#E07A5F", "#1C7293", "#5FB3A3", "#8E5572", "#3D405B", "#C44E52"]


def savefig(name, tight=True):
    PLOT_COUNTER[0] += 1
    fname = f"{PLOT_COUNTER[0]:02d}_{name}"
    if tight:
        plt.tight_layout()
    plt.savefig(outp(fname), dpi=150, bbox_inches="tight")
    print("  saved:", fname)
    if SHOW_PLOTS:
        plt.show(block=False)
        plt.pause(0.6)
    else:
        plt.close()


print("=" * 70)
print("STEP 1: Loading data")
print("=" * 70)
df_raw = pd.read_csv(DATA_PATH, encoding="latin1")
print("Raw shape:", df_raw.shape)

# --------------------------------------------------------------------------
# 1. CLEANING
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Cleaning & leakage removal")
print("=" * 70)

df = df_raw.copy()

diff_check = (df["Days for shipping (real)"] - df["Days for shipment (scheduled)"] > 0).astype(int)
agreement = (diff_check == df["Late_delivery_risk"]).mean()
print(f"Delivery Status / Days-for-shipping(real) agree with target in {agreement:.1%} of rows -> LEAKAGE, dropping.")

drop_cols = [
    "Delivery Status", "Days for shipping (real)",
    "Customer Email", "Customer Password", "Customer Fname", "Customer Lname",
    "Customer Street", "Customer Zipcode",
    "Product Description", "Product Image",
    "Order Zipcode",
    "Order Item Cardprod Id", "Product Card Id",
    "Order Id", "Order Item Id", "Order Customer Id", "Customer Id",
    "Order Item Total",
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

df["order date (DateOrders)"] = pd.to_datetime(df["order date (DateOrders)"])
df["Order Month"] = df["order date (DateOrders)"].dt.month
df["Order Weekday"] = df["order date (DateOrders)"].dt.dayofweek
df["Order Year"] = df["order date (DateOrders)"].dt.year
df["Order Hour"] = df["order date (DateOrders)"].dt.hour
df = df.drop(columns=["order date (DateOrders)", "shipping date (DateOrders)"])

print("Cleaned shape:", df.shape)
df.to_pickle(outp("df_clean_raw.pkl"))

# --------------------------------------------------------------------------
# 2. ENCODING
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Encoding categorical features")
print("=" * 70)

drop_more = ["Customer City", "Customer State", "Order City", "Order State",
             "Order Country", "Product Name", "Product Status", "Category Id",
             "Department Id", "Product Category Id"]
df_model = df.drop(columns=[c for c in drop_more if c in df.columns])

cat_cols = ["Type", "Category Name", "Customer Country", "Customer Segment",
            "Department Name", "Market", "Order Region", "Order Status", "Shipping Mode"]
df_enc = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)
print("Encoded shape:", df_enc.shape)

# ==========================================================================
# 3. DIMENSIONALITY REDUCTION & FEATURE CROSSES
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 3.5: Dimensionality Reduction (PCA)")
print("=" * 70)

X_pca_raw = df_enc.drop(columns=["Late_delivery_risk"]).select_dtypes(include=[np.number]).fillna(0)
scaler_pca = StandardScaler()
X_scaled_pca = scaler_pca.fit_transform(X_pca_raw)

pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X_scaled_pca)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--', color='#1C7293')
plt.title("PCA Explained Variance (Cumulative)")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.axhline(y=0.80, color='r', linestyle='-', label="80% Threshold")
plt.legend()
savefig("pca_scree_plot.png")

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_enc["Late_delivery_risk"], 
                      cmap="coolwarm", alpha=0.3, s=10)
plt.title("2D PCA Projection of Supply Chain Data")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.colorbar(scatter, label="Late Delivery Risk")
savefig("pca_2d_projection.png")

print("\n" + "=" * 70)
print("STEP 3.6: Generating Mathematical Interaction Features")
print("=" * 70)

poly_features = ["Days for shipment (scheduled)", "Sales", "Order Item Discount Rate", "Order Item Quantity"]
poly_subset = df_enc[poly_features].fillna(0)

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
interactions = poly.fit_transform(poly_subset)
feature_names = poly.get_feature_names_out(poly_features)

interaction_df = pd.DataFrame(interactions, columns=feature_names, index=df_enc.index)
interaction_df = interaction_df.drop(columns=poly_features)

df_enc = pd.concat([df_enc, interaction_df], axis=1)
print(f"Added {interaction_df.shape[1]} non-linear interaction features to the design matrix.")
df_enc.to_pickle(outp("df_encoded.pkl"))

# ==========================================================================
# 4. DEEP EXPLORATORY DATA ANALYSIS
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 4: Exploratory Data Analysis")
print("=" * 70)

overall_risk = df["Late_delivery_risk"].mean()
print(f"Overall late-delivery risk rate: {overall_risk:.1%}")

plt.figure(figsize=(5, 4))
sns.countplot(x="Late_delivery_risk", data=df, hue="Late_delivery_risk",
              palette=["#4C72B0", "#DD8452"], legend=False)
plt.title("Class Distribution: Late Delivery Risk")
savefig("class_balance.png")

plt.figure(figsize=(7, 4))
sns.barplot(x="Shipping Mode", y="Late_delivery_risk", data=df,
            hue="Shipping Mode", estimator=np.mean, palette="viridis", legend=False)
plt.axhline(overall_risk, color="gray", linestyle="--", linewidth=1, label="Overall mean")
plt.legend()
plt.title("Late Delivery Risk Rate by Shipping Mode")
savefig("risk_by_shipping_mode.png")

plt.figure(figsize=(10, 6))
region_risk = df.groupby("Order Region")["Late_delivery_risk"].mean().sort_values(ascending=False)
sns.barplot(x=region_risk.values, y=region_risk.index, hue=region_risk.index,
            palette="mako", legend=False)
plt.title("Late Delivery Risk Rate by Order Region")
savefig("risk_by_region.png")

plt.figure(figsize=(7, 4))
market_risk = df.groupby("Market")["Late_delivery_risk"].mean().sort_values(ascending=False)
sns.barplot(x=market_risk.index, y=market_risk.values, hue=market_risk.index,
            palette="crest", legend=False)
plt.title("Late Delivery Risk Rate by Market")
savefig("risk_by_market.png")

plt.figure(figsize=(6, 4))
seg_risk = df.groupby("Customer Segment")["Late_delivery_risk"].mean().sort_values(ascending=False)
sns.barplot(x=seg_risk.index, y=seg_risk.values, hue=seg_risk.index,
            palette="flare", legend=False)
plt.title("Late Delivery Risk Rate by Customer Segment")
savefig("risk_by_customer_segment.png")

plt.figure(figsize=(9, 5))
dept_risk = df.groupby("Department Name")["Late_delivery_risk"].mean().sort_values(ascending=False)
sns.barplot(x=dept_risk.values, y=dept_risk.index, hue=dept_risk.index,
            palette="rocket", legend=False)
plt.title("Late Delivery Risk Rate by Department")
savefig("risk_by_department.png")

plt.figure(figsize=(9, 6))
top_cats = df["Category Name"].value_counts().head(15).index
cat_risk = df[df["Category Name"].isin(top_cats)].groupby("Category Name")["Late_delivery_risk"].mean().sort_values(ascending=False)
sns.barplot(x=cat_risk.values, y=cat_risk.index, hue=cat_risk.index,
            palette="mako", legend=False)
plt.title("Late Delivery Risk Rate — Top 15 Categories")
savefig("risk_by_category_top15.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
status_counts = df["Order Status"].value_counts()
sns.barplot(x=status_counts.values, y=status_counts.index, ax=axes[0],
            hue=status_counts.index, palette="Blues_r", legend=False)
axes[0].set_title("Order Status — Volume")
status_risk = df.groupby("Order Status")["Late_delivery_risk"].mean().sort_values(ascending=False)
sns.barplot(x=status_risk.values, y=status_risk.index, ax=axes[1],
            hue=status_risk.index, palette="Oranges_r", legend=False)
axes[1].set_title("Order Status — Mean Late Delivery Risk")
savefig("order_status_volume_and_risk.png")

plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="Days for shipment (scheduled)", hue="Late_delivery_risk",
             multiple="dodge", bins=range(0, 8), palette=["#4C72B0", "#DD8452"])
plt.title("Scheduled Shipping Days by Delivery Risk")
savefig("scheduled_days_by_risk.png")

plt.figure(figsize=(6, 4))
sched_risk = df.groupby("Days for shipment (scheduled)")["Late_delivery_risk"].mean()
plt.plot(sched_risk.index, sched_risk.values, marker="o", color="#065A82", linewidth=2)
plt.title("Late Delivery Risk vs Scheduled Shipping Window")
savefig("risk_vs_scheduled_days.png")

plt.figure(figsize=(6, 4))
sns.histplot(df["Sales"], bins=60, color="#55A868")
plt.xlim(0, 500)
plt.title("Distribution of Sales per Order Item")
savefig("sales_distribution.png")

plt.figure(figsize=(6, 4))
sns.boxplot(x="Late_delivery_risk", y="Order Item Discount Rate", data=df,
            hue="Late_delivery_risk", palette=["#4C72B0", "#DD8452"], legend=False)
plt.title("Order Discount Rate vs Late Delivery Risk")
savefig("discount_rate_vs_risk.png")

plt.figure(figsize=(6, 4))
sns.boxplot(x="Late_delivery_risk", y="Order Item Profit Ratio", data=df,
            hue="Late_delivery_risk", palette=["#4C72B0", "#DD8452"], legend=False,
            showfliers=False)
plt.title("Order Profit Ratio vs Late Delivery Risk")
savefig("profit_ratio_vs_risk.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
month_vol = df.groupby("Order Month").size()
axes[0].bar(month_vol.index, month_vol.values, color="#1C7293")
axes[0].set_title("Order Volume by Month")
month_risk = df.groupby("Order Month")["Late_delivery_risk"].mean()
axes[1].plot(month_risk.index, month_risk.values, marker="o", color="#E07A5F", linewidth=2)
axes[1].set_title("Late Delivery Risk Rate by Month")
savefig("monthly_volume_and_risk.png")

plt.figure(figsize=(6, 4))
wd_risk = df.groupby("Order Weekday")["Late_delivery_risk"].mean()
wd_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
plt.bar(wd_labels, wd_risk.values, color="#8E5572")
plt.title("Late Delivery Risk Rate by Order Weekday")
savefig("weekday_risk.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
hour_vol = df.groupby("Order Hour").size()
axes[0].plot(hour_vol.index, hour_vol.values, marker="o", color="#3D405B")
axes[0].set_title("Order Volume by Hour")
hour_risk = df.groupby("Order Hour")["Late_delivery_risk"].mean()
axes[1].plot(hour_risk.index, hour_risk.values, marker="o", color="#E07A5F")
axes[1].set_title("Late Delivery Risk Rate by Hour")
savefig("hourly_volume_and_risk.png")

plt.figure(figsize=(9, 5))
top_countries = df["Order Country"].value_counts().head(10).index
country_risk = df[df["Order Country"].isin(top_countries)].groupby("Order Country")["Late_delivery_risk"].mean().sort_values(ascending=False)
sns.barplot(x=country_risk.values, y=country_risk.index, hue=country_risk.index,
            palette="crest", legend=False)
plt.title("Late Delivery Risk Rate — Top 10 Countries")
savefig("risk_by_top10_countries.png")

num_cols = df.select_dtypes(include=[np.number]).columns
corr = df[num_cols].corr()
plt.figure(figsize=(14, 11))
sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3)
plt.title("Correlation Heatmap of Numeric Features")
savefig("correlation_heatmap.png")

plt.figure(figsize=(7, 5))
sample = df.sample(min(8000, len(df)), random_state=42)
sns.scatterplot(data=sample, x="Days for shipment (scheduled)", y="Sales",
                 hue="Late_delivery_risk", alpha=0.35, palette=["#4C72B0", "#DD8452"], s=18)
plt.ylim(0, 500)
plt.title("Sales vs Scheduled Shipping Days")
savefig("sales_vs_scheduled_days_scatter.png")

plt.figure(figsize=(6, 4))
type_risk = df.groupby("Type")["Late_delivery_risk"].mean().sort_values(ascending=False)
sns.barplot(x=type_risk.index, y=type_risk.values, hue=type_risk.index,
            palette="flare", legend=False)
plt.title("Late Delivery Risk Rate by Payment Type")
savefig("risk_by_payment_type.png")

plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="Order Item Quantity", hue="Late_delivery_risk",
             multiple="dodge", bins=range(1, 7), palette=["#4C72B0", "#DD8452"])
plt.title("Order Item Quantity by Delivery Risk")
savefig("quantity_by_risk.png")

plt.figure(figsize=(6, 4))
sns.boxplot(x="Late_delivery_risk", y="Benefit per order", data=df,
            hue="Late_delivery_risk", palette=["#4C72B0", "#DD8452"], legend=False,
            showfliers=False)
plt.title("Benefit per Order vs Late Delivery Risk")
savefig("benefit_per_order_vs_risk.png")

pivot1 = df.pivot_table(index="Shipping Mode", columns="Order Region",
                         values="Late_delivery_risk", aggfunc="mean")
plt.figure(figsize=(16, 4.5))
sns.heatmap(pivot1, cmap="YlOrRd", annot=False, linewidths=0.3)
plt.title("Late Delivery Risk Rate: Shipping Mode x Order Region")
savefig("heatmap_shipping_mode_x_region.png")

pivot2 = df.pivot_table(index="Shipping Mode", columns="Customer Segment",
                         values="Late_delivery_risk", aggfunc="mean")
plt.figure(figsize=(6, 4.5))
sns.heatmap(pivot2, cmap="YlOrRd", annot=True, fmt=".2f", linewidths=0.3)
plt.title("Late Delivery Risk Rate: Shipping Mode x Customer Segment")
savefig("heatmap_shipping_mode_x_segment.png")

status_ct = pd.crosstab(df["Late_delivery_risk"], df["Order Status"], normalize="index")
status_ct.index = ["On-time", "Late-risk"]
plt.figure(figsize=(10, 5))
status_ct.plot(kind="bar", stacked=True, colormap="tab20", ax=plt.gca())
plt.title("Order Status Composition")
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
savefig("order_status_composition_stacked.png")

plt.figure(figsize=(6, 4))
year_risk = df.groupby("Order Year")["Late_delivery_risk"].mean()
plt.plot(year_risk.index, year_risk.values, marker="o", color="#065A82", linewidth=2)
plt.title("Late Delivery Risk Trend by Year")
plt.xticks(year_risk.index)
savefig("yearly_risk_trend.png")

plt.figure(figsize=(7, 5))
sns.scatterplot(data=sample, x="Order Item Discount Rate", y="Order Item Quantity",
                 hue="Late_delivery_risk", alpha=0.35, palette=["#4C72B0", "#DD8452"], s=18)
plt.title("Order Quantity vs Discount Rate")
savefig("quantity_vs_discount_scatter.png")

print("Generating deep distribution plots (Violin)...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.violinplot(x="Late_delivery_risk", y="Days for shipment (scheduled)", data=df, 
               palette=["#4C72B0", "#DD8452"], ax=axes[0], inner="quartile")
axes[0].set_title("Scheduled Days Density by Risk")
sns.violinplot(x="Late_delivery_risk", y="Sales", data=df, 
               palette=["#4C72B0", "#DD8452"], ax=axes[1], inner="quartile")
axes[1].set_ylim(0, 800)
axes[1].set_title("Sales Density by Risk")
sns.violinplot(x="Late_delivery_risk", y="Order Item Profit Ratio", data=df, 
               palette=["#4C72B0", "#DD8452"], ax=axes[2], inner="quartile")
axes[2].set_ylim(-1, 1)
axes[2].set_title("Profit Ratio Density by Risk")
savefig("deep_violin_distributions.png")

# ==========================================================================
# 5. MODELING (LR, DT, KNN, ANN)
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 5: Model training")
print("=" * 70)

y = df_enc["Late_delivery_risk"]
X = df_enc.drop(columns=["Late_delivery_risk"])
X = X.astype({c: "int8" for c in X.select_dtypes(include="bool").columns})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

results = {}
proba = {}

def evaluate(name, model, Xtr, Xte, needs_proba=True):
    t0 = time.time()
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    t1 = time.time()
    res = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "train_time_s": t1 - t0,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    if needs_proba and hasattr(model, "predict_proba"):
        p = model.predict_proba(Xte)[:, 1]
        proba[name] = p
        res["roc_auc"] = roc_auc_score(y_test, p)
        res["avg_precision"] = average_precision_score(y_test, p)
    results[name] = res
    print(f"  {name}: acc={res['accuracy']:.3f}  prec={res['precision']:.3f}  "
          f"rec={res['recall']:.3f}  f1={res['f1']:.3f}  time={res['train_time_s']:.1f}s")
    return model, pred

lr = LogisticRegression(max_iter=1000, random_state=42)
lr_model, lr_pred = evaluate("Logistic Regression", lr, X_train_s, X_test_s)

dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model, dt_pred = evaluate("Decision Tree", dt, X_train, X_test)

print("  Running KNN...")
knn = KNeighborsClassifier(n_neighbors=15, n_jobs=-1)
knn_model, knn_pred = evaluate("KNN", knn, X_train_s, X_test_s)

print("  Running Artificial Neural Network (ANN)...")
ann = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', 
                    solver='adam', max_iter=300, random_state=42, 
                    early_stopping=True, validation_fraction=0.1)
ann_model, ann_pred = evaluate("Artificial Neural Network", ann, X_train_s, X_test_s)

plt.figure(figsize=(7, 4.5))
plt.plot(ann_model.loss_curve_, color="#8E5572", linewidth=2)
plt.title("ANN Training Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
savefig("ann_loss_curve.png")

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
for ax, (name, res) in zip(axes, results.items()):
    cm = np.array(res["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred: On-time", "Pred: Late"],
                yticklabels=["Actual: On-time", "Actual: Late"])
    ax.set_title(name)
savefig("confusion_matrices.png")

plt.figure(figsize=(6.5, 5.5))
for (name, p), color in zip(proba.items(), PALETTE):
    fpr, tpr, _ = roc_curve(y_test, p)
    auc = roc_auc_score(y_test, p)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)
plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
plt.title("ROC Curves")
plt.legend()
savefig("roc_curves.png")

plt.figure(figsize=(6.5, 5.5))
for (name, p), color in zip(proba.items(), PALETTE):
    prec, rec, _ = precision_recall_curve(y_test, p)
    ap = average_precision_score(y_test, p)
    plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=color, linewidth=2)
plt.title("Precision-Recall Curves")
plt.legend()
savefig("precision_recall_curves.png")

# 5.6 Cost-Sensitive Business Evaluation
print("\nGenerating Cost-Sensitive Profit Impact Analysis...")
y_test_np = np.array(y_test)
C_FN_array = X_test["Order Item Profit Ratio"].values 
C_FP_scalar = 0.15 

plt.figure(figsize=(8, 5))
for (name, p), color in zip(proba.items(), PALETTE):
    thresholds = np.linspace(0.05, 0.95, 50)
    expected_costs = []
    
    for thresh in thresholds:
        preds = (p >= thresh).astype(int)
        false_negatives = (y_test_np == 1) & (preds == 0)
        false_positives = (y_test_np == 0) & (preds == 1)
        
        total_penalty = np.sum(C_FN_array[false_negatives]) + np.sum(false_positives * C_FP_scalar)
        expected_costs.append(total_penalty)
        
    plt.plot(thresholds, expected_costs, label=name, color=color, linewidth=2)

plt.xlabel("Decision Threshold (Predicted Probability)")
plt.ylabel("Total Expected Financial Penalty")
plt.title("Expected Value Framework: Financial Impact vs. Threshold")
plt.legend()
savefig("cost_sensitive_evaluation.png")

# ==========================================================================
# 6. CROSS-VALIDATION
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 6: 5-fold stratified cross-validation (40k subsample)")
print("=" * 70)

X_sub, _, y_sub, _ = train_test_split(X, y, train_size=40000, random_state=42, stratify=y)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1"]

pipes = {
    "Logistic Regression": Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))]),
    "Decision Tree": Pipeline([("clf", DecisionTreeClassifier(max_depth=10, random_state=42))]),
}

cv_results = {}
for name, pipe in pipes.items():
    cvr = cross_validate(pipe, X_sub, y_sub, cv=skf, scoring=scoring, n_jobs=1)
    summary = {m: [float(cvr[f"test_{m}"].mean()), float(cvr[f"test_{m}"].std())] for m in scoring}
    cv_results[name] = summary

plt.figure(figsize=(8, 5))
names = list(cv_results.keys())
acc_means = [cv_results[n]["accuracy"][0] for n in names]
acc_stds = [cv_results[n]["accuracy"][1] for n in names]
f1_means = [cv_results[n]["f1"][0] for n in names]
f1_stds = [cv_results[n]["f1"][1] for n in names]
x = np.arange(len(names))
width = 0.35
plt.bar(x - width / 2, acc_means, width, yerr=acc_stds, capsize=5, label="Accuracy", color="#065A82")
plt.bar(x + width / 2, f1_means, width, yerr=f1_stds, capsize=5, label="F1-score", color="#E07A5F")
plt.xticks(x, names)
plt.title("5-Fold Cross-Validation Results")
plt.legend()
savefig("cv_results_errorbars.png")

# ==========================================================================
# 7. FEATURE IMPORTANCE & PARTIAL DEPENDENCE
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 7: Feature importance & Partial Dependence")
print("=" * 70)

cols = list(X.columns)
imp = pd.Series(dt_model.feature_importances_, index=cols).sort_values(ascending=False)
plt.figure(figsize=(8, 6))
top_imp = imp.head(15)
plt.barh(top_imp.index[::-1], top_imp.values[::-1], color="#55A868")
plt.title("Decision Tree: Top 15 Feature Importances")
savefig("dt_importance.png")

print("\nGenerating Partial Dependence Plots (Deep Interpretability)...")
top_numeric_features = ["Days for shipment (scheduled)", "Sales", "Order Item Discount Rate"]
features_to_plot = [col for col in top_numeric_features if col in X_train.columns]

if features_to_plot:
    fig, ax = plt.subplots(figsize=(12, 5))
    display = PartialDependenceDisplay.from_estimator(
        dt_model, X_train, features_to_plot,
        kind="average", ax=ax, grid_resolution=50
    )
    plt.subplots_adjust(top=0.9)
    fig.suptitle("Partial Dependence: How Top Features Drive Late Delivery Risk")
    savefig("partial_dependence_plots.png")

# ==========================================================================
# 8. K-MEANS CLUSTERING & SILHOUETTE VALIDATION
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 8: Exploratory K-Means clustering")
print("=" * 70)

agg = df.groupby(["Order Region", "Category Name"]).agg(
    late_risk_rate=("Late_delivery_risk", "mean"),
    avg_scheduled_days=("Days for shipment (scheduled)", "mean"),
    avg_sales=("Sales", "mean"),
    order_volume=("Sales", "count"),
).reset_index()
agg = agg[agg["order_volume"] >= 10]

features = ["late_risk_rate", "avg_scheduled_days", "avg_sales", "order_volume"]
Xk = agg[features].copy()
Xk["order_volume"] = np.log1p(Xk["order_volume"])
kscaler = StandardScaler()
Xks = kscaler.fit_transform(Xk)

k_final = 4
km = KMeans(n_clusters=k_final, random_state=42, n_init=10)
agg["cluster"] = km.fit_predict(Xks)

plt.figure(figsize=(7, 5))
sc = plt.scatter(agg["avg_scheduled_days"], agg["late_risk_rate"], c=agg["cluster"],
                  cmap="tab10", s=agg["order_volume"] / agg["order_volume"].max() * 300 + 20, alpha=0.7)
plt.xlabel("Average Scheduled Shipping Days")
plt.ylabel("Late Delivery Risk Rate")
plt.title(f"K-Means Clusters: Region x Category Delivery Behaviour (k={k_final})")
plt.colorbar(sc, label="Cluster")
savefig("kmeans_clusters.png")

print("\nCalculating Silhouette Scores for K-Means...")
silhouette_avg = silhouette_score(Xks, agg["cluster"])
sample_silhouette_values = silhouette_samples(Xks, agg["cluster"])

fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
y_lower = 10

for i in range(k_final):
    ith_cluster_silhouette_values = sample_silhouette_values[agg["cluster"] == i]
    ith_cluster_silhouette_values.sort()
    
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = plt.cm.tab10(float(i) / k_final)
    ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

ax1.set_title("Silhouette Plot for Cluster Geometry Validation")
ax1.set_xlabel("Silhouette Coefficient Values")
ax1.set_ylabel("Cluster Label")
ax1.axvline(x=silhouette_avg, color="red", linestyle="--", label=f"Mean Silhouette = {silhouette_avg:.3f}")
ax1.legend()
savefig("kmeans_silhouette_plot.png")

# ==========================================================================
# 9. ADVANCED ML DIAGNOSTICS & TIME-SERIES EXPANSION (22 New Plots)
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 9: Generating 20+ Advanced Analytics & Diagnostics")
print("=" * 70)

from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve

# --- MODULE A: RIGOROUS MODEL DIAGNOSTICS ---

# 1. Calibration Curve (Reliability Diagram)
print("Generating Calibration Curves...")
plt.figure(figsize=(7, 7))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))
ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for name, p in proba.items():
    if name != "Decision Tree": # DT probabilities are too discrete for clean calibration curves
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, p, n_bins=10)
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label=name, color=PALETTE[list(proba.keys()).index(name)])
        ax2.hist(p, range=(0, 1), bins=10, histtype="step", lw=2, color=PALETTE[list(proba.keys()).index(name)])
ax1.set_ylabel("Fraction of positives")
ax1.set_title("Calibration Plots (Reliability Curve)")
ax1.legend(loc="lower right")
ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
savefig("calibration_reliability_curve.png")

# 2. Permutation Feature Importance (Logistic Regression)
print("Calculating Permutation Importance (Rigorous Feature Impact)...")
perm_importance = permutation_importance(lr_model, X_test_s, y_test, n_repeats=5, random_state=42, n_jobs=-1)
sorted_idx = perm_importance.importances_mean.argsort()[-15:] # Top 15
plt.figure(figsize=(8, 6))
plt.boxplot(perm_importance.importances[sorted_idx].T, vert=False, labels=np.array(X.columns)[sorted_idx])
plt.title("Permutation Feature Importance (Logistic Regression)")
plt.xlabel("Decrease in Accuracy Score")
savefig("permutation_importance.png")

# 3. Learning Curve (Bias vs Variance Analysis)
print("Generating Learning Curves for Logistic Regression...")
train_sizes, train_scores, test_scores = learning_curve(
    LogisticRegression(max_iter=1000), X_train_s, y_train, cv=3, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 5), scoring="f1")
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(7, 5))
plt.plot(train_sizes, train_mean, color="#065A82", marker="o", markersize=5, label="Training F1")
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color="#065A82")
plt.plot(train_sizes, test_mean, color="#E07A5F", marker="s", markersize=5, linestyle="--", label="Validation F1")
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color="#E07A5F")
plt.title("Learning Curve: Asymptotic Convergence Analysis")
plt.xlabel("Number of Training Samples (N)")
plt.ylabel("F1 Score")
plt.legend(loc="lower right")
savefig("learning_curve_convergence.png")

# 4. Error Residuals Distribution
print("Plotting Prediction Residuals...")
lr_residuals = y_test - proba["Logistic Regression"]
plt.figure(figsize=(7, 4))
sns.histplot(lr_residuals, bins=50, kde=True, color="#3D405B")
plt.title("Distribution of Logistic Regression Residuals (y - p)")
plt.xlabel("Residual Error")
savefig("residual_distribution.png")

# --- MODULE B: DEEP TIME-SERIES DYNAMICS ---

# Re-establish temporal index for time-series math
df_ts = df.copy()
df_ts["order date (DateOrders)"] = df_raw["order date (DateOrders)"] # Recover dropped column
df_ts["order date (DateOrders)"] = pd.to_datetime(df_ts["order date (DateOrders)"])
df_ts.set_index("order date (DateOrders)", inplace=True)
df_ts = df_ts.sort_index()

# 5. Daily Rolling Average Risk (7-day window)
daily_risk = df_ts.resample("D")["Late_delivery_risk"].mean()
rolling_risk = daily_risk.rolling(window=7).mean()
plt.figure(figsize=(12, 4))
plt.plot(daily_risk.index, daily_risk.values, alpha=0.3, color="#C44E52", label="Daily Avg")
plt.plot(rolling_risk.index, rolling_risk.values, color="#1C7293", linewidth=2, label="7-Day Rolling Mean")
plt.title("Temporal Volatility: 7-Day Rolling Average of Delivery Risk")
plt.ylabel("Late Risk Probability")
plt.legend()
savefig("timeseries_7day_rolling_risk.png")

# 6. Day of the Month Risk Matrix
df["Order Day of Month"] = df_raw["order date (DateOrders)"].apply(lambda x: pd.to_datetime(x).day)
pivot_dom = df.pivot_table(index="Order Month", columns="Order Day of Month", values="Late_delivery_risk", aggfunc="mean")
plt.figure(figsize=(15, 5))
sns.heatmap(pivot_dom, cmap="viridis", linewidths=0.1)
plt.title("Calendar Physics: Delivery Risk by Month vs. Day-of-Month")
savefig("heatmap_month_vs_dayofmonth.png")

# 7. Year-Over-Year (YoY) Monthly Overlay
yoy_risk = df.pivot_table(index="Order Month", columns="Order Year", values="Late_delivery_risk", aggfunc="mean")
yoy_risk.plot(figsize=(8, 5), marker="o", colormap="tab10")
plt.title("Year-Over-Year (YoY) Monthly Risk Trajectories")
plt.ylabel("Mean Late Risk")
savefig("timeseries_yoy_overlay.png")

# 8. Shipping Velocity by Quarter
df["Order Quarter"] = df_raw["order date (DateOrders)"].apply(lambda x: pd.to_datetime(x).quarter)
plt.figure(figsize=(7, 5))
sns.boxplot(x="Order Quarter", y="Days for shipment (scheduled)", data=df, hue="Late_delivery_risk", palette=["#4C72B0", "#DD8452"])
plt.title("Shipping Velocity Constraints by Quarter")
savefig("scheduled_days_by_quarter.png")

# --- MODULE C: GEOGRAPHICAL & NETWORK TOPOLOGY ---

# 9. Geo-Risk Network Matrix (Source Market vs Destination Region)
geo_pivot = df.pivot_table(index="Market", columns="Order Region", values="Late_delivery_risk", aggfunc="mean")
plt.figure(figsize=(12, 6))
sns.heatmap(geo_pivot, cmap="magma", annot=True, fmt=".2f", linewidths=0.5)
plt.title("Geo-Risk Network: Source Market to Destination Region")
savefig("geo_network_risk_matrix.png")

# 10. Risk Variance by Top 20 Cities
top_cities = df["Order City"].value_counts().head(20).index if "Order City" in df.columns else df_raw["Order City"].value_counts().head(20).index
city_risk = df_raw[df_raw["Order City"].isin(top_cities)].groupby("Order City")["Late_delivery_risk"].mean().sort_values(ascending=True)
plt.figure(figsize=(8, 7))
plt.barh(city_risk.index, city_risk.values, color="#5FB3A3")
plt.title("Micro-Geography: Risk Rate in Top 20 Volume Cities")
savefig("risk_top20_cities.png")

# 11. Delivery Status Probabilities (Transition Proxies)
status_probs = df_raw["Delivery Status"].value_counts(normalize=True)
plt.figure(figsize=(7, 7))
plt.pie(status_probs.values, labels=status_probs.index, autopct='%1.1f%%', colors=sns.color_palette("Set2"))
plt.title("Global Probabilistic State of Delivery Status")
savefig("delivery_status_pie.png")

# 12. Suspected Fraud Analytics (Volume over Time)
fraud_df = df_raw[df_raw["Order Status"] == "SUSPECTED_FRAUD"].copy()
fraud_df["order date (DateOrders)"] = pd.to_datetime(fraud_df["order date (DateOrders)"])
fraud_ts = fraud_df.set_index("order date (DateOrders)").resample("M").size()
plt.figure(figsize=(10, 4))
plt.plot(fraud_ts.index, fraud_ts.values, color="red", marker="x", linestyle="-")
plt.title("System Anomalies: Suspected Fraud Volume over Time")
plt.ylabel("Fraudulent Orders")
savefig("fraud_volume_timeseries.png")

# --- MODULE D: FINANCIAL PHYSICS & PARETO OPTIMIZATION ---

# 13. Pareto Curve of Profit Margins
profits = df_raw["Order Item Profit Ratio"].sort_values(ascending=False).values
cumulative_profit = np.cumsum(profits) / np.sum(profits[profits > 0]) # Normalize positive space
plt.figure(figsize=(7, 5))
plt.plot(np.linspace(0, 100, len(cumulative_profit)), cumulative_profit * 100, color="purple", lw=2)
plt.axhline(80, color="gray", linestyle="--")
plt.axvline(20, color="gray", linestyle="--")
plt.title("Financial Physics: Pareto Optimization Curve (Profit Ratio)")
plt.xlabel("Percentage of Orders (%)")
plt.ylabel("Cumulative Profit (%)")
savefig("pareto_profit_curve.png")

# 14. Profit vs Discount Rate Phase Space
plt.figure(figsize=(8, 6))
sample_fin = df_raw.sample(5000, random_state=42)
sns.kdeplot(data=sample_fin, x="Order Item Discount Rate", y="Order Item Profit Ratio", hue="Late_delivery_risk", fill=True, alpha=0.5, palette=["blue", "orange"])
plt.title("Financial Phase Space: Discount Rate vs Profit Ratio Density")
savefig("financial_phase_space_kde.png")

# 15. Item Price vs Quantity Risk Boundary
plt.figure(figsize=(8, 6))
sns.scatterplot(data=sample_fin, x="Order Item Product Price", y="Order Item Quantity", hue="Late_delivery_risk", palette=["#4C72B0", "#DD8452"], alpha=0.4, s=20)
plt.title("Risk Boundary Matrix: Product Price vs Item Quantity")
savefig("price_vs_quantity_scatter.png")

# 16. Total Sales Financial Distribution (Log Scale)
plt.figure(figsize=(7, 4))
sns.histplot(np.log1p(df_raw["Sales"]), bins=50, kde=True, color="#8E5572")
plt.title("Log-Normal Distribution of Sales Volume")
plt.xlabel("Log(Sales + 1)")
savefig("log_sales_distribution.png")

# 17. Late Delivery Rate by Customer Segment x Payment Type
pivot_fin = df_raw.pivot_table(index="Customer Segment", columns="Type", values="Late_delivery_risk", aggfunc="mean")
plt.figure(figsize=(8, 4))
sns.heatmap(pivot_fin, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Financial Risk: Customer Segment vs Payment Mechanism")
savefig("heatmap_segment_vs_payment.png")

# 18. Product Category Cannibalization / Co-variance Matrix (Top 10)
top10_cats = df_raw["Category Name"].value_counts().head(10).index
subset_cats = df_raw[df_raw["Category Name"].isin(top10_cats)]
ct_cats = pd.crosstab(subset_cats["Order Region"], subset_cats["Category Name"])
plt.figure(figsize=(10, 6))
sns.heatmap(ct_cats, cmap="Blues", annot=False)
plt.title("Market Topography: Product Category vs Order Region Demand")
savefig("category_region_covariance.png")

# 19. Average Delivery Delay by Department
delay = df_raw["Days for shipping (real)"] - df_raw["Days for shipment (scheduled)"]
df_raw["Delay_Magnitude"] = delay
dept_delay = df_raw.groupby("Department Name")["Delay_Magnitude"].mean().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
plt.bar(dept_delay.index, dept_delay.values, color="#E07A5F")
plt.xticks(rotation=45, ha="right")
plt.title("Operational Physics: Mean Delay Magnitude (Days) by Department")
plt.ylabel("Days Late (Negative = Early)")
savefig("delay_magnitude_by_department.png")

# 20. Benefit per Order Outlier Analysis
plt.figure(figsize=(8, 4))
sns.violinplot(x="Customer Segment", y="Benefit per order", data=df_raw, palette="Set3")
plt.title("Financial Outliers: Benefit per Order Density by Segment")
savefig("benefit_outliers_segment.png")

# 21. Department vs Payment Type Heatmap
dept_pay = df_raw.pivot_table(index="Department Name", columns="Type", values="Late_delivery_risk", aggfunc="mean")
plt.figure(figsize=(8, 6))
sns.heatmap(dept_pay, cmap="YlGnBu", annot=True, fmt=".2f")
plt.title("Risk Matrix: Department Demand vs Payment Type")
savefig("heatmap_dept_vs_payment.png")

# 22. Scheduled Days Density by Order Status
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df_raw, x="Days for shipment (scheduled)", hue="Order Status", common_norm=False, alpha=0.5, linewidth=2)
plt.title("Scheduling Physics: Time Horizon Densities across Order States")
savefig("kde_scheduled_days_order_status.png")

# ==========================================================================
# 10. STATISTICAL MECHANICS, MARKOV DYNAMICS & SURVIVAL ANALYSIS
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 10: Deep System Dynamics & Thermodynamic Entropy")
print("=" * 70)

import networkx as nx
from scipy.stats import entropy

# --- 1. Shannon Entropy of Delivery States by Region ---
print("Calculating Information Entropy (Systemic Chaos)...")
# Calculate the probability distribution of Delivery Status within each region
status_probs = df_raw.groupby('Order Region')['Delivery Status'].value_counts(normalize=True).unstack(fill_value=0)
# Apply Shannon entropy across the distributions
region_entropy = status_probs.apply(lambda row: entropy(row, base=2), axis=1).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=region_entropy.values, y=region_entropy.index, palette="magma")
plt.title("Systemic Chaos: Shannon Entropy of Delivery States by Region")
plt.xlabel("Entropy H(X) (Bits)")
plt.axvline(region_entropy.mean(), color="gray", linestyle="--", label="Global Mean Entropy")
plt.legend()
savefig("shannon_entropy_by_region.png")

# --- 2. Markov Transition Matrix (Scheduled vs Realized Time) ---
print("Computing Markov State-Space Transitions...")
# Filter to valid positive days
mc_df = df_raw[(df_raw['Days for shipment (scheduled)'] >= 0) & (df_raw['Days for shipping (real)'] >= 0)]
transition_matrix = pd.crosstab(
    mc_df['Days for shipment (scheduled)'], 
    mc_df['Days for shipping (real)'], 
    normalize='index' # P(Real | Scheduled)
)

plt.figure(figsize=(10, 8))
sns.heatmap(transition_matrix, cmap="YlOrRd", annot=True, fmt=".2f", cbar_kws={'label': 'Transition Probability P(i, j)'})
plt.title("Markov Transition Matrix: Scheduled vs. Realized Shipping Days")
plt.xlabel("Realized Shipping Days (State j)")
plt.ylabel("Scheduled Shipping Days (State i)")
savefig("markov_transition_matrix.png")

# --- 3. Network Topology & Eigenvector Centrality ---
print("Mapping Graph Theory Bottlenecks...")
# Build a directed graph from Source Market to Destination Region weighted by risk
edges = df_raw.groupby(['Market', 'Order Region'])['Late_delivery_risk'].mean().reset_index()
G = nx.from_pandas_edgelist(edges, source='Market', target='Order Region', edge_attr='Late_delivery_risk', create_using=nx.DiGraph())

# Calculate Eigenvector Centrality (Influence of a node in the risk network)
try:
    centrality = nx.eigenvector_centrality_numpy(G, weight='Late_delivery_risk')
    cent_df = pd.Series(centrality).sort_values(ascending=False).head(15)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=cent_df.values, y=cent_df.index, palette="mako")
    plt.title("Eigenvector Centrality: Top Structural Risk Bottlenecks")
    plt.xlabel("Centrality Score")
    savefig("eigenvector_centrality_network.png")
except Exception as e:
    print(f"  Skipped Centrality calculation: {e}")

# --- 4. Supply Chain Volatility (The Bullwhip Effect Proxy) ---
print("Analyzing Temporal Volatility (Bullwhip Variance)...")
# Variance of sales volume over 7-day rolling windows
df_ts = df.copy()
df_ts['Date'] = pd.to_datetime(df_raw['order date (DateOrders)'])
daily_sales = df_ts.groupby(df_ts['Date'].dt.date)['Sales'].sum()
rolling_variance = daily_sales.rolling(window=7).var()

plt.figure(figsize=(12, 4))
plt.plot(daily_sales.index, rolling_variance, color="#8E5572", linewidth=1.5)
plt.title("System Instability: 7-Day Rolling Variance of Sales Volume (Bullwhip Proxy)")
plt.ylabel("Sales Variance $\sigma^2$")
savefig("bullwhip_rolling_variance.png")

# --- 5. Survival Analysis (Kaplan-Meier Time-to-Delivery) ---
print("Computing Kaplan-Meier Survival Curves...")
try:
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    
    plt.figure(figsize=(9, 6))
    # We treat "Days for shipping (real)" as duration, and assume all are 'events' (observed)
    # Compare Standard vs First Class
    for mode in ["Standard Class", "First Class"]:
        subset = df_raw[df_raw["Shipping Mode"] == mode]
        T = subset["Days for shipping (real)"]
        E = np.ones(len(subset)) # All deliveries were ultimately completed in this historical dataset
        
        kmf.fit(T, event_observed=E, label=mode)
        kmf.plot_survival_function(linewidth=2)
        
    plt.title("Survival Analysis: Probability of Order Remaining Undelivered Over Time")
    plt.xlabel("Days Since Order ($t$)")
    plt.ylabel("Survival Probability $\hat{S}(t)$")
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label="Median Delivery Horizon")
    plt.legend()
    savefig("survival_analysis_kaplan_meier.png")
except ImportError:
    print("  Skipped Survival Analysis: 'lifelines' library not installed. Run 'pip install lifelines'")
# ==========================================================================
# DONE
# ==========================================================================
print("\n" + "=" * 70)
print(f"ALL DONE. {PLOT_COUNTER[0]} plots + result files written to: {OUT_DIR}")
print("=" * 70)

if SHOW_PLOTS:
    print("\nAll plot windows are open on screen. Close them (or Ctrl+C in this")
    print("console) when you're done reviewing — the script will then exit.")
    plt.show()