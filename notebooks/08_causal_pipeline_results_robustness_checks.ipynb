import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

sns.set(style="whitegrid")


# ======================================================
# PART A: REAL PARALLEL TRENDS VISUAL (ACTUAL DATA)
# ======================================================

def plot_parallel_trends_real(long_df, title="Parallel Trends (Actual Data)"):
    """
    Plot parallel trends using your real long_df panel.
    Requires columns: sales, treated, relative_day
    """
    if "relative_day" not in long_df.columns:
        raise ValueError("long_df must contain `relative_day` for parallel trends plot.")

    parallel_data = (
        long_df.groupby(['relative_day', 'treated'])['sales']
               .mean()
               .reset_index()
               .assign(Group=lambda df: df['treated'].map({1: 'Treated', 0: 'Control'}))
    )

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=parallel_data, x='relative_day', y='sales',
                 hue='Group', marker='o')
    plt.axvline(0, color='black', linestyle='--', label='Treatment Date')
    plt.title(title)
    plt.xlabel('Days Relative to Treatment')
    plt.ylabel('Average Sales')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()



# ======================================================
# PART B: EVENT STUDY (ACTUAL DATA)
# ======================================================

def event_study_real(long_df):
    """
    Run and plot an event study using real long_df.
    Requires: sales, treated, relative_day
    """

    event_df = long_df.copy()

    # Remove baseline day -1 as reference category
    event_df = event_df[event_df['relative_day'] != -1].copy()
    event_df['relative_day_str'] = event_df['relative_day'].astype(str)

    model = smf.ols("sales ~ C(relative_day_str) * treated", data=event_df).fit()

    # Extract treatment interaction coefficients
    coefs = model.params.filter(like='C(relative_day_str)[T.').filter(like=':treated')
    coefs = coefs.reset_index()
    coefs.columns = ['term', 'estimate']
    coefs['relative_day'] = coefs['term'].str.extract(
        r'\[T\.(-?\d+)\]:treated')[0].astype(int)

    # Confidence intervals
    ci = model.conf_int().loc[coefs['term']]
    coefs['ci_low'] = ci[0].values
    coefs['ci_high'] = ci[1].values

    # Plot
    plt.figure(figsize=(10, 5))
    plt.axvline(0, color='black', linestyle='--', label='Treatment')
    plt.axhline(0, color='gray', linestyle='-')
    plt.errorbar(
        coefs['relative_day'], coefs['estimate'],
        yerr=[coefs['estimate'] - coefs['ci_low'],
              coefs['ci_high'] - coefs['estimate']],
        fmt='o', capsize=4
    )

    plt.title("Event Study: Treatment Effect Over Time (Actual Data)")
    plt.xlabel("Days Relative to Treatment")
    plt.ylabel("Treatment Effect on Sales")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

    return model



# ======================================================
# PART C: ROBUSTNESS CHECKS FOR DiD + IPW MODEL
# ======================================================

def did_robustness_checks(long_df, final_df, did_model):
    """
    Full robustness suite:
      - Real pre-trends regression
      - Model fit stats
      - Residuals
      - Sensitivity to covariates
      - Placebo test
      - Heterogeneous effects
      - Event study (real)
    """

    # --------------------------------------
    # 1. Parallel Trends (Real Data)
    # --------------------------------------
    print("\n========== 1. Pre-Treatment Parallel Trends ==========")
    pre_df = long_df[long_df['post'] == 0]
    pre_model = smf.ols("sales ~ treated + C(store_id)", data=pre_df).fit()
    print(pre_model.summary())


    # --------------------------------------
    # 2. Model Fit Checks
    # --------------------------------------
    print("\n========== 2. DiD + IPW Model Fit ==========")
    print(f"AIC: {did_model.aic:.2f}")
    print(f"BIC: {did_model.bic:.2f}")
    print(f"F-statistic: {did_model.fvalue:.2f}, p-value: {did_model.f_pvalue:.4f}")


    # --------------------------------------
    # 3. Residual Diagnostics
    # --------------------------------------
    print("\n========== 3. Residual Diagnostics ==========")
    residuals = did_model.resid

    plt.figure(figsize=(6, 3))
    sns.histplot(residuals, kde=True)
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.show()

    sm.qqplot(residuals, line='s')
    plt.title("Q-Q Plot of Residuals")
    plt.tight_layout()
    plt.show()


    # --------------------------------------
    # 4. Covariate Sensitivity
    # --------------------------------------
    print("\n========== 4. Sensitivity to Covariates ==========")
    sen_covs = ['Store_Size', 'Customer_Age', 'Foot_Traffic']
    long_cov = long_df.copy()

    for cov in sen_covs:
        long_cov[cov] = final_df[cov].repeat(2).reset_index(drop=True)

    formula = "sales ~ treated * post + " + " + ".join(sen_covs)
    cov_model = smf.wls(formula, data=long_cov, weights=long_cov['ipw']).fit()
    print(cov_model.summary())


    # --------------------------------------
    # 5. Placebo Test
    # --------------------------------------
    print("\n========== 5. Placebo Test (Shift Treatment Earlier) ==========")

    placebo_df = long_df.copy()
    if "relative_day" not in placebo_df:
        placebo_df['relative_day'] = np.where(placebo_df['post'] == 1, 1, -1)

    placebo_df['post_placebo'] = placebo_df['relative_day'] >= -10
    placebo_df['post'] = placebo_df['post_placebo'].astype(int)

    placebo_model = smf.wls(
        "sales ~ treated * post",
        data=placebo_df,
        weights=placebo_df['ipw']
    ).fit()

    print(placebo_model.summary())


    # --------------------------------------
    # 6. Heterogeneous Treatment Effects
    # --------------------------------------
    print("\n========== 6. Heterogeneous Treatment Effects ==========")

    long_df['treated_traffic'] = (
        long_df['treated'] * final_df['Foot_Traffic'].repeat(2).reset_index(drop=True)
    )

    het_model = smf.wls(
        "sales ~ treated * post + treated_traffic",
        data=long_df,
        weights=long_df['ipw']
    ).fit()

    print(het_model.summary())


    # --------------------------------------
    # 7. Event Study (Actual Data)
    # --------------------------------------
    print("\n========== 7. Event Study (Actual Data) ==========")
    event_study_real(long_df)



# ======================================================
# HOW TO USE
# ======================================================

# Parallel Trends (Actual)
# plot_parallel_trends_real(long_df)

# Event Study (Actual)
# event_model = event_study_real(long_df)

# Robustness Checks
# did_robustness_checks(long_df, final_df, did_model)
