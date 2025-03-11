import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import chi2_contingency
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_confint, proportion_effectsize


def run_ab_test():
    st.title("マルチクリエイティブ A/B テスト：CTR & CVR の統計検定と拡張解析（ベイズ可視化付き）")

    # クリエイティブ数の指定（最低2群）
    num_creatives = st.sidebar.number_input("クリエイティブの数", min_value=2, value=2, step=1)

    creatives_data = []
    group_labels = []
    # 各クリエイティブのデータ入力（ラベルは A, B, C, ...）
    for i in range(num_creatives):
        label = chr(65 + i)  # 0->A, 1->B, ...
        group_labels.append(label)
        st.sidebar.header(f"クリエイティブ {label} のデータ")
        impressions = st.sidebar.number_input(f"インプレッション数 ({label})", value=100000, step=1, key=f"imp_{label}")
        clicks = st.sidebar.number_input(f"クリック数 ({label})", value=1000, step=1, key=f"click_{label}")
        conversions = st.sidebar.number_input(f"コンバージョン数 ({label})", value=50, step=1, key=f"conv_{label}")
        creatives_data.append({
            "label": label,
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions
        })

    # 入力値のチェック
    for data in creatives_data:
        if data["clicks"] > data["impressions"]:
            st.error(f"クリエイティブ {data['label']} のクリック数はインプレッション数以下である必要があります。")
            return
        if data["conversions"] > data["impressions"]:
            st.error(f"クリエイティブ {data['label']} のコンバージョン数はインプレッション数以下である必要があります。")
            return

    # 入力データの表示
    st.write("## 入力されたデータ")
    df = pd.DataFrame(creatives_data)
    st.dataframe(df)

    # 各指標の計算（CTR, CVR）
    for data in creatives_data:
        data["ctr"] = data["clicks"] / data["impressions"] if data["impressions"] > 0 else 0
        data["cvr"] = data["conversions"] / data["impressions"] if data["impressions"] > 0 else 0

    st.write("## 各クリエイティブのCTRとCVR")
    for data in creatives_data:
        st.write(f"クリエイティブ {data['label']}: CTR = {data['ctr']:.3%}, CVR = {data['cvr']:.3%}")

    # 1. 全体的な検定（CTR）
    # 各群について、[クリック数, インプレッション数-クリック数] の表を作成
    table_ctr = [[data["clicks"], data["impressions"] - data["clicks"]] for data in creatives_data]
    chi2_ctr, p_ctr, dof_ctr, expected_ctr = chi2_contingency(table_ctr)
    st.write("## CTR の全体的な検定結果 (Chi-squared test)")
    st.write(f"Chi2値: {chi2_ctr:.3f}, p値: {p_ctr:.3f}")
    if p_ctr < 0.05:
        best = max(creatives_data, key=lambda d: d["ctr"])
        st.success(f"統計的に有意な差があり、最高のCTRはクリエイティブ {best['label']} です。")
    else:
        st.info("CTR に統計的な有意差は見られません。")

    # 2. 全体的な検定（CVR）
    table_cvr = [[data["conversions"], data["impressions"] - data["conversions"]] for data in creatives_data]
    chi2_cvr, p_cvr, dof_cvr, expected_cvr = chi2_contingency(table_cvr)
    st.write("## CVR の全体的な検定結果 (Chi-squared test)")
    st.write(f"Chi2値: {chi2_cvr:.3f}, p値: {p_cvr:.3f}")
    if p_cvr < 0.05:
        best = max(creatives_data, key=lambda d: d["cvr"])
        st.success(f"統計的に有意な差があり、最高のCVRはクリエイティブ {best['label']} です。")
    else:
        st.info("CVR に統計的な有意差は見られません。")

    # 3. 信頼区間 (Wilson法)
    st.write("## 信頼区間の計算 (95% CI, Wilson法)")
    st.write("**CTR**")
    for data in creatives_data:
        ci = proportion_confint(data["clicks"], data["impressions"], alpha=0.05, method='wilson')
        st.write(f"{data['label']}: {ci[0]:.3%} ~ {ci[1]:.3%}")
    st.write("**CVR**")
    for data in creatives_data:
        ci = proportion_confint(data["conversions"], data["impressions"], alpha=0.05, method='wilson')
        st.write(f"{data['label']}: {ci[0]:.3%} ~ {ci[1]:.3%}")

    # 4. 効果サイズの評価（ペアワイズ比較）
    st.write("## 効果サイズの評価 (ペアワイズ比較)")
    comparisons = []
    for i in range(len(creatives_data)):
        for j in range(i + 1, len(creatives_data)):
            data_i = creatives_data[i]
            data_j = creatives_data[j]
            delta_ctr = data_i["ctr"] - data_j["ctr"]
            relative_change_ctr = (delta_ctr / data_j["ctr"]) if data_j["ctr"] != 0 else np.nan
            delta_cvr = data_i["cvr"] - data_j["cvr"]
            relative_change_cvr = (delta_cvr / data_j["cvr"]) if data_j["cvr"] != 0 else np.nan
            comparisons.append({
                "比較": f"{data_i['label']} vs {data_j['label']}",
                "CTR 差分": delta_ctr,
                "CTR 相対変化率": relative_change_ctr,
                "CVR 差分": delta_cvr,
                "CVR 相対変化率": relative_change_cvr
            })
    df_comp = pd.DataFrame(comparisons)
    st.dataframe(df_comp.style.format({
        "CTR 差分": "{:.3%}",
        "CTR 相対変化率": "{:.1%}",
        "CVR 差分": "{:.3%}",
        "CVR 相対変化率": "{:.1%}"
    }))

    # 5. サンプルサイズ・パワー分析（ペアワイズ比較）
    st.write("## サンプルサイズ・パワー分析 (有意水準5%, 検出力80%) - ペアワイズ比較")
    analysis = NormalIndPower()
    power_results = []
    for i in range(len(creatives_data)):
        for j in range(i + 1, len(creatives_data)):
            data_i = creatives_data[i]
            data_j = creatives_data[j]
            # CTR のサンプルサイズ計算
            try:
                effect_size_ctr = proportion_effectsize(data_i["ctr"], data_j["ctr"])
                required_n_ctr = analysis.solve_power(effect_size=effect_size_ctr, alpha=0.05, power=0.8, ratio=1)
            except Exception as e:
                required_n_ctr = None
            # CVR のサンプルサイズ計算
            try:
                effect_size_cvr = proportion_effectsize(data_i["cvr"], data_j["cvr"])
                required_n_cvr = analysis.solve_power(effect_size=effect_size_cvr, alpha=0.05, power=0.8, ratio=1)
            except Exception as e:
                required_n_cvr = None
            power_results.append({
                "比較": f"{data_i['label']} vs {data_j['label']}",
                "CTR 必要サンプル数": required_n_ctr,
                "CVR 必要サンプル数": required_n_cvr
            })
    df_power = pd.DataFrame(power_results)
    st.dataframe(df_power.style.format({
        "CTR 必要サンプル数": "{:,.0f}",
        "CVR 必要サンプル数": "{:,.0f}"
    }))

    # 6. ベイズ推定 (Beta分布)
    st.write("## ベイズ推定 (Beta分布)")
    n_samples = 500000
    # CTR：事前分布 Beta(2, 98) を使用
    alpha_prior_ctr = 2
    beta_prior_ctr = 98
    for data in creatives_data:
        data["posterior_ctr"] = np.random.beta(data["clicks"] + alpha_prior_ctr,
                                               data["impressions"] - data["clicks"] + beta_prior_ctr,
                                               n_samples)
    # CVR：事前分布 Beta(1, 1) を使用
    for data in creatives_data:
        data["posterior_cvr"] = np.random.beta(data["conversions"] + 1,
                                               data["impressions"] - data["conversions"] + 1,
                                               n_samples)

    # 各群が最高となる確率の算出（CTR）
    all_posteriors_ctr = np.array([data["posterior_ctr"] for data in creatives_data])
    best_ctr_counts = np.argmax(all_posteriors_ctr, axis=0)
    prob_best_ctr = {}
    st.write("### CTR: 各クリエイティブが最高である確率")
    for idx, label in enumerate(group_labels):
        prob_best_ctr[label] = np.mean(best_ctr_counts == idx)
        st.write(f"クリエイティブ {label}: {prob_best_ctr[label] * 100:.1f}%")

    # 各群が最高となる確率の算出（CVR）
    all_posteriors_cvr = np.array([data["posterior_cvr"] for data in creatives_data])
    best_cvr_counts = np.argmax(all_posteriors_cvr, axis=0)
    prob_best_cvr = {}
    st.write("### CVR: 各クリエイティブが最高である確率")
    for idx, label in enumerate(group_labels):
        prob_best_cvr[label] = np.mean(best_cvr_counts == idx)
        st.write(f"クリエイティブ {label}: {prob_best_cvr[label] * 100:.1f}%")

    # 7. 事後分布の可視化（CTR, CVR）
    st.write("### 事後分布の可視化 (ベイズ推定）")
    colors = sns.color_palette("tab10", n_colors=num_creatives)

    # CTRの事後分布プロット
    fig_ctr, ax_ctr = plt.subplots(figsize=(8, 5))
    for i, data in enumerate(creatives_data):
        sns.kdeplot(data["posterior_ctr"], fill=True, alpha=0.4,
                    label=f"Creative {data['label']}", ax=ax_ctr, color=colors[i])
    ax_ctr.set_title("Posterior Distributions (CTR)")
    ax_ctr.set_xlabel("CTR")
    ax_ctr.legend()
    st.pyplot(fig_ctr)

    # CVRの事後分布プロット
    fig_cvr, ax_cvr = plt.subplots(figsize=(8, 5))
    for i, data in enumerate(creatives_data):
        sns.kdeplot(data["posterior_cvr"], fill=True, alpha=0.4,
                    label=f"Creative {data['label']}", ax=ax_cvr, color=colors[i])
    ax_cvr.set_title("Posterior Distributions (CVR)")
    ax_cvr.set_xlabel("CVR")
    ax_cvr.legend()
    st.pyplot(fig_cvr)

    # 8. HDI (Highest Density Interval) の計算とプロット
    def compute_hdi(samples, cred_mass=0.95):
        """
        与えられたサンプルからHDIを計算する関数
        """
        sorted_samples = np.sort(samples)
        n_samples_sorted = len(sorted_samples)
        interval_idx_inc = int(np.floor(cred_mass * n_samples_sorted))
        n_intervals = n_samples_sorted - interval_idx_inc
        interval_width = sorted_samples[interval_idx_inc:] - sorted_samples[:n_intervals]
        min_idx = np.argmin(interval_width)
        hdi_lower = sorted_samples[min_idx]
        hdi_upper = sorted_samples[min_idx + interval_idx_inc]
        return hdi_lower, hdi_upper

    # CTRのHDIプロット
    st.write("### 事後分布の95% HDI (CTR)")
    fig_hdi_ctr, ax_hdi_ctr = plt.subplots(figsize=(8, 5))
    for i, data in enumerate(creatives_data):
        hdi = compute_hdi(data["posterior_ctr"], 0.95)
        sns.kdeplot(data["posterior_ctr"], fill=True, alpha=0.4,
                    label=f"Creative {data['label']}", ax=ax_hdi_ctr, color=colors[i])
        ax_hdi_ctr.axvline(hdi[0], color=colors[i], linestyle='--')
        ax_hdi_ctr.axvline(hdi[1], color=colors[i], linestyle='--',
                           label=f"{data['label']} 95% HDI: {hdi[0]:.3%} ~ {hdi[1]:.3%}")
    ax_hdi_ctr.set_title("Posterior Distributions 95% HDI (CTR)")
    ax_hdi_ctr.set_xlabel("CTR")
    ax_hdi_ctr.legend()
    st.pyplot(fig_hdi_ctr)

    # CVRのHDIプロット
    st.write("### 事後分布の95% HDI (CVR)")
    fig_hdi_cvr, ax_hdi_cvr = plt.subplots(figsize=(8, 5))
    for i, data in enumerate(creatives_data):
        hdi = compute_hdi(data["posterior_cvr"], 0.95)
        sns.kdeplot(data["posterior_cvr"], fill=True, alpha=0.4,
                    label=f"Creative {data['label']}", ax=ax_hdi_cvr, color=colors[i])
        ax_hdi_cvr.axvline(hdi[0], color=colors[i], linestyle='--')
        ax_hdi_cvr.axvline(hdi[1], color=colors[i], linestyle='--',
                           label=f"{data['label']} 95% HDI: {hdi[0]:.3%} ~ {hdi[1]:.3%}")
    ax_hdi_cvr.set_title("Posterior Distributions 95% HDI (CVR)")
    ax_hdi_cvr.set_xlabel("CVR")
    ax_hdi_cvr.legend()
    st.pyplot(fig_hdi_cvr)


if __name__ == "__main__":
    run_ab_test()