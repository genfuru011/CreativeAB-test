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

    num_creatives = st.sidebar.number_input("クリエイティブの数", min_value=2, value=2, step=1)
    creatives_data = []
    group_labels = []

    for i in range(num_creatives):
        label = chr(65 + i)
        group_labels.append(label)
        st.sidebar.header(f"クリエイティブ {label} のデータ")
        impressions = st.sidebar.number_input(f"インプレッション数 ({label})", value=100000, step=1, key=f"imp_{label}")
        clicks = st.sidebar.number_input(f"クリック数 ({label})", value=1000, step=1, key=f"click_{label}")
        conversions = st.sidebar.number_input(f"コンバージョン数 ({label})", value=50, step=1, key=f"conv_{label}")
        creatives_data.append({"label": label, "impressions": impressions, "clicks": clicks, "conversions": conversions})

    for data in creatives_data:
        if data["clicks"] > data["impressions"]:
            st.error(f"クリエイティブ {data['label']} のクリック数はインプレッション数以下である必要があります。")
            return
        if data["conversions"] > data["clicks"]:
            st.error(f"クリエイティブ {data['label']} のコンバージョン数はクリック数以下である必要があります。")
            return

    st.write("## 入力されたデータ")
    df = pd.DataFrame(creatives_data)
    st.dataframe(df)

    for data in creatives_data:
        data["ctr"] = data["clicks"] / data["impressions"] if data["impressions"] > 0 else 0
        data["cvr"] = data["conversions"] / data["clicks"] if data["clicks"] > 0 else 0

    st.write("## 各クリエイティブのCTRとCVR")
    for data in creatives_data:
        st.write(f"クリエイティブ {data['label']}: CTR = {data['ctr']:.3%}, CVR = {data['cvr']:.3%}")

    table_ctr = [[d["clicks"], d["impressions"] - d["clicks"]] for d in creatives_data]
    chi2_ctr, p_ctr, _, _ = chi2_contingency(table_ctr)
    st.write("## CTR の全体的な検定結果 (Chi-squared test)")
    st.write(f"Chi2値: {chi2_ctr:.3f}, p値: {p_ctr:.3f}")
    if p_ctr < 0.05:
        best = max(creatives_data, key=lambda d: d["ctr"])
        st.success(f"統計的に有意な差があり、最高のCTRはクリエイティブ {best['label']} です。")
    else:
        st.info("CTR に統計的な有意差は見られません。")

    table_cvr = [[d["conversions"], d["clicks"] - d["conversions"]] for d in creatives_data]
    chi2_cvr, p_cvr, _, _ = chi2_contingency(table_cvr)
    st.write("## CVR の全体的な検定結果 (Chi-squared test)")
    st.write(f"Chi2値: {chi2_cvr:.3f}, p値: {p_cvr:.3f}")
    if p_cvr < 0.05:
        best = max(creatives_data, key=lambda d: d["cvr"])
        st.success(f"統計的に有意な差があり、最高のCVRはクリエイティブ {best['label']} です。")
    else:
        st.info("CVR に統計的な有意差は見られません。")

    st.write("## 信頼区間の計算 (95% CI, Wilson法)")
    st.write("**CTR**")
    for d in creatives_data:
        ci = proportion_confint(d["clicks"], d["impressions"], alpha=0.05, method='wilson')
        st.write(f"{d['label']}: {ci[0]:.3%} ~ {ci[1]:.3%}")
    st.write("**CVR**")
    for d in creatives_data:
        ci = proportion_confint(d["conversions"], d["clicks"], alpha=0.05, method='wilson')
        st.write(f"{d['label']}: {ci[0]:.3%} ~ {ci[1]:.3%}")

    st.write("## 効果サイズの評価 (ペアワイズ比較)")
    comparisons = []
    for i in range(len(creatives_data)):
        for j in range(i + 1, len(creatives_data)):
            a, b = creatives_data[i], creatives_data[j]
            comparisons.append({
                "比較": f"{a['label']} vs {b['label']}",
                "CTR 差分": a["ctr"] - b["ctr"],
                "CTR 相対変化率": (a["ctr"] - b["ctr"]) / b["ctr"] if b["ctr"] != 0 else np.nan,
                "CVR 差分": a["cvr"] - b["cvr"],
                "CVR 相対変化率": (a["cvr"] - b["cvr"]) / b["cvr"] if b["cvr"] != 0 else np.nan
            })
    st.dataframe(pd.DataFrame(comparisons).style.format({
        "CTR 差分": "{:.3%}",
        "CTR 相対変化率": "{:.1%}",
        "CVR 差分": "{:.3%}",
        "CVR 相対変化率": "{:.1%}"
    }))

    st.write("## サンプルサイズ・パワー分析")
    analysis = NormalIndPower()
    power_results = []
    for i in range(len(creatives_data)):
        for j in range(i + 1, len(creatives_data)):
            a, b = creatives_data[i], creatives_data[j]
            try:
                es_ctr = proportion_effectsize(a["ctr"], b["ctr"])
                n_ctr = analysis.solve_power(effect_size=es_ctr, alpha=0.05, power=0.8, ratio=1)
            except:
                n_ctr = None
            try:
                es_cvr = proportion_effectsize(a["cvr"], b["cvr"])
                n_cvr = analysis.solve_power(effect_size=es_cvr, alpha=0.05, power=0.8, ratio=1)
            except:
                n_cvr = None
            power_results.append({"比較": f"{a['label']} vs {b['label']}", "CTR 必要サンプル数": n_ctr, "CVR 必要サンプル数": n_cvr})
    st.dataframe(pd.DataFrame(power_results).style.format({
        "CTR 必要サンプル数": "{:,.0f}",
        "CVR 必要サンプル数": "{:,.0f}"
    }))

    st.write("## ベイズ推定 (Beta分布)")
    n_samples = 500000
    for d in creatives_data:
        d["posterior_ctr"] = np.random.beta(d["clicks"] + 2, d["impressions"] - d["clicks"] + 98, n_samples)
        d["posterior_cvr"] = np.random.beta(d["conversions"] + 1, d["clicks"] - d["conversions"] + 1, n_samples)

    def show_prob_best(metric):
        all_post = np.array([d[f"posterior_{metric}"] for d in creatives_data])
        best_counts = np.argmax(all_post, axis=0)
        st.write(f"### {metric.upper()}: 各クリエイティブが最高である確率")
        for idx, label in enumerate(group_labels):
            st.write(f"クリエイティブ {label}: {np.mean(best_counts == idx) * 100:.1f}%")

    show_prob_best("ctr")
    show_prob_best("cvr")

    def plot_distributions(metric):
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, d in enumerate(creatives_data):
            sns.kdeplot(d[f"posterior_{metric}"], fill=True, alpha=0.4, label=f"Creative {d['label']}", ax=ax)
        ax.set_title(f"Posterior Distributions ({metric.upper()})")
        ax.set_xlabel(metric.upper())
        ax.legend()
        st.pyplot(fig)

    plot_distributions("ctr")
    plot_distributions("cvr")

    def compute_hdi(samples, cred_mass=0.95):
        sorted_samples = np.sort(samples)
        interval_idx_inc = int(np.floor(cred_mass * len(sorted_samples)))
        interval_width = sorted_samples[interval_idx_inc:] - sorted_samples[:len(sorted_samples) - interval_idx_inc]
        min_idx = np.argmin(interval_width)
        return sorted_samples[min_idx], sorted_samples[min_idx + interval_idx_inc]

    def plot_hdi(metric):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, d in enumerate(creatives_data):
        hdi = compute_hdi(d[f"posterior_{metric}"], 0.95)

        # KDEプロットして、その色を取得
        kdeplot_result = sns.kdeplot(
            d[f"posterior_{metric}"],
            fill=True,
            alpha=0.4,
            label=f"Creative {d['label']}",
            ax=ax
        )

        # get_lines() は Axes に付属する全てのLine2Dを返す
        # 最後に追加された線（kdeの輪郭線）から色を取得する
        lines = ax.get_lines()
        color = lines[-1].get_color() if lines else 'black'

        ax.axvline(hdi[0], color=color, linestyle='--')
        ax.axvline(hdi[1], color=color, linestyle='--', label=f"{d['label']} 95% HDI: {hdi[0]:.3%} ~ {hdi[1]:.3%}")
    
    ax.set_title(f"Posterior Distributions 95% HDI ({metric.upper()})")
    ax.set_xlabel(metric.upper())
    ax.legend()
    st.pyplot(fig)


if __name__ == "__main__":
    run_ab_test()
