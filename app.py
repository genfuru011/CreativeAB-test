import streamlit as st
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint, proportion_effectsize
from statsmodels.stats.power import NormalIndPower
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def run_ab_test():
    st.title("A/Bテスト：CTR & CVR の統計検定と拡張解析（ベイズ可視化付き）")

    # サイドバーで各クリエイティブのデータ入力
    st.sidebar.header("クリエイティブAのデータ")
    impressions_A = st.sidebar.number_input("インプレッション数 (A)", value=202787, step=1)
    clicks_A = st.sidebar.number_input("クリック数 (A)", value=1107, step=1)
    conversions_A = st.sidebar.number_input("コンバージョン数 (A)", value=19, step=1)

    st.sidebar.header("クリエイティブBのデータ")
    impressions_B = st.sidebar.number_input("インプレッション数 (B)", value=1401179, step=1)
    clicks_B = st.sidebar.number_input("クリック数 (B)", value=10505, step=1)
    conversions_B = st.sidebar.number_input("コンバージョン数 (B)", value=169, step=1)

    # 入力値の簡易チェック
    if clicks_A > impressions_A or clicks_B > impressions_B:
        st.error("クリック数はインプレッション数以下である必要があります。")
        return
    if conversions_A > impressions_A or conversions_B > impressions_B:
        st.error("コンバージョン数はインプレッション数以下である必要があります。")
        return

    st.write("## 入力されたデータ")
    st.write(f"**クリエイティブA:** インプレッション: {impressions_A}, クリック: {clicks_A}, コンバージョン: {conversions_A}")
    st.write(f"**クリエイティブB:** インプレッション: {impressions_B}, クリック: {clicks_B}, コンバージョン: {conversions_B}")

    # 指標の計算
    ctr_A = clicks_A / impressions_A
    ctr_B = clicks_B / impressions_B
    cvr_A = conversions_A / impressions_A
    cvr_B = conversions_B / impressions_B

    # Z検定（CTR, CVR）
    count_clicks = [clicks_A, clicks_B]
    nobs_impressions = [impressions_A, impressions_B]
    stat_click, p_click = sm.stats.proportions_ztest(count_clicks, nobs_impressions)

    count_cv = [conversions_A, conversions_B]
    stat_cv, p_cv = sm.stats.proportions_ztest(count_cv, nobs_impressions)

    st.write("## 統計検定の結果")

    # CTRの結果出力と優劣判定
    st.subheader("クリック率 (CTR) の検定")
    st.write("クリエイティブA CTR: {:.3%}".format(ctr_A))
    st.write("クリエイティブB CTR: {:.3%}".format(ctr_B))
    st.write("z値: {:.3f}".format(stat_click))
    st.write("p値: {:.3f}".format(p_click))
    if p_click < 0.05:
        if ctr_A > ctr_B:
            st.success("統計的に有意な差があり、クリエイティブAの方がCTRが高いです。")
        else:
            st.success("統計的に有意な差があり、クリエイティブBの方がCTRが高いです。")
    else:
        st.info("CTR に統計的な有意差は見られません。")

    # CVRの結果出力と優劣判定
    st.subheader("コンバージョン率 (CVR) の検定")
    st.write("クリエイティブA CVR: {:.3%}".format(cvr_A))
    st.write("クリエイティブB CVR: {:.3%}".format(cvr_B))
    st.write("z値: {:.3f}".format(stat_cv))
    st.write("p値: {:.3f}".format(p_cv))
    if p_cv < 0.05:
        if cvr_A > cvr_B:
            st.success("統計的に有意な差があり、クリエイティブAの方がCVRが高いです。")
        else:
            st.success("統計的に有意な差があり、クリエイティブBの方がCVRが高いです。")
    else:
        st.info("CVR に統計的な有意差は見られません。")

    # 1. 信頼区間 (Wilson法)
    ci_ctr_A = proportion_confint(clicks_A, impressions_A, alpha=0.05, method='wilson')
    ci_ctr_B = proportion_confint(clicks_B, impressions_B, alpha=0.05, method='wilson')
    ci_cvr_A = proportion_confint(conversions_A, impressions_A, alpha=0.05, method='wilson')
    ci_cvr_B = proportion_confint(conversions_B, impressions_B, alpha=0.05, method='wilson')

    st.write("## 信頼区間の計算 (95% CI, Wilson法)")
    st.write("**CTR**")
    st.write("A: {:.3%} ~ {:.3%}".format(ci_ctr_A[0], ci_ctr_A[1]))
    st.write("B: {:.3%} ~ {:.3%}".format(ci_ctr_B[0], ci_ctr_B[1]))
    st.write("**CVR**")
    st.write("A: {:.3%} ~ {:.3%}".format(ci_cvr_A[0], ci_cvr_A[1]))
    st.write("B: {:.3%} ~ {:.3%}".format(ci_cvr_B[0], ci_cvr_B[1]))

    # 2. 効果サイズ (差分・相対変化率)
    delta_ctr = ctr_A - ctr_B
    relative_change_ctr = (delta_ctr / ctr_B) if ctr_B != 0 else np.nan
    delta_cvr = cvr_A - cvr_B
    relative_change_cvr = (delta_cvr / cvr_B) if cvr_B != 0 else np.nan

    st.write("## 効果サイズの評価")
    st.write("**CTRの効果サイズ**")
    st.write("CTRの差分: {:.3%}".format(delta_ctr))
    st.write("相対変化率: {:.1f}%".format(relative_change_ctr * 100))
    st.write("**CVRの効果サイズ**")
    st.write("CVRの差分: {:.3%}".format(delta_cvr))
    st.write("相対変化率: {:.1f}%".format(relative_change_cvr * 100))



    # 4. サンプルサイズ・パワー分析
    st.write("## サンプルサイズ・パワー分析 (有意水準5%, 検出力80%)")
    analysis = NormalIndPower()
    # CTR 用の効果サイズ
    try:
        effect_size_ctr = proportion_effectsize(ctr_A, ctr_B)
        required_n_ctr = analysis.solve_power(effect_size=effect_size_ctr, alpha=0.05, power=0.8, ratio=1)
        st.write(f"CTRテストで必要なサンプルサイズ（各群）: {required_n_ctr:,.0f}")
    except Exception as e:
        st.write("CTRサンプルサイズ計算に失敗しました: ", e)

    # CVR 用の効果サイズ
    try:
        effect_size_cvr = proportion_effectsize(cvr_A, cvr_B)
        required_n_cvr = analysis.solve_power(effect_size=effect_size_cvr, alpha=0.05, power=0.8, ratio=1)
        st.write(f"CVRテストで必要なサンプルサイズ（各群）: {required_n_cvr:,.0f}")
    except Exception as e:
        st.write("CVRサンプルサイズ計算に失敗しました: ", e)

    # 5. ベイズ統計による代替検定 (Beta分布 + 事後分布の可視化)
    st.write("## ベイズ推定 (Beta分布)")

    n_samples = 100000
    # CTRのベイズ的推定
    posterior_A_ctr = np.random.beta(clicks_A + 1, impressions_A - clicks_A + 1, n_samples)
    posterior_B_ctr = np.random.beta(clicks_B + 1, impressions_B - clicks_B + 1, n_samples)
    prob_B_better_ctr = np.mean(posterior_B_ctr > posterior_A_ctr)

    # CVRのベイズ的推定
    posterior_A_cvr = np.random.beta(conversions_A + 1, impressions_A - conversions_A + 1, n_samples)
    posterior_B_cvr = np.random.beta(conversions_B + 1, impressions_B - conversions_B + 1, n_samples)
    prob_B_better_cvr = np.mean(posterior_B_cvr > posterior_A_cvr)

    st.write(f"**CTR**: クリエイティブBがAより高い確率: {prob_B_better_ctr*100:.1f}%")
    st.write(f"**CVR**: クリエイティブBがAより高い確率: {prob_B_better_cvr*100:.1f}%")

    # 5-1. 事後分布の可視化
    st.write("### 事後分布の可視化(ベイズ推定） (CTR, CVR)")

    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    # CTR分布
    sns.kdeplot(posterior_A_ctr, fill=True, alpha=0.4, label='A', ax=ax3, color='blue')
    sns.kdeplot(posterior_B_ctr, fill=True, alpha=0.4, label='B', ax=ax3, color='green')
    ax3.set_title("Posterior Distributions (CTR)")
    ax3.set_xlabel("CTR")
    ax3.legend()

    # CVR分布
    sns.kdeplot(posterior_A_cvr, fill=True, alpha=0.4, label='A', ax=ax4, color='red')
    sns.kdeplot(posterior_B_cvr, fill=True, alpha=0.4, label='B', ax=ax4, color='gold')
    ax4.set_title("Posterior Distributions (CVR)")
    ax4.set_xlabel("CVR")
    ax4.legend()

    st.pyplot(fig3)

    def compute_hdi(samples, cred_mass=0.95):
        """
        与えられたサンプルからHDIを計算する関数。
        samples: 1次元のサンプル配列
        cred_mass: 信頼度（例: 0.95）
        """
        sorted_samples = np.sort(samples)
        n_samples = len(sorted_samples)
        interval_idx_inc = int(np.floor(cred_mass * n_samples))
        n_intervals = n_samples - interval_idx_inc
        interval_width = sorted_samples[interval_idx_inc:] - sorted_samples[:n_intervals]
        min_idx = np.argmin(interval_width)
        hdi_lower = sorted_samples[min_idx]
        hdi_upper = sorted_samples[min_idx + interval_idx_inc]
        return hdi_lower, hdi_upper

    # 例: CTRの事後分布を用いて95% HDIを計算
    hdi_ctr_A = compute_hdi(posterior_A_ctr, 0.95)
    hdi_ctr_B = compute_hdi(posterior_B_ctr, 0.95)

    # HDIの表示とグラフへの重ね描き
    fig_hdi, ax_hdi = plt.subplots(figsize=(8, 4))
    sns.kdeplot(posterior_A_ctr, fill=True, alpha=0.4, label='Creative A', ax=ax_hdi, color='blue')
    sns.kdeplot(posterior_B_ctr, fill=True, alpha=0.4, label='Creative B', ax=ax_hdi, color='green')
    # HDIをラインで表示
    ax_hdi.axvline(hdi_ctr_A[0], color='blue', linestyle='--')
    ax_hdi.axvline(hdi_ctr_A[1], color='blue', linestyle='--',
                   label=f"A 95% HDI: {hdi_ctr_A[0]:.3%}~{hdi_ctr_A[1]:.3%}")
    ax_hdi.axvline(hdi_ctr_B[0], color='green', linestyle='--')
    ax_hdi.axvline(hdi_ctr_B[1], color='green', linestyle='--',
                   label=f"B 95% HDI: {hdi_ctr_B[0]:.3%}~{hdi_ctr_B[1]:.3%}")
    ax_hdi.set_title("Posterior Distributions 95% HDI (CTR)")
    ax_hdi.set_xlabel("CTR")
    ax_hdi.legend()
    st.pyplot(fig_hdi)

    # CVR版 HDI の計算
    hdi_cvr_A = compute_hdi(posterior_A_cvr, 0.95)
    hdi_cvr_B = compute_hdi(posterior_B_cvr, 0.95)

    # HDIの表示とグラフへの重ね描き（CVR版）
    fig_hdi_cvr, ax_hdi_cvr = plt.subplots(figsize=(8, 4))
    sns.kdeplot(posterior_A_cvr, fill=True, alpha=0.4, label='Creative A', ax=ax_hdi_cvr, color='red')
    sns.kdeplot(posterior_B_cvr, fill=True, alpha=0.4, label='Creative B', ax=ax_hdi_cvr, color='gold')

    # CVRのHDIをラインで表示
    ax_hdi_cvr.axvline(hdi_cvr_A[0], color='red', linestyle='--')
    ax_hdi_cvr.axvline(hdi_cvr_A[1], color='red', linestyle='--',
                       label=f"A 95% HDI: {hdi_cvr_A[0]:.3%} ~ {hdi_cvr_A[1]:.3%}")
    ax_hdi_cvr.axvline(hdi_cvr_B[0], color='gold', linestyle='--')
    ax_hdi_cvr.axvline(hdi_cvr_B[1], color='gold', linestyle='--',
                       label=f"B 95% HDI: {hdi_cvr_B[0]:.3%} ~ {hdi_cvr_B[1]:.3%}")
    ax_hdi_cvr.set_title("Posterior Distributions 95% HDI (CVR)")
    ax_hdi_cvr.set_xlabel("CVR")
    ax_hdi_cvr.legend()
    st.pyplot(fig_hdi_cvr)


if __name__ == "__main__":
    run_ab_test()