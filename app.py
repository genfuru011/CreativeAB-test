import streamlit as st
import statsmodels.api as sm

def run_ab_test():
    st.title("A/Bテスト：CTR & CVR の統計検定")

    # サイドバーで各クリエイティブのデータ入力
    st.sidebar.header("クリエイティブAのデータ")
    impressions_A = st.sidebar.number_input("インプレッション数 (A)", value=20000, step=1)
    clicks_A = st.sidebar.number_input("クリック数 (A)", value=1000, step=1)
    conversions_A = st.sidebar.number_input("コンバージョン数 (A)", value=100, step=1)

    st.sidebar.header("クリエイティブBのデータ")
    impressions_B = st.sidebar.number_input("インプレッション数 (B)", value=18000, step=1)
    clicks_B = st.sidebar.number_input("クリック数 (B)", value=800, step=1)
    conversions_B = st.sidebar.number_input("コンバージョン数 (B)", value=90, step=1)

    # 入力値の簡易チェック
    if clicks_A > impressions_A or clicks_B > impressions_B:
        st.error("クリック数はインプレッション数以下である必要があります。")
        return
    if conversions_A > impressions_A or conversions_B > impressions_B:
        st.error("コンバージョン数はインプレッション数以下である必要があります。")
        return

    st.write("## 入力されたデータ")
    st.write("**クリエイティブA:** インプレッション: {}, クリック: {}, コンバージョン: {}"
             .format(impressions_A, clicks_A, conversions_A))
    st.write("**クリエイティブB:** インプレッション: {}, クリック: {}, コンバージョン: {}"
             .format(impressions_B, clicks_B, conversions_B))

    # クリック率 (CTR) の計算
    ctr_A = clicks_A / impressions_A
    ctr_B = clicks_B / impressions_B

    # コンバージョン率 (CVR) の計算（コンバージョン数 ÷ インプレッション数）
    cvr_A = conversions_A / impressions_A
    cvr_B = conversions_B / impressions_B

    # クリック率 (CTR) の検定
    count_clicks = [clicks_A, clicks_B]
    nobs_impressions = [impressions_A, impressions_B]
    stat_click, p_click = sm.stats.proportions_ztest(count_clicks, nobs_impressions)

    # コンバージョン率 (CVR) の検定
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
        # 統計的有意差がある場合、どちらが優れているか判定
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
        # 統計的有意差がある場合、どちらが優れているか判定
        if cvr_A > cvr_B:
            st.success("統計的に有意な差があり、クリエイティブAの方がCVRが高いです。")
        else:
            st.success("統計的に有意な差があり、クリエイティブBの方がCVRが高いです。")
    else:
        st.info("CVR に統計的な有意差は見られません。")


if __name__ == "__main__":
    run_ab_test()
