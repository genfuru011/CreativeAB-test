# A/Bテスト 統計検定アプリ

このアプリは、Streamlitを用いてA/Bテストにおけるクリック率（CTR）およびコンバージョン率（CVR）の統計検定（Z検定）を実施するためのツールです。ユーザーはサイドバーから各バリアントのインプレッション数、クリック数、コンバージョン数を入力し、各種統計検定の結果を確認できます。

特徴
	•	クリック率（CTR）の検定
インプレッション数に対するクリック数の差をZ検定により評価します。
	•	コンバージョン率（CVR）の検定
インプレッション数に対するコンバージョン数の差をZ検定により評価します。
	•	入力値のチェック
入力されたクリック数やコンバージョン数がインプレッション数を超えていないかを検証します。

必要な環境
	•	Python 3.x
	•	Streamlit
	•	statsmodels

インストール方法

以下の手順で環境をセットアップしてください。
	1.	Pythonのインストール
Pythonがインストールされていない場合は、公式サイトからインストールしてください。
	2.	仮想環境の作成（推奨）

python -m venv venv
source venv/bin/activate  # Windowsの場合は venv\Scripts\activate


	3.	必要なライブラリのインストール
以下のコマンドで必要なパッケージをインストールします。

pip install streamlit statsmodels



使い方
	1.	アプリのコード（例: app.py）を作成し、以下のコードを貼り付けます。

import streamlit as st
import statsmodels.api as sm

def run_ab_test():
    st.title("A/Bテスト：CTR & CVR の統計検定")
    
    # サイドバーで各バリアントのデータ入力
    st.sidebar.header("バリアントAのデータ")
    impressions_A = st.sidebar.number_input("インプレッション数 (A)", value=20000, step=1)
    clicks_A = st.sidebar.number_input("クリック数 (A)", value=1000, step=1)
    conversions_A = st.sidebar.number_input("コンバージョン数 (A)", value=100, step=1)
    
    st.sidebar.header("バリアントBのデータ")
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
    st.write("**バリアントA:** インプレッション: {}, クリック: {}, コンバージョン: {}"
             .format(impressions_A, clicks_A, conversions_A))
    st.write("**バリアントB:** インプレッション: {}, クリック: {}, コンバージョン: {}"
             .format(impressions_B, clicks_B, conversions_B))
    
    # クリック率 (CTR) の検定
    count_clicks = [clicks_A, clicks_B]
    nobs_impressions = [impressions_A, impressions_B]
    stat_click, p_click = sm.stats.proportions_ztest(count_clicks, nobs_impressions)
    
    # コンバージョン率 (CVR) の検定（コンバージョン数 ÷ インプレッション数）
    count_cv = [conversions_A, conversions_B]
    stat_cv, p_cv = sm.stats.proportions_ztest(count_cv, nobs_impressions)
    
    st.write("## 統計検定の結果")
    
    st.subheader("クリック率 (CTR) の検定")
    st.write("z値: {:.3f}".format(stat_click))
    st.write("p値: {:.3f}".format(p_click))
    if p_click < 0.05:
        st.success("CTR に統計的に有意な差があります")
    else:
        st.info("CTR に統計的な有意差は見られません")
    
    st.subheader("コンバージョン率 (CVR) の検定")
    st.write("z値: {:.3f}".format(stat_cv))
    st.write("p値: {:.3f}".format(p_cv))
    if p_cv < 0.05:
        st.success("CVR に統計的に有意な差があります")
    else:
        st.info("CVR に統計的な有意差は見られません")

if __name__ == "__main__":
    run_ab_test()


	2.	ターミナルで以下のコマンドを実行し、アプリを起動します。

streamlit run app.py


	3.	ブラウザが自動的に起動し、アプリのインターフェースが表示されます。
サイドバーから各バリアントのデータを入力し、統計検定の結果を確認してください。

カスタマイズ例
	•	検定方法の変更:
必要に応じて、片側検定や信頼区間の計算など、検定方法の選択肢を追加できます。
	•	データ入力の改善:
入力値のバリデーションやエラー処理をさらに充実させることで、より堅牢なアプリにすることができます。

ライセンス

このプロジェクトはMITライセンスのもとで公開されています。ライセンスの詳細はLICENSEファイルをご参照ください。
