## 1단계 : ETF 데이터 불러오기
import pandas as pd

# 1. ETF 데이터 불러오기
df = pd.read_csv('data/etf_list.csv', encoding = "utf-8-sig")

# 2. 데이터 미리보기
print("[INFO] 데이터 크기 :", df.shape)
print(df.head(5))





## 2.단계 : 시가총액 상위 5개 ETF 시각화
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Malgun Gothic' # 한글 폰트 지정
mpl.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# 숫자형 변환
df['시가총액(억)'] = pd.to_numeric(df['시가총액(억)'], errors = "coerce")

# 시가총액 상위 5개 ETF
top5 = df.sort_values("시가총액(억)", ascending = False).head(5)
print("\n[INFO] 시가총액 상위 5개 ETF:")
print(top5[["종목명", "시가총액(억)"]])

# 막대그래프 시각화
plt.figure(figsize = (8, 5))
plt.bar(top5['종목명'], top5['시가총액(억)'], color = 'skyblue')
plt.title("시가총액 상위 5개 ETF", fontsize = 14)
plt.xlabel("ETF 종목명")
plt.ylabel("시가총액(억)")
plt.xticks(rotation = 30, ha = 'right')
plt.ylim(0, top5["시가총액(억)"].max() * 1.1) # 위쪽 여백 확보

# 막대 위에 수치 표시
for i, v in enumerate(top5["시가총액(억)"]):
    plt.text(i, v + 2000, f"{v:,.0f}", ha = 'center', va = 'bottom', fontsize = 9, color = 'black', fontweight = 'bold')

plt.tight_layout()
plt.subplots_adjust(top = 0.9) # 전체 그래프 여백 조정
plt.show()




## 3단계 : 3개월 수익률(%) TOP5 ETF 분석
df["3개월수익률(%)"] = pd.to_numeric(df["3개월수익률(%)"], errors = "coerce")

top5_return = df.sort_values("3개월수익률(%)", ascending = False).head(5)
print("\n[INFO] 3개월 수익률 상위 5개 ETF:")
print(top5_return[["종목명", "3개월수익률(%)"]])

# 수익률 시각화
plt.figure(figsize = (8, 5))
plt.bar(top5_return['종목명'], top5_return['3개월수익률(%)'], color = 'salmon')
plt.title("3개월 수익률 상위 5개 ETF", fontsize = 14)
plt.xlabel("ETF 종목명")
plt.ylabel("3개월 수익률(%)")
plt.xticks(rotation = 30, ha = 'right')
plt.ylim(0, top5_return["3개월수익률(%)"].max() * 1.1) # 위쪽 여백 확보

# 막대 위에 수치 표시
for i, v in enumerate(top5_return["3개월수익률(%)"]):
    plt.text(i, v + (v * 0.02), f"{v:,.2f}", ha = 'center', va = 'bottom', fontsize = 9, color = 'black', fontweight = 'bold')

plt.tight_layout()
plt.subplots_adjust(top = 0.9) # 전체 그래프 여백 조정
plt.show()




## 4단계 : ETF 전체의 3개월 수익률 분포(히스토그램)
import numpy as np

returns = pd.to_numeric(df["3개월수익률(%)"], errors = "coerce").dropna()

plt.figure(figsize = (8, 5))
plt.hist(returns, bins =30, color = "skyblue", edgecolor = "black", alpha = 0.7)
plt.title("ETF 3개월 수익률 분포", fontsize = 14)
plt.xlabel("3개월 수익률(%)")
plt.ylabel("ETF 개수")
plt.axvline(returns.mean(), color = 'red', linestyle = '--', label = f'평균 {returns.mean():.2f}%')
plt.legend()
plt.tight_layout()
plt.show()

print(f"[INFO] 평균 수익률: {returns.mean():.2f}%")
print(f"[INFO] 중앙값 수익률: {returns.median():.2f}%")
print(f"[INFO] 표준편차: {returns.std():.2f}")




## 5단계 : 시가총액 vs 3개월 수익률 산점도(scatter plot)
import matplotlib.pyplot as plt

df["시가총액(억)"] = pd.to_numeric(df["시가총액(억)"], errors = "coerce")
df["3개월수익률(%)"] = pd.to_numeric(df["3개월수익률(%)"], errors = "coerce")

# 결측치 제거
filtered = df.dropna(subset = ["시가총액(억)", "3개월수익률(%)"])

plt.figure(figsize = (8, 6))
plt.scatter(filtered["3개월수익률(%)"], filtered["시가총액(억)"], 
            color = "mediumseagreen", alpha = 0.6, edgecolors = "black", linewidths = 0.5)
plt.title("3개월 수익률 vs 시가총액", fontsize = 14)
plt.xlabel("3개월 수익률(%)")
plt.ylabel("시가총액(억)")
plt.grid(True, linestyle = "--", alpha = 0.5)

# 상관계수 계산
corr = filtered["3개월수익률(%)"].corr(filtered["시가총액(억)"])
plt.text(filtered["3개월수익률(%)"].max() * 1.1, filtered["시가총액(억)"].max() * 1.1,
         f"상관계수 r = {corr:.3f}", fontsize = 11, color = "red")

plt.tight_layout()
plt.show()

print(f"[INFO] 3개월 수익률과 시가총액의 상관계수 : {corr:.3f}")





## 6단계 : ETF 지표 간 상관관계 분석 (Heatmap 시각화)
# 주요 수치들 간에 서로 얼마나 연관되어 있는지 확인
import seaborn as sns
import matplotlib.pyplot as plt

# 분석에 사용할 숫자형 컬럼 선택
numeric_cols = ["시가총액(억)", "거래대금", "등락률(%)", "3개월수익률(%)", "NAV"]

# 숫자형으로 변환
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors = "coerce")
    
corr = df[numeric_cols].corr().round(2)
print("[INFO] 상관계수 행렬:")
print(corr)

# 히트맵 시각화
plt.figure(figsize = (7, 5))
sns.heatmap(corr, annot = True, cmap = "coolwarm", center = 0, linewidths = 0.5, fmt = ".2f")
plt.title("ETF 주요 지표 간 상관관계 Heatmap", fontsize = 14)
plt.tight_layout()
plt.show()




## 7단계 : K-Means 클러스터링
# ETF를 특성별로 그룹화 진행
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 사용할 지표 선택
cols = ["시가총액(억)", "거래대금", "3개월수익률(%)", "NAV"]

# 숫자형 변환 + 결측치 제거
for c in cols:
    df[c] = pd.to_numeric(df[c], errors = "coerce")
X = df[cols].dropna()

# 데이터 표준화 (평균 0, 표준편차 1)
scaler = StandardScaler() # 변수 단위가 달라 변수 단위 맞춰주는 작업
X_scaled = scaler.fit_transform(X)

# KMean로 3개 클러스터 분류
kmeans = KMeans(n_clusters = 3, random_state = 42, n_init = 10)
labels = kmeans.fit_predict(X_scaled)

# 결과를 원본 df에 합치기
df_clustered = df.loc[X.index].copy()
df_clustered["클러스터"] = labels

print("[INFO] 각 클러스터별 ETF 개수:")
print(df_clustered["클러스터"].value_counts())

# 시각화 (시가총액 vs 3개월 수익률)
plt.figure(figsize = (8, 6))
for cluster_id in sorted(df_clustered["클러스터"].unique()):
    cluster_data = df_clustered[df_clustered["클러스터"] == cluster_id]
    plt.scatter(cluster_data["3개월수익률(%)"], cluster_data["시가총액(억)"], 
                label = f"Cluster {cluster_id}", alpha = 0.6, edgecolors = "black")

plt.title("K-Means 클러스터링: 3개월 수익률 vs 시가총액", fontsize = 14)
plt.xlabel("3개월 수익률(%)")
plt.ylabel("시가총액(억)")
plt.legend()
plt.grid(True, linestyle = "--", alpha = 0.5)
plt.tight_layout()
plt.show()




