# [Data] ANOVA(분산 분석)와 사후 분석의 심화 이해

## 1. ANOVA (Analysis of Variance)란?
세 개 이상의 집단(그룹) 간의 평균 차이가 통계적으로 유의미한지 확인하는 기법입니다.

* **핵심 질문**: "집단들 사이의 점수 차이가 단순한 우연일까, 아니면 정말로 집단 간에 실질적인 차이가 존재하는 것일까?"
* **F-검정 통계량**: (집단 간 분산 / 집단 내 분산)의 비율입니다. 이 값이 클수록 우연히 발생했을 확률인 $p-value$가 낮아집니다.
* **p-value (유의확률)**: 보통 **0.05**를 기준으로 하며, 이보다 작으면 "최소한 한 집단은 통계적으로 확실히 다르다"고 결론 내립니다.

---

## 2. 3개 반(A, B, C) 평균 비교 실습
우리가 직접 짠 코드는 3개 그룹의 평균을 설정하고, 그 차이를 통계량과 그래프로 증명하는 과정이었습니다.

### 📝 실습 코드 (`ANOVA_Analysis.py`)

```python
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 생성 (재현성을 위해 seed 고정)
np.random.seed(42)

# [실험 환경] 
# 실험 1(차이 큼): 60, 75, 80 
# 실험 2(차이 작음): 70, 72, 73
group_A = np.random.normal(60, 10, 30) 
group_B = np.random.normal(75, 10, 30) 
group_C = np.random.normal(80, 10, 30) 

# 데이터 프레임 생성 ('method'는 집단을 구분하는 이름표 역할)
df = pd.DataFrame({
    'score': np.concatenate([group_A, group_B, group_C]),
    'method': ['A'] * 30 + ['B'] * 30 + ['C'] * 30
})

# 2. ANOVA Test 실행
f_stat, p_val = stats.f_oneway(group_A, group_B, group_C)
print(f"ANOVA 결과: F={f_stat:.4f}, p-value={p_val:.8f}")

# 3. 사후 분석 (Tukey HSD) - 어떤 그룹끼리 다른지 일대일 비교
tukey = pairwise_tukeyhsd(df['score'], df['method'], alpha=0.05)
print(tukey)

# 4. 시각화 (Boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(x='method', y='score', data=df, hue='method', palette='Set2', legend=False)
sns.stripplot(x='method', y='score', data=df, color=".25", alpha=0.5) # 데이터 포인트 시각화
plt.title('Comparison of Test Scores by Study Method')
plt.show()
```

---

## 3. 실험 결과 비교 분석
실습을 통해 확인한 평균값의 거리에 따른 변화를 정리한 표입니다.

| 분석 지표 | 실험 1: 변별력 확실 | 실험 2: 변별력 모호 |
| :--- | :--- | :--- |
| **평균 설정** | 60, 75, 80 | 70, 72, 73 |
| **ANOVA p-value** | **0.0000... (유의미)** | **0.1255... (무의미)** |
| **Tukey (reject)** | **모두 True** | **모두 False** |
| **Boxplot 형태** | 상자들이 계단식으로 확실히 분리됨 | 상자들이 서로 위아래로 많이 겹침 |
| **데이터 해석** | 이 변수는 그룹 구분의 핵심 지표임 | 이 변수는 그룹 구분의 도움이 안 됨 |



---

## 4. 데이터 분석가의 인사이트 (TIL 정리 핵심)

* **p-value의 의미**: $p-value < 0.05$일 때, 해당 변수는 결과값을 예측하는 데 매우 유의미한 Feature(특성)가 됩니다.
* **사후 분석의 필연성**: ANOVA는 누군가 다르다는 사실만 알려줍니다. 실제 비즈니스 의사결정(예: 고객 등급별 맞춤 전략)을 위해서는 **Tukey HSD** 같은 사후 분석으로 구체적인 차이를 파악해야 합니다.
* **시각화와 통계의 결합**: 통계 수치만 보는 것보다 **Boxplot**으로 데이터의 겹침(Overlap) 정도를 확인하는 것이 데이터의 변별력을 판단하는 가장 직관적인 방법입니다.
