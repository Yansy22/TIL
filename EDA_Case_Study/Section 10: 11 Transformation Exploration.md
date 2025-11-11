# Section 10. 변수 변환 탐색 (Transformation Exploration)

이 섹션의 목적은 "모델의 성능과 안정성을 높이기 위해, 수치형 피처(데이터)를 어떤 '모양'으로 가공(변환)하는 것이 가장 최적인가?"라는 질문에 답하는 것입니다.

**핵심 전제:** 많은 **선형 모델(Logistic Regression 등)** 은 데이터가 **정규 분포(종 모양)** 라고 가정합니다. 데이터가 한쪽으로 심하게 쏠려있으면(Skewed), 모델이 불안정해지고 성능이 저하될 수 있습니다.

이 섹션은 "진단 → 치료 → 검증"의 5단계로 진행됩니다.

---

## 1단계: 🩺 진단 (Analyze original distributions)

* **목적:** "현재 데이터의 건강 상태(분포)는 어떠한가?"를 진단합니다.
* **방법:** `stats.normaltest` (p-value), `skew` (왜도/쏠림), `kurt` (첨도/뾰족함)를 **숫자**로 계산하고, 히스토그램(실제)과 정규분포(이상)를 **그림**으로 비교합니다.
* **코드 예시:**
    ```python
    # 1. 숫자로 진단 (정규성 검정)
    for feature in numeric_features:
        statistic, p_value = stats.normaltest(train_df[feature])
        skew = train_df[feature].skew()
        # ... (distribution_stats DataFrame에 저장) ...
    print(distribution_stats.round(4))

    # 2. 그림으로 진단 (시각화)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, feature in enumerate(numeric_features):
        data = train_df[feature]
        # 실제 분포 (파란색 막대)
        axes[idx].hist(data, bins=30, density=True, alpha=0.7, color='skyblue')
        # 이상적인 정규분포 (빨간색 선)
        axes[idx].plot(x, stats.norm.pdf(x, data.mean(), data.std()), 'r-', label='Normal')
    ```
* **💡 인사이트:**
    * `Is_Normal: False`가 대부분이며, 히스토그램(파란색)이 정규분포(빨간색)와 다릅니다.
    * **결론:** 데이터가 한쪽으로 쏠려(Skewed) 있으므로, **"치료(Transformation)"가 필요합니다.**

---

## 2단계: 💊 수동 치료 (Manual Transformations)

* **목적:** "1차 진료". `log`, `sqrt`, `square` 등 **"일반적인 변환(치료법)"** 7가지를 수동으로 테스트하여 "쏠림(Skewness)"을 가장 잘 고치는 변환을 찾습니다.
* **방법:** 7가지 변환을 각각 적용한 뒤, `'Combined_Score': abs(skew) + abs(kurt) / 10` (0에 가까울수록 좋음)라는 심사 점수로 1등을 뽑습니다.

* **코드 예시:**
    ```python
    transformations = {
        'Original': lambda x: x,
        'Log': lambda x: np.log1p(x),
        'Square Root': lambda x: np.sqrt(x),
        # ... (Square, Reciprocal 등) ...
    }
    
    # 7가지 변환을 모두 테스트하고 Combined_Score 계산
    for feature in numeric_features:
        for trans_name, trans_func in transformations.items():
            # ... (try-except로 skew, kurt 계산) ...
            transformation_results.append({ ... 'Combined_Score': ... })

    # 점수가 가장 낮은(가장 정규분포에 가까운) 변환을 찾음
    best_transformations = results_df.loc[results_df.groupby('Feature')['Combined_Score'].idxmin()]
    print(best_transformations.round(3))
    ```
* **💡 인사이트:**
    * `age`는 `Cube Root`(세제곱근)일 때, `tenure`는 `Square`(제곱)일 때 '이론적으로' 가장 대칭적인 모양이 됨을 확인합니다.

---

## 3단계: 🔬 자동 최적화 (Box-Cox and Yeo-Johnson)

* **목적:** "2차 정밀 진료". 2단계의 '수동' 방식을 넘어, `Box-Cox`와 `Yeo-Johnson`이라는 **"전문 변환 알고리즘"** 을 사용해 수학적으로 최적화된 $\lambda$(람다) 값을 **자동으로** 찾습니다.
* **방법:**
    * **Box-Cox:** 전통적인 자동 변환기. (단, 0 또는 음수 값이 있으면 실패)
    * **Yeo-Johnson:** Box-Cox의 업그레이드 버전. (0/음수 값도 처리 가능)
* **코드 예시:**
    ```python
    fig, axes = plt.subplots(len(numeric_features), 3, ...)
    for idx, feature in enumerate(numeric_features):
        data = train_df[feature].values
        
        # 1. Original (파란색)
        axes[idx, 0].hist(data, ...)
        
        # 2. Box-Cox (초록색) - 0 값이 있으면 실패 (Not applicable)
        if (data > 0).all():
            transformed_bc, lambda_bc = boxcox(data)
            axes[idx, 1].hist(transformed_bc, ...)
        
        # 3. Yeo-Johnson (빨간색) - 항상 성공
        pt = PowerTransformer(method='yeo-johnson')
        transformed_yj = pt.fit_transform(data.reshape(-1, 1)).ravel()
        axes[idx, 2].hist(transformed_yj, ...)
    ```
* **결과 및 해석:**
    
    * **시각적:** `payment_interval`처럼 쏠려있던 원본(파란색)이 `Yeo-Johnson`(빨간색)을 통해 **대칭적인 모양**으로 성공적으로 "치료"되는 것을 눈으로 확인합니다.
    * **안정성:** `Box-Cox`는 0 값이 포함된 피처에서 실패("Not applicable")합니다.
    * **💡 인사이트:** "쏠림(Skewness)"을 잡는 가장 안정적이고 강력한 "처방전"은 **Yeo-Johnson** 변환임을 확인합니다.

---

## 4단계: 🚑 강제 변환 (Quantile Transformation)

* **목적:** "최종 수술". 3단계 `Yeo-Johnson`으로도 "쏠림"은 잡았지만 "울퉁불퉁함(다봉성)"을 잡지 못했을 경우, **"원본의 모양을 무시하고 강제로 완벽한 분포를 빚어내는"** 가장 공격적인 변환입니다.
* **방법:** 데이터의 '값' 대신 '순위(Quantile)'를 사용합니다.
    1.  `output_distribution='normal'` (초록색): 데이터를 **강제로 완벽한 '종 모양'** 으로 재배치합니다.
    2.  `output_distribution='uniform'` (빨간색): 데이터를 **강제로 완벽한 '평지'** 로 재배치합니다.
* **코드 예시:**
    ```python
    qt_normal = QuantileTransformer(output_distribution='normal')
    qt_uniform = QuantileTransformer(output_distribution='uniform')

    for idx, feature in enumerate(numeric_features[:3]):
        # 'normal' (종 모양)으로 강제 변환
        transformed_normal = qt_normal.fit_transform(data).ravel()
        axes[0, idx].hist(transformed_normal, ...)
        
        # 'uniform' (평지 모양)으로 강제 변환
        transformed_uniform = qt_uniform.fit_transform(data).ravel()
        axes[1, idx].hist(transformed_uniform, ...)
    ```
* **💡 인사이트:**
    * 이 변환은 모델의 안정성을 극대화할 수 있지만, 원본 데이터의 의미(예: 50세와 51세의 1살 차이)를 **완전히 파괴**합니다.
    * "성능"을 위해 "해석"을 희생할 때 사용하는 최종 카드입니다.

---

## 5단계: 🏁 최종 실전 검증 (Performance Test)

* **목적:** "그래서... 1~4번의 이론적인 변환 중, **'실제로' 모델 성능에 가장 도움이 되는 변환은 무엇인가?"**에 대한 **최종 답**을 찾는 과정입니다.
* **방법:**
    1.  **전역 테스트 (Global):** "모든 피처에 'Original' 적용" vs "모S든 피처에 'Yeo-Johnson' 적용" 등 5가지 **전략**을 `RandomForestClassifier`로 5-겹 교차 검증하여 **F1-Score(성능)** 를 비교합니다.
    2.  **개별 테스트 (Individual):** 각 피처별로 6가지 변환을 적용하여, **타겟(`y`)과의 관계(MI Score)** 가 가장 높아지는 '최적의 변환'을 찾습니다.
* **코드 예시 (전역 테스트):**
    ```python
    transformation_pipelines = {
        'Original': StandardScaler(),
        'Yeo-Johnson': PowerTransformer(method='yeo-johnson'),
        'Quantile-Normal': QuantileTransformer(output_distribution='normal'),
        # ...
    }
    
    for trans_name, transformer in transformation_pipelines.items():
        pipeline = Pipeline([('transform', transformer), ('model', RandomForestClassifier(...))])
        scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_macro')
        # ... (결과 저장) ...

    performance_df.plot(kind='bar', y='Mean_F1', ...)
    ```
* **코드 예시 (개별 테스트):**
    ```python
    for feature in numeric_features:
        transforms_to_test = { ... }
        for trans_name, transformed_data in transforms_to_test.items():
            # 타겟(y)과의 MI Score 계산
            mi_score = mutual_info_classif(transformed_data, y, ...)[0]
            # ... (최고 점수 찾기) ...
    
    ax.bar(features, improvements, color=colors, ...)
    ```
