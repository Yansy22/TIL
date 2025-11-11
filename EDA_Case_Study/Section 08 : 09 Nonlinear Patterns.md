# EDA 심화 분석: 비선형 패턴(Non-linear Patterns) 규명

단순 피처 중요도(Feature Importance)를 넘어, 피처가 타겟 변수(`support_needs`)와 **"어떻게(How)"** 관계를 맺는지 심층적으로 분석한다.

이 분석의 목적은 다음과 같다.
1.  관계의 **'모양'**을 시각적으로 이해한다. (1번)
2.  의미 있는 **'경계선(임계값)'**을 찾는다. (2번)
3.  모델이 학습하기 좋은 **'최적의 변환 형태'**를 숫자로 확정한다. (3번)

---

## 1. 다항 회귀 (Polynomial Fit): 관계의 '모양' 시각화

* **목적:** 피처와 타겟의 관계가 단순한 `직선(1차)`인지, `U자형(2차)`인지, `S자형(3차)`인지 '눈'으로 직접 확인한다.
* **방법:** 비선형성이 강할 것으로 의심되는 상위 3개 피처를 뽑아(`top_nonlinear_features`), 1, 2, 3차 곡선을 피팅(fitting)하여 실제 데이터(파란 점)를 가장 잘 따라가는지 본다.

* **코드 예시:**
    ```python
    # (correlation_comparison에서 비선형성 점수 상위 3개 피처 이름 추출)
    top_nonlinear_features = correlation_comparison.nlargest(3, 'nonlinear_strength')['Feature'].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, feature in enumerate(top_nonlinear_features):
        # ... (데이터를 20개 구간(bin)으로 쪼개고 평균을 계산) ...
        bins = pd.qcut(train_df[feature], q=20, duplicates='drop')
        bin_means = train_df.groupby(bins)['support_needs'].mean()
        bin_centers = train_df.groupby(bins)[feature].mean()
        
        axes[idx].scatter(bin_centers, bin_means, s=100, alpha=0.6, label='Actual')
        
        # 1, 2, 3차 다항식 피팅
        for degree in [1, 2, 3]:
            poly_fit = np.poly1d(np.polyfit(bin_centers, bin_means, degree))
            axes[idx].plot(x_range, poly_fit(x_range), label=f'Degree {degree}', alpha=0.8)
    # ... (생략) ...
    ```

* **결과 및 해석:**
    
    * **age:** 파란 점(실제)이 명백한 **U자형**이다. `Degree 2`(주황선)와 `Degree 3`(초록선)가 `Degree 1`(직선)보다 데이터를 훨씬 잘 설명한다.
    * **contract_length:** `Degree 2`(주황선)가 '초반에 높고 후반에 급락'하는 **역U자형** 패턴을 가장 잘 따른다.
    * **frequent:** `Degree 1, 2, 3` 선이 **모두 겹친다.** 이는 굳이 복잡한 곡선이 필요 없는 **선형(직선) 관계**임을 의미한다.

* **💡 1차 인사이트:**
    * `age`와 `contract_length`는 비선형 피처이다. (`age**2`, `contract_length**2` 사용 고려)
    * `frequent`는 선형 피처이다. (그대로 사용)

---

## 2. 결정 트리 임계값 (Decision Tree): '경계선' 찾기

* **목적:** 1번이 '부드러운 곡선'을 찾았다면, 2번은 "Support Level이 급격하게 바뀌는" **'날카로운 경계선(임계값)'**을 찾기 위함이다. (예: "50.5세"를 기준으로 뭔가 바뀐다)
* **방법:** 각 피처마다 단순한 결정 트리(max_depth=3)를 학습시켜, 모델이 '질문'을 던지는 기준값(threshold)을 추출한다.

* **코드 예시:**
    ```python
    threshold_effects = {}
    for idx, feature in enumerate(numeric_features):
        dt = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt.fit(train_df[[feature]], y)
        
        # 트리의 분기점(임계값) 추출
        tree = dt.tree_
        thresholds = []
        def get_thresholds(node=0): # ... (임계값 추출 함수) ...
        
        get_thresholds()
        threshold_effects[feature] = sorted(thresholds)
        
        # 시각화 (실제 분포 + 예측선 + 임계값 점선)
        axes[idx].hist(...) # 실제 분포 (배경)
        ax2 = axes[idx].twinx()
        ax2.plot(feature_range, predictions, 'r-') # 예측 (빨간 계단선)
        for thresh in thresholds:
            axes[idx].axvline(x=thresh, color='black', linestyle='--') # 임계값 (검은 점선)
    # ... (생략) ...
    print("Identified Thresholds:")
    for feature, thresholds in threshold_effects.items():
        print(f"{feature}: {[f'{t:.1f}' for t in thresholds]}")
    ```

* **결과 및 해석:**
    
    * **`age`, `payment_interval`, `contract_length`:** 빨간 예측선이 **'계단'** 모양이다. 이는 이 피처들이 **단독으로도 예측력**이 있으며, 검은 점선(임계값)이 매우 의미 있음을 뜻한다.
    * **`tenure`, `frequent`:** 빨간 예측선이 **'수평'**이다. 이 피처들은 단독으로는 레벨을 구분할 수 없다. (다른 피처와 '조합'되어야 함)
    * `print` 결과: `contract_length: ['60.0', '225.0']`

* **💡 2차 인사이트:**
    * "계약 기간 225일" (`contract_length`)이나 "나이 50.5세" (`age`) 등이 비즈니스적으로 **의미 있는 경계선**이다.
    * 이는 `is_contract_long = (df['contract_length'] > 225)` 같은 **새로운 범주형 파생 변수**를 만드는 데 활용할 수 있다.

---

## 3. 변수 변환 테스트 (Transformation Test): '최적의 형태' 확정

* **목적:** 1, 2번의 시각적 인사이트를 **'숫자(MI 점수)'**로 증명한다. 선형 모델의 성능을 높이기 위해, 피처에 `log`, `sqrt`, `square` 등 어떤 변환을 적용하는 것이 타겟 변수와 **가장 강력한 관계(정보량)**를 갖게 되는지 정량적으로 테스트한다.
* **방법:** 5가지 변환(`original`, `log`, `sqrt`, `square`, `reciprocal`)을 각 피처에 적용한 뒤, 타겟(`y`)과의 **상호 정보량(MI)** 점수를 계산하여 가장 높은 점수를 받은 변환을 찾는다.

* **코드 예시:**
    ```python
    transformations = {
        'original': lambda x: x,
        'log': lambda x: np.log1p(x),
        'sqrt': lambda x: np.sqrt(x),
        'square': lambda x: x**2,
        'reciprocal': lambda x: 1 / (x + 1)
    }
    transformation_mi = pd.DataFrame(...)

    for feature in numeric_features:
        for trans_name, trans_func in transformations.items():
            # ... (변환 후 MI 점수 계산) ...
            mi = mutual_info_classif(transformed.reshape(-1, 1), y, random_state=42)[0]
            transformation_mi.loc[feature, trans_name] = mi

    # 최고 점수를 받은 변환법 자동 추출
    best_transformations = transformation_mi.idxmax(axis=1)
    print("\nBest Transformation for Each Feature:")
    for feature, best_trans in best_transformations.items():
        # ... (원본 대비 향상도 출력) ...
    ```

* **결과 및 해석:**
    ```
    Best Transformation for Each Feature:
    tenure: square (+0.6% MI improvement)
    payment_interval: sqrt (+0.3% MI improvement)
    after_interaction: sqrt (+1.0% MI improvement)
    contract_length: square (+1.3% MI improvement)
    ```
    * `tenure`와 `contract_length`는 `square`(제곱) 변환이, `payment_interval`은 `sqrt`(제곱근) 변환이 MI 점수가 가장 높았다.

* **💡 3차 인사이트 (최종 결론):**
    * 이 결과는 **선형 모델(예: 로지스틱 회귀)의 성능을 극대화**하기 위한 구체적인 피처 엔지니어링 가이드라인이 된다.
    * `age`가 U자형(1번)이었던 것과 `age`의 최적 변환이 `square`(3번, 결과 생략됨)라는 것이 일치한다.
    * 따라서 선형 모델 사용 시, `tenure`는 `tenure**2`로, `payment_interval`은 `np.sqrt(payment_interval)`로 변환해서 넣어야 모델 성능이 향상될 것이다.
