# Section 9. 고객 프로파일링 및 세분화 (Customer Profiling & Segmentation)

이 분석은 4단계로 진행된다:
1.  **정량적 프로파일링:** 타겟별(Level 0, 1, 2) 통계적 '원천 데이터'를 확보한다.
2.  **정성적 페르소나:** 1번의 통계표를 '대표 인물 1명'으로 단순 요약(스케치)한다.
3.  **의미있는 세분화:** '평균'의 함정(2번)을 넘어, 도메인 지식과 EDA 인사이트로 '핵심 그룹'을 직접 정의하고 검증한다.
4.  **고객 여정 분석:** '시간(가입 기간)'에 따라 고객 행동이 어떻게 변하는지 추적한다.

---

## 1. 정량적 프로파일링 (Comprehensive Profiles)

* **목적:** 모든 분석의 '기초(Foundation)'가 되는 객관적인 통계표를 만든다. "지원 레벨(0, 1, 2)별로 고객들은 **통계적으로** 어떻게 다른가?"를 확인한다.
* **방법:** `support_needs` 레벨로 그룹을 나누고, 각 그룹의 모든 피처에 대한 기초 통계치(평균, 중앙값, 분포 등)를 계산한다.

* **코드 예시:**
    ```python
    profiles = {}
    for level in range(3):
        subset = train_df[train_df['support_needs'] == level]
        
        profile = {
            'size': len(subset),
            'percentage': len(subset) / len(train_df) * 100
        }
        
        # Numeric statistics
        for feature in numeric_features:
            profile[f'{feature}_mean'] = subset[feature].mean()
            profile[f'{feature}_median'] = subset[feature].median()
            # ... (std, 25%, 75%...)
        
        # Categorical distributions
        profile['gender_dist'] = subset['gender'].value_counts(normalize=True).to_dict()
        # ... (subscription_dist, contract_dist...)
        
        profiles[f'Level_{level}'] = profile

    profile_df = pd.DataFrame(profiles).T
    print(profile_df[mean_cols].round(1))
    ```

* **결과 및 해석:**
    ```
    Mean Values by Feature:
              age_mean tenure_mean frequent_mean ...
    Level_0  36.960271   31.544520     15.989438 ...
    Level_1  41.452091   31.281909     15.730264 ...
    Level_2  41.666868   30.709463     15.556752 ...
    ```
    * **💡 인사이트:**
        * `age`: Level 0(낮은 지원) 고객은 평균 37세로, Level 1, 2(약 41.5세)보다 **눈에 띄게 어리다.**
        * `payment_interval`, `after_interaction`: 레벨 0 -> 2로 갈수록 **평균값이 꾸준히 증가**한다. (강한 양의 관계)
        * `tenure`, `frequent`: 세 그룹 간 **평균 차이가 거의 없다.** (단독으로는 예측력이 낮음)

---

## 2. 정성적 페르소나 (Typical Personas)

* **목적:** 1번의 복잡한 통계표를 **"각 레벨을 대표하는 가상 고객 1명"** 의 '인물 스케치'로 단순 요약한다. (빠른 요약용)
* **방법:** 각 레벨의 **중앙값(Median, 수치형)** 과 **최빈값(Mode, 범주형)** 을 뽑아 대표 프로필을 만들고, **레이더 차트(Radar Chart)** 로 시각화한다.

* **코드 예시:**
    ```python
    personas = {}
    for level in range(3):
        subset = train_df[train_df['support_needs'] == level]
        
        # 1. 중앙값(median)과 최빈값(mode)으로 대표값 추출
        persona = {
            'typical_age': int(subset['age'].median()),
            'typical_gender': subset['gender'].mode()[0],
            'typical_contract': int(subset['contract_length'].mode()[0]),
            # ... (etc)
        }
        
        # 2. 사람이 이해하는 단어로 변환 (e.g., 42 -> "middle-aged")
        age_group = 'young' if persona['typical_age'] < 35 else ...
        
        # 3. 텍스트 설명 생성
        persona['description'] = f"A {age_group} {persona['typical_gender']} customer..."
        
        personas[f'Level_{level}_Persona'] = persona

    # ... (Radar Chart 시각화 코드) ...
    
    print("=== Customer Personas ===")
    for name, persona in personas.items():
        print(f"\n{name}:\nDescription: {persona['description']}")
    ```

* **결과 및 해석:**
    
    * **레이더 차트:** Level 0, 1, 2의 '프로필 형태'를 비교한다. Level 2(초록색)가 `Payment Interval` 축에서 유독 바깥쪽으로 뻗어있는 것을 한눈에 볼 수 있다.
    * **텍스트 설명:**
        * Level 1: "중년 남성(M), 360일 계약..."
        * Level 2: "중년 여성(F), 30일 계약..."
    * **💡 인사이트:** Level 2(높은 지원) 고객은 Level 1과 나이대는 비슷하지만, **'성별(여성)'** 과 **'계약 기간(30일 단기)'** 에서 결정적인 차이를 보인다.
    * **⚠️ 한계:** 이 방식은 '평균'적인 모습만 보여준다. 만약 분포가 봉우리 2개(Bimodal)라면, **중앙값은 아무도 없는 '가운데'를 대표로 뽑는 오류**를 범할 수 있다. (e.g., 20대/50대가 많은데 35세를 대표로 뽑음)

---

## 3. 의미있는 세분화 (Meaningful Segments)

* **목적:** 2번(중앙값)의 한계를 극복하기 위해, **'도메인 지식'** 과 **'EDA 인사이트(PCA 등)'** 를 기반으로 **"비즈니스적으로 의미 있는" 고객 그룹**을 직접 정의하고, 이 그룹들의 지원 레벨을 역으로 추적한다.
* **방법:** `(age < 30) & (frequent > 20)`처럼 구체적인 '규칙(Rule)'으로 5개의 세그먼트를 정의하고, 각 세그먼트의 `support_needs` 비율을 계산한다.

* **코드 예시:**
    ```python
    segments = {
        'Digital Natives': ( # 예: 도메인 지식 기반
            (train_df['age'] < 30) & (train_df['frequent'] > 20)
        ),
        'High Tenure + Low Frequency': ( # 예: PCA(PC2) 인사이트 기반
            (train_df['tenure'] > 40) & (train_df['frequent'] < 10)
        ),
        # ... (Premium Loyalists, Value Seekers 등)
    }

    segment_analysis = []
    for segment_name, mask in segments.items():
        segment_data = train_df[mask]
        # ... (Level_0_Rate, Level_1_Rate, Level_2_Rate 계산) ...
        segment_analysis.append({ ... })

    segment_df = pd.DataFrame(segment_analysis)
    
    # 누적 막대 그래프로 시각화
    segment_df.set_index('Segment')[support_cols].plot(kind='bar', stacked=True, ...)
    ```

* **💡 인사이트:**
    이 분석은 **"가장 실행 가능한(Actionable)"** 결과를 준다. 마케팅팀이 정의한 'Digital Natives' 그룹이 Level 2 지원을 얼마나 요구하는지(위험도) 정확히 알려줄 수 있다.

---

## 4. 고객 여정 분석 (Customer Journey Analysis)

* **목적:** 고객을 '정적인 스냅샷'이 아닌, **"시간의 흐름(가입 기간)"** 에 따라 행동과 요구가 어떻게 **'변화'** 하는지 동적으로 추적한다. (고객 생애주기 분석)
* **방법:** `tenure`(가입 기간)를 `pd.cut`으로 5개 그룹(0-6m, 6-12m, 1-2y...)으로 나누고 (코호트 분석), 각 단계별로 핵심 지표(지원 요구 비율, 접속 빈도 등)가 어떻게 변하는지 꺾은선 그래프로 그린다.

* **코드 예시:**
    ```python
    # 1. 가입 기간(tenure) 기준으로 5개 그룹(Phase) 생성
    tenure_bins = [0, 6, 12, 24, 36, 60]
    tenure_labels = ['0-6m', '6-12m', '1-2y', '2-3y', '3y+']
    train_df['tenure_phase'] = pd.cut(train_df['tenure'], bins=tenure_bins, labels=tenure_labels)

    # 2. 각 Phase별로 핵심 지표 평균 계산
    for phase in tenure_labels:
        phase_data = train_df[train_df['tenure_phase'] == phase]
        # ... (Avg_Frequent, Level_2_Rate, VIP_Subscription_Rate 등 계산) ...
        journey_metrics.append({ ... })
    
    journey_df = pd.DataFrame(journey_metrics)

    # 3. 꺾은선 그래프로 시각화
    journey_df.plot(x='Phase', y=metric, kind='line', marker='o', ...)
    ```

* **결과 및 해석:**
    
    * **`High Support Need Rate (%)`**가 **W자 형태**를 보인다.
    * **`Avg_Frequent`**, **`Avg_Payment_Interval`** 역시 **1-2년 차에 최고점**을 찍는다.
    * **💡 인사이트 (고객 스토리):**
        1.  **1차 위기 (0-6m):** 신규 고객이 서비스 적응 문제로 지원 요구(29.0%)가 높다.
        2.  **안정기 (6-12m):** 적응을 마치고 지원 요구가 가장 낮아진다(25.8%).
        3.  **2차 위기 (1-2y):** 고객이 **'핵심 유저'** 가 되어 **가장 활발하게 활동(Frequent/Payment 최고)** 하며, 이 과정에서 **지원 요구도 다시 최고치(29.1%)** 로 치솟는다.
    * **🔥 핵심 결론:** 1~2년 차 고객은 '가장 가치 있는 고객'이자 '가장 불만이 많은' 고객이다. 이들을 놓치지 않기 위한 **집중 관리가 필요한 '골든 타임'** 이다.
