# 2025-11-04 Study Log

# 📘 Section 6 : Created Features

## 🎯 목표
파생 변수의 종류에 대해 알아보고, 모델 기반 변수 선정(Feature Importance)을 통해  
데이터가 어떤 특성에 의해 영향을 받는지 분석한다.

---

## 🧱 도메인 기반 피처 생성 (`create_domain_features`)

고객 특성(나이, 가입기간, 결제주기 등)을 활용해 **비즈니스적으로 해석 가능한 피처**를 직접 설계했다.

| 피처 그룹 | 예시 | 의미 |
|------------|------|------|
| 고객 세그먼트 | `age_group`, `loyalty_level` | 고객 연령대 및 충성도 구간화 |
| 고객 가치 | `subscription_value`, `customer_lifetime_value` | 상품 가치와 계약 기간을 통한 고객 Lifetime Value |
| 활동성 지표 | `interaction_rate`, `activity_intensity` | 이용 빈도와 상호작용 강도 조합 |
| 결제 관련 | `monthly_payment_frequency`, `tenure_contract_ratio` | 결제 주기와 계약 기간 비율 |
| 복합지표 | `is_high_value`, `churn_risk` | VIP 여부 및 이탈 위험 예측 지표 |

```python
def create_domain_features(df):
  # 1. Customer segment related
  # Age groups
  df['age_group'] = pd.cut(df['age'], bins=[0,25,35,50,100],
                           labels=['young','middle','senior','elder'])
  df['age_group_encoded'] = df['age_group'].map({'young':0, 'middle':1, 'senior':2, 'elder':3})

  # Customer loyalty (based on tenure)
  df['loyalty_level'] = pd.cut(df['tenure'], bins=[0,12,24,36,100],
                               labels=['new','regular','loyal','vip'])
  df['loyalty_level_encoded'] = df['loyalty_level'].map({'new':0, 'regular':1, 'loyal':2, 'vip':3})

  # 2. Customer value score
  # Subscription value = subscription type X contract length
  df['subscription_value'] = df['subscription_encoded'] * df['contract_length']

  # Customer lifetime value indicator
  df['customer_lifetime_value'] = df['tenure'] * df['subscription_encoded'] * df['payment_interval']

  # 3. Activity indicators
  # Interaction ratio
  df['interaction_rate'] = df['after_interaction'] / (df['frequent'] + 1) # +1 to avoid division by zero

  # Activity intensity
  df['activity_intensity'] = df['frequent'] * df['after_interaction']

  # 4. Payment related
  # Monthly average payment cycle
  df['monthly_payment_frequency'] = 30 / (df['payment_interval'] + 1)

  # Loyalty vs contract ratio
  df['tenure_contract_ratio'] = df['tenure'] / df['contract_length']

  # 5. Composite indicators
  # High value customer flag
  df['is_high_value'] = ((df['subscription_type'] == 'vip') &
                         (df['contract_length'] >= 90) &
                         (df['tenure'] > 24)).astype(int)

  # Churn risk flag
  df['churn_risk'] = ((df['after_interaction'] < df['after_interaction'].quantile(0.25)) &
                (df['frequent'] < df['frequent'].quantile(0.25))).astype(int)

  return df

# Apply feature creation
train_df = create_domain_features(train_df)
test_df = create_domain_features(test_df)

print("✓ Domain-based feature creation completed")
print(f"Number of created features: {len([col for col in train_df.columns if col not in train_original.columns])}")
```

> **요약:**  
> 이 단계는 데이터에 의미적 풍부함을 더하기 위한 ‘도메인 감각형 EDA’다.  
> 단순 수치 조합이 아니라, 실제 고객 행동 패턴을 반영하도록 설계되었다.

---

## ⚙️ 상호작용 및 비율 피처 (`create_interaction_features`)

변수 간 관계를 모델이 직접 학습하지 않아도 되게끔,  
**명시적 interaction 및 비율형 피처**를 추가했다.

| 유형 | 예시 | 목적 |
|------|------|------|
| 수치형 간 | `age * tenure`, `payment_interval * frequent` | 장기 고객의 활동성 효과 반영 |
| 범주형-수치형 | `gender_encoded * age`, `subscription_encoded * frequent` | 성별·구독유형에 따른 행동 차이 |
| 비율형 | `age_tenure_ratio`, `payment_tenure_ratio` | 성장 속도 및 소비 주기 비교 |
| 다항항 | `after_interaction_squared` | 비선형 관계(활동성 급증 등) 반영 |

```python
def create_interaction_features(df):
  # 1. Numerical variable interactions
  df['age_tenure_interaction'] = df['age'] * df['tenure']
  df['payment_frequent_interaction'] = df['payment_interval'] * df['frequent']
  df['after_payment_interaction'] = df['after_interaction'] * df['payment_interval']

  # 2. Categorical-numerical interactions
  # Age effect by gender
  df['gender_age_interaction'] = df['gender_encoded'] * df['age']

  # Activity by subscription type
  df['subscription_frequent_interaction'] = df['subscription_encoded'] * df['frequent']

  # 3. Ratio-based interactions
  df['age_tenure_ratio'] = df['age'] / (df['tenure'] + 1)
  df['payment_tenure_ratio'] = df['payment_interval'] / (df['tenure'] + 1)

  # 4. Polynomial features (2nd degree)
  df['after_interaction_squared'] = df['after_interaction'] ** 2
  df['payment_interval_squared'] = df['payment_interval'] ** 2

  return df

# Apply interaction features
train_df = create_interaction_features(train_df)
test_df = create_interaction_features(test_df)

print("✓ Interaction feature creation completed")

```

> **요약:**  
> Interaction 피처는 변수 간 복합 관계를 모델이 쉽게 학습하도록 도와준다.  
> 특히 거리 기반 모델보다는 트리 기반 모델에서 효과적이다.

---

## 📊 통계 기반 피처 (`create_statistical_features`)

`subscription_type` 그룹별 평균/표준편차를 계산하여,  
개별 고객의 값이 그 그룹에서 얼마나 벗어났는지를 `z-score` 형태로 표현했다.

| 계산 항목 | 예시 | 해석 |
|------------|------|------|
| 그룹 통계 | `subscription_age_mean`, `subscription_tenure_std` | 구독유형별 평균/표준편차 |
| 상대적 위치 | `age_zscore`, `tenure_zscore` | 해당 그룹에서의 상대적 위치 |

```python
def create_statistical_features(train,test):
  # Calculate group statistics (based on train data)

  # 1. Average metrics by subscription type
  subscription_stats = train.groupby('subscription_type').agg({
      'age':['mean','std'],
      'tenure':['mean','std'],
      'after_interaction':['mean','std']
  })
  subscription_stats.columns = ['_'.join(col) for col in subscription_stats.columns]

  # Apply to Train
  for col in subscription_stats.columns:
    train = train.merge(
        subscription_stats[col].reset_index(),
        on = 'subscription_type',
        how = 'left'
    )
    train.rename(columns={col: f'subscription_{col}'}, inplace=True)

  # Apply to Test
  for col in subscription_stats.columns:
    test = test.merge(
        subscription_stats[col].reset_index(),
        on = 'subscription_type',
        how = 'left'
    )
    test.rename(columns={col: f'subscription_{col}'}, inplace=True)

  # 2. Individual z-scores (deviation from group)
  for base_col in ['age','tenure','after_interaction']:
    mean_col = f'subscription_{base_col}_mean'
    std_col = f'subscription_{base_col}_std'

    train[f'{base_col}_zscore'] = (train[base_col] - train[mean_col]) / (train[std_col] + 1e-8)
    test[f'{base_col}_zscore'] = (test[base_col] - test[mean_col]) / (test[std_col] + 1e-8)

  return train, test

# Apply statistical features
train_df, test_df = create_statistical_features(train_df, test_df)

print("✓ Statistical feature creation completed")
```

> **요약:**  
> 단순 평균값이 아닌 “**해당 그룹 내에서의 특이성**”을 측정한 고급형 피처 설계이다.  
> 그룹 단위 통계와 개인 단위 데이터를 결합해, ‘평균에서 얼마나 벗어난 고객인가’를 모델이 학습하게 한다.

---

## 🌲 Feature Importance 분석

랜덤포레스트 모델로 모든 피처를 학습시켜 중요도를 계산했다.

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 모델 학습
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

# 중요도 계산
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# 상위 10개 출력
print("=== Top 10 Important Features ===")
print(feature_importance.head(10))
```

## 🌲 파생 변수 종류 간 중요도 계산

파생 변수 종류 별로 중요도를 계산해서 비교했다.

```python
# Categorize features
feature_categories = {
    'Original': original_features,
    'Domain': ['age_group_encoded', 'loyalty_encoded', 'subscription_value',
               'customer_lifetime_value', 'interaction_rate', 'activity_intensity',
               'monthly_payment_frequency', 'tenure_contract_ratio',
               'is_high_value', 'churn_risk'],
    'Interaction': [col for col in feature_cols if 'interaction' in col or 'ratio' in col],
    'Statistical': [col for col in feature_cols if 'subscription_' in col or 'zscore' in col],
    'Polynomial': [col for col in feature_cols if 'squared' in col]
}

# Sum importance by category
category_importance = {}
for category, features in feature_categories.items():
  features_in_model = [f for f in features if f in feature_importance['feature'].values]
  importance_sum = feature_importance[feature_importance['feature'].isin(features_in_model)]['importance'].sum()
  category_importance[category] = importance_sum


# Visualization
plt.figure(figsize=(10,6))
categories = list(category_importance.keys())
importances = list(category_importance.values())

bars = plt.bar(categories, importances, color=plt.cm.Set3(range(len(categories))))
plt.title('Total Importance by Feature Category', fontsize=14)
plt.ylabel('Total Importance')
plt.xticks(rotation=45)

# Display values
for bar, imp in zip(bars, importances):
  plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
           f'{imp:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

