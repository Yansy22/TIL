# 차원 축소: PCA vs t-SNE (개념 및 활용법)

PCA와 t-SNE는 모두 고차원(피처가 많은) 데이터를 2D/3D로 시각화하는 차원 축소 기법입니다. 하지만 그 목적과 해석 방법은 완전히 다릅니다.

---

## 1. PCA (주성분 분석)

PCA는 **데이터의 전체적인 구조(분산)**를 보존하는 데 중점을 둡니다. "데이터가 가장 넓게 퍼져 있는 방향(축)"을 찾아(PC1), 그다음 수직 방향으로 가장 넓게 퍼진 축(PC2)을 찾는 방식입니다.

### ✅ 활용 1: 차원 축소 가능성 확인 (Scree Plot)

* **목적:** 6개의 피처를 몇 개의 '핵심 요약 피처'로 줄일 수 있는지 확인합니다.
* **코드 예시:**
    ```python
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    cumulative_var_ratio = np.cumsum(pca.explained_variance_ratio_)

    plt.plot(range(1, len(cumulative_var_ratio)+1), cumulative_var_ratio, 'ro-')
    plt.axhline(y=0.95, color='k', linestyle='--') # 95% 기준선
    plt.title('PCA Explained Variance')
    plt.show()
    ```
* **결과 해석:**
    * **(만약 차원 축소가 가능하다면):** 빨간 선이 2~3번째 지점에서 95% 기준선을 넘습니다. -> "6개 피처는 2~3개로 요약 가능하다."
    * **(이번 EDA의 경우):**
        
        빨간 선이 5~6번째가 되어서야 95% 기준선을 넘습니다.
    * **💡 인사이트:** 6개 피처는 각자 고유한 정보를 가져서 **차원 축소(요약)가 부적절하다**는 결론을 내립니다.

### ✅ 활용 2: 숨겨진 패턴(테마) 발견 (Component Loadings)

* **목적:** 차원 축소와 별개로, 피처 간의 숨겨진 관계를 '엿보기' 위해 사용합니다. (EDA의 핵심)
* **코드 예시:**
    ```python
    loadings = pd.DataFrame(pca.components_[:3].T, columns=['PC1','PC2','PC3'], index=numeric_features)
    sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0)
    plt.title('PCA Component Loadings')
    plt.show()
    ```
* **결과 해석:**
    
    * **PC1 ("고객 가치 축"):** `age`(0.5), `payment_interval`(0.59)과 `contract_length`(-0.5)가 반대로 묶입니다.
    * **PC2 ("충성도 vs 활동성 축"):** `tenure`(0.71)와 `frequent`(-0.69)가 **강하게 반대로 묶입니다.**
    * **💡 인사이트:** "오래된 고객(`tenure`)은 접속이 뜸하고(`frequent`), 자주 접속하는 고객은 신규 고객이다"라는 **새로운 도메인 지식**을 발견했습니다. 이는 `frequent / tenure` 같은 파생 변수 생성의 강력한 근거가 됩니다.

---

## 2. t-SNE (t-분포 확률적 이웃 임베딩)

t-SNE는 **국지적인 이웃 관계(유사도)**를 보존하는 데 중점을 둡니다. "원본 데이터에서 가까웠던 점들은, 2D에서도 무조건 가깝게" 뭉쳐놓는 것을 목표로 합니다.

### ✅ 활용: 군집(Cluster) 존재 여부 확인

* **목적:** "데이터가 자연스러운 덩어리(군집)로 나뉘는가?"라는 질문에만 답하기 위해 사용합니다.
* **코드 예시:**
    ```python
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_sample)

    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_sample, cmap='viridis', alpha=0.6)
    plt.colorbar(label='Support Level')
    plt.title('t-SNE Visualization')
    plt.show()
    ```
* **결과 해석:**
    
    * **관찰:** `support_needs` 레벨(색상)별로 뚜렷한 '섬'이 나뉘지 않습니다. 보라색(0), 초록색(1), 노란색(2) 점들이 모두 **뒤죽박죽 섞여 있습니다.**
    * **💡 인사이트:**
        1.  이 수치형 피처 6개만으로는 `support_needs` 레벨을 구분하기에 **정보가 부족하다**는 강력한 증거입니다.
        2.  `gender`, `subscription_type` 같은 **범주형 피처**나 **파생 피처**가 모델링에 반드시 필요함을 시사합니다.
* **⚠️ 해석 주의사항:**
    * **축(t-SNE 1, 2)은 아무 의미 없습니다.** (해석 금지)
    * **군집 간의 거리, 군집의 크기/밀도**는 아무 의미 없습니다. (해석 금지)
    * 오직 **"같은 색끼리 뭉쳤는가, 섞였는가?"**만 봅니다.



# 페르소나 기반 가설 검증 (조건부 분석)

PCA/t-SNE가 기계 중심의 '탐색'이었다면, 이 분석은 **도메인 지식**과 **EDA 인사이트**를 결합하여 "사람이 이해할 수 있는" 구체적인 가설을 검증하는 단계입니다.

* **도메인 지식:** "젊고 신규 고객이 지원 요청이 많을 것이다."
* **EDA 인사이트 (from PCA):** "`tenure`와 `frequent`가 반대로 움직인다."

이 둘을 조합하여 **"오래됐지만 접속이 뜸한 고객(`High Tenure + Low Frequency`)"** 같은 구체적인 페르소나(가설)를 만듭니다.

### ✅ 활용: 페르소나별 타겟 비율 비교

* **목적:** 내가 정의한 페르소나 그룹이 **전체 평균(Baseline)** 대비 `support_needs` 비율이 얼마나 다른지 확인합니다.
* **코드 예시:**
    ```python
    # 1. 가설 정의 (도메인 지식 + EDA 인사이트)
    conditions = [
        {'name':'Young + New + Low Payment',
         'filter': (train_df['age']<30) & (train_df['tenure']<12) & (train_df['payment_interval'] < 10)},
        {'name':'High Tenure + Low Frequency', # <--- PCA(PC2) 인사이트 활용
         'filter':(train_df['tenure']>40) & (train_df['frequent']<10)}
    ]
    
    # 2. 전체 평균(Baseline) 준비
    baseline = train_df['support_needs'].value_counts(normalize=True).sort_index()

    # 3. 시각화 (반복문)
    for idx, cond in enumerate(conditions):
        subset = train_df[cond['filter']]
        support_dist = subset['support_needs'].value_counts(normalize=True).sort_index()
        
        support_dist.plot(kind='bar', ax=axes[idx])
        axes[idx].set_title(f"{cond['name']}\n(n={len(subset)})")
        
        # 4. 베이스라인(빨간 점선)과 비교
        axes[idx].axhline(y=baseline[2], color='red', linestyle='--') # Level 2 기준선
    ```
