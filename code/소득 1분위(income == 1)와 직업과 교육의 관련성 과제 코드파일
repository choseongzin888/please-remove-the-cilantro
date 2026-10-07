import numpy as np
import pandas as pd

# 1. 결측치 처리 및 변수 생성
# 소득(income) 변수에서 9는 코드 결측이므로 NaN으로 처리합니다.
df['income_clean'] = df['income'].replace(9, np.nan)

# 소득 1분위(노출) vs 소득 2-4분위(비노출) 지시 변수 생성
income_q1 = (df['income_clean'] == 1).to_numpy()
income_q24 = (df['income_clean'].isin([2, 3, 4])).to_numpy()

# 교육수준 초졸 이하 (edu == 1)
# edu 변수에서 결측이나 무응답 코드가 있다면 NaN 처리가 필요할 수 있습니다.
edu_low = (df['edu'] == 1).to_numpy()

# 2. 도메인 정의
DOMS_INC = {
    '전체': np.ones(len(df), bool),
    '소득 1분위 (노출)': income_q1,
    '소득 2-4분위 (비노출)': income_q24
}

def row_cont_inc(label, var):
    """연속형: 가중 평균 (SE)"""
    r = {'특성': label, 'n': int(df[var].notna().sum())}
    for name, dom in DOMS_INC.items():
        th, se = svymean_domain(df[var], w, ST, PS, dom)
        r[name] = f'{th:.1f} ({se:.2f})'
    return r

def row_cat_inc(label, mask):
    """범주형: n (가중 %)"""
    mask = np.asarray(mask, float)
    r = {'특성': label, 'n': int(mask.sum())}
    for name, dom in DOMS_INC.items():
        th, se = svymean_domain(mask, w, ST, PS, dom)
        r[name] = f'{th*100:.1f}'
    return r

# 3. Table 1 조립
t1_income = pd.DataFrame([
    row_cont_inc('연령 (세)', 'age'),
    row_cat_inc('  65세 이상', df.age >= 65),
    row_cat_inc('소득 1분위 (하)', df['income_clean'] == 1),
    row_cat_inc('읍면 거주', df.urban == 2),
    row_cat_inc('교육수준 (초졸 이하)', df.edu == 1),
    row_cont_inc('BMI (kg/m²)', 'bmi'),
    row_cat_inc('  비만 (BMI≥25)', df.bmi >= 25),
    row_cont_inc('수축기혈압 (mmHg)', 'sbp'),
    row_cat_inc('고혈압', df.htn == 1),
    row_cat_inc('현재흡연', df.smk == 1),
    row_cat_inc('규칙적 운동', df.exercise == 2),
])

display(t1_income)


| 변수 | n | 전체 | 소득 1분위 (노출) | 소득 2-4분위 (비노출) |
|---|---|---|---|---|
| **연령** ||||
| - 연령 (세) | 5367 | 49.1 (0.16) | 55.7 (0.42) | 46.4 (0.21) |
| - 65세 이상 | 1607 | 16.8 | 30.2 | 11.2 |
| **소득 및 지역** ||||
| - 소득 1분위 (하) | 1469 | 22.7 | 100.0 | 0.0 |
| - 읍면 거주 | 2156 | 22.9 | 34.2 | 18.8 |
| **교육 수준 및 신체계측** ||||
| - 교육수준 (초졸 이하) | 494 | 5.8 | 12.1 | 3.0 |
| - BMI (kg/m²) | 5367 | 23.8 (0.05) | 24.1 (0.10) | 23.7 (0.07) |
| - 비만 (BMI≥25) | 2035 | 36.5 | 39.1 | 35.5 |
| **임상 및 건강행태** ||||
| - 수축기혈압 (mmHg) | 5367 | 122.8 (0.45) | 127.0 (0.66) | 121.0 (0.48) |
| - 고혈압 | 1652 | 24.4 | 35.0 | 20.2 |
| - 현재흡연 | 1295 | 24.1 | 25.4 | 23.7 |
| - 규칙적 운동 | 2238 | 41.5 | 41.9 | 41.6 |
