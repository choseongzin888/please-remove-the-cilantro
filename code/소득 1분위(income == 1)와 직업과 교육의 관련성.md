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
