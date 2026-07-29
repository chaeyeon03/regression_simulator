import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import io

# -----------------------------------------------------------------------------
# 1. 페이지 설정 및 초기화
# -----------------------------------------------------------------------------
st.set_page_config(page_title="CSV 데이터로 배우는 선형회귀 실험실", layout="wide")

# 세션 상태 초기화 (탭 간 데이터 및 모델 유지)
if 'df' not in st.session_state:
    st.session_state.df = None
if 'simple_model_res' not in st.session_state:
    st.session_state.simple_model_res = None
if 'multi_model_res' not in st.session_state:
    st.session_state.multi_model_res = None

# -----------------------------------------------------------------------------
# 2. 공통 도우미 함수
# -----------------------------------------------------------------------------
def generate_sample_data():
    """예제 데이터(미세먼지 농도)를 생성하는 함수"""
    np.random.seed(42)
    n = 150
    temperature = np.random.uniform(-5, 35, n)
    humidity = np.random.uniform(20, 90, n)
    wind_speed = np.random.uniform(0.5, 8.0, n)
    rainfall = np.random.exponential(2, n) # 강수량은 0에 가깝게
    
    # 미세먼지(pm25) 식: 기온↑ 습도↑ 풍속↓ 강수량↓ 일때 증가하는 경향 + 노이즈
    base_pm25 = 40 + (temperature * 0.3) + (humidity * 0.5) - (wind_speed * 4.5) - (rainfall * 1.5)
    noise = np.random.normal(0, 8, n)
    pm25 = base_pm25 + noise
    pm25 = np.where(pm25 < 0, 0, pm25) # 음수 방지
    
    df = pd.DataFrame({
        'temperature': temperature.round(1),
        'humidity': humidity.round(1),
        'wind_speed': wind_speed.round(1),
        'rainfall': rainfall.round(1),
        'pm25': pm25.round(1)
    })
    return df

def calculate_metrics(y_true, y_pred, n, p):
    """모델 평가 지표를 계산하는 함수"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    # 조정된 R^2 계산 (데이터 수가 변수 개수보다 충분히 많아야 함)
    if n - p - 1 > 0:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        adj_r2 = None
    return mae, mse, rmse, r2, adj_r2

def load_csv(file):
    """인코딩 오류를 방지하며 CSV 파일을 읽어오는 함수"""
    try:
        return pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding='cp949')
        except Exception as e:
            st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
            return None
    except Exception as e:
        st.error(f"알 수 없는 오류가 발생했습니다: {e}")
        return None

# -----------------------------------------------------------------------------
# 3. 사이드바 구성
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🧭 학습 네비게이션")
    st.info("우측의 탭을 클릭하여 학습 단계를 이동하세요. 반드시 데이터를 먼저 업로드해야 다음 단계를 진행할 수 있습니다.")
    st.markdown("---")
    st.markdown("### 📚 핵심 용어 사전")
    st.markdown("""
    * **독립변수(X):** 원인이 되는 변수 (입력값)
    * **종속변수(y):** 결과가 되는 변수 (예측값)
    * **회귀계수:** X가 1 증가할 때 y의 변화량(기울기)
    * **잔차:** 실제값과 예측값의 차이
    * **MAE:** 평균 절대 오차
    * **RMSE:** 평균 제곱근 오차
    * **R² (결정계수):** 모델의 설명력 (1에 가까울수록 좋음)
    """)

# -----------------------------------------------------------------------------
# 4. 메인 화면 및 탭 구성
# -----------------------------------------------------------------------------
st.title("📊 CSV 데이터로 배우는 선형회귀 실험실")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. 학습 안내", 
    "2. 데이터 업로드", 
    "3. 데이터 탐색", 
    "4. 단순선형회귀", 
    "5. 다중선형회귀", 
    "6. 평가 및 비교"
])

# ==========================================
# 탭 1: 학습 안내
# ==========================================
with tab1:
    st.header("1. 선형회귀(Linear Regression)란?")
    
    st.markdown("""
    **회귀(Regression)**란 주어진 데이터들을 가장 잘 설명하는 선(수학적 모델)을 찾아, 새로운 데이터가 주어졌을 때 결과를 예측하는 분석 방법입니다.
    이 중에서도 **선형회귀**는 변수들 사이의 관계를 '직선(선형)'으로 가정하고 식을 만듭니다.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**단순선형회귀 (Simple Linear Regression)**\n\n하나의 원인(독립변수 X)으로 결과(종속변수 y)를 예측합니다.")
        st.latex(r"\hat{y} = \beta_0 + \beta_1 x")
        st.markdown("* $\hat{y}$: 예측값 (종속변수)\n* $x$: 입력값 (독립변수)\n* $\\beta_1$: 기울기 (회귀계수)\n* $\\beta_0$: 절편 (x가 0일 때의 값)")
    with col2:
        st.success("**다중선형회귀 (Multiple Linear Regression)**\n\n여러 개의 원인(독립변수 X1, X2...)을 종합하여 결과를 예측합니다.")
        st.latex(r"\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n")
        st.markdown("* 여러 변수가 예측에 어떻게 기여하는지 동시에 파악할 수 있습니다.")

    st.warning("""
    **🚨 주의: 상관관계 ≠ 인과관계**  
    선형회귀 모델이 두 변수 사이에 강한 관계(높은 기울기나 상관계수)를 찾아냈다고 해서, 그것이 반드시 'A가 B의 원인이다'를 의미하지는 않습니다. 데이터는 경향성을 보여줄 뿐, 진짜 원인은 도메인 지식으로 판단해야 합니다.
    """)
    
    with st.expander("🤔 학생용 탐구 질문 (클릭하여 열기)"):
        st.markdown("""
        1. 단순선형회귀 식에서 기울기($\\beta_1$)의 부호가 양수(+)라는 것은 두 변수가 어떤 관계라는 뜻일까요?
        2. 실제값과 예측값의 차이를 '잔차'라고 합니다. 이 잔차가 모두 0이 되는 선형회귀 선을 그릴 수 있을까요?
        """)

# ==========================================
# 탭 2: 데이터 업로드
# ==========================================
with tab2:
    st.header("2. CSV 데이터 업로드")
    st.write("분석할 데이터를 업로드하세요. 데이터가 없다면 예제 데이터를 사용할 수 있습니다.")
    
    # 예제 데이터 다운로드 버튼
    sample_df = generate_sample_data()
    csv = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 미세먼지 예제 데이터 다운로드 (CSV)",
        data=csv,
        file_name='pm25_sample_data.csv',
        mime='text/csv',
    )
    st.markdown("---")
    
    uploaded_file = st.file_uploader("CSV 파일을 올려주세요", type=['csv'])
    
    if uploaded_file is not None:
        df = load_csv(uploaded_file)
        if df is not None:
            st.session_state.df = df
            
    # 데이터가 로드된 경우 정보 표시
    if st.session_state.df is not None:
        df = st.session_state.df
        st.success("데이터가 성공적으로 로드되었습니다!")
        
        st.subheader("데이터 미리보기")
        st.dataframe(df.head())
        
        col1, col2, col3 = st.columns(3)
        col1.metric("전체 행(데이터 개수)", f"{df.shape[0]} 개")
        col2.metric("전체 열(변수 개수)", f"{df.shape[1]} 개")
        
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        
        col3.metric("숫자형 열 개수", f"{len(num_cols)} 개")
        
        if len(num_cols) < 2:
            st.error("⚠️ 숫자형 데이터 열이 2개 미만입니다. 선형회귀를 실행하려면 최소 2개의 숫자형 열이 필요합니다.")
        else:
            st.write(f"**숫자형 변수:** {', '.join(num_cols)}")
            if cat_cols:
                st.write(f"**문자/기타 변수:** {', '.join(cat_cols)} (회귀분석에서 제외됩니다)")
            
            st.subheader("결측값(비어있는 데이터) 확인")
            missing = df.isna().sum()
            if missing.sum() == 0:
                st.info("결측값이 없습니다. 깨끗한 데이터입니다!")
            else:
                st.warning("일부 변수에 결측값이 있습니다. 모델 학습 시 해당 행은 자동으로 제외됩니다.")
                st.dataframe(missing[missing > 0].to_frame(name="결측값 개수"))
                
    else:
        st.info("👆 위에서 CSV 파일을 업로드하거나 예제 데이터를 다운로드 후 업로드해주세요.")
        
    with st.expander("🤔 학생용 탐구 질문"):
        st.markdown("""
        1. 왜 문자형 데이터(예: 날짜, 도시 이름)는 그대로 선형회귀에 사용할 수 없을까요?
        2. 결측값을 삭제하는 대신 다른 값으로 채운다면 어떤 값으로 채우는 것이 좋을까요?
        """)

# ==========================================
# 탭 3: 데이터 탐색 (EDA)
# ==========================================
with tab3:
    st.header("3. 데이터 탐색 (EDA)")
    
    if st.session_state.df is None:
        st.warning("탭 2에서 데이터를 먼저 업로드해주세요.")
    else:
        df = st.session_state.df
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if len(num_cols) < 2:
            st.error("숫자형 열이 부족하여 탐색할 수 없습니다.")
        else:
            st.subheader("1) 변수별 기술통계량")
            st.dataframe(df[num_cols].describe())
            
            st.markdown("---")
            st.subheader("2) 변수 분포 확인 (히스토그램)")
            hist_col = st.selectbox("분포를 확인할 변수를 선택하세요:", num_cols)
            fig_hist = px.histogram(df, x=hist_col, nbins=30, title=f"{hist_col}의 분포")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            st.markdown("---")
            st.subheader("3) 두 변수의 관계 확인 (산점도)")
            sc_col1, sc_col2 = st.columns(2)
            with sc_col1:
                x_col = st.selectbox("X축 변수 선택:", num_cols, index=0)
            with sc_col2:
                # y축 기본값은 x축과 다르게 설정
                y_idx = 1 if len(num_cols) > 1 else 0
                y_col = st.selectbox("Y축 변수 선택:", num_cols, index=y_idx)
                
            if x_col == y_col:
                st.warning("X축과 Y축에 서로 다른 변수를 선택해주세요.")
            else:
                fig_scatter = px.scatter(df, x=x_col, y=y_col, opacity=0.7, 
                                         title=f"{x_col}와(과) {y_col}의 산점도")
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # 산점도 관찰용 안내문
                st.info(f"""
                **그래프 관찰 포인트:**
                * 점들이 왼쪽 아래에서 오른쪽 위로 향하나요? (양의 관계)
                * 점들이 왼쪽 위에서 오른쪽 아래로 향하나요? (음의 관계)
                * 특정 패턴 없이 둥글게 퍼져 있나요? (관계 없음)
                * 무리에서 혼자 동떨어진 점(이상치)이 보이나요?
                """)
            
            st.markdown("---")
            st.subheader("4) 전체 변수 간의 상관계수 (히트맵)")
            st.write("""
            **상관계수(Correlation Coefficient)**는 -1에서 1 사이의 값을 가집니다.
            * **1에 가까울수록**: 강한 양(+)의 관계
            * **-1에 가까울수록**: 강한 음(-)의 관계
            * **0에 가까울수록**: 선형 관계가 없음
            """)
            corr = df[num_cols].corr()
            fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", 
                                 color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            st.plotly_chart(fig_corr, use_container_width=True)
            
    with st.expander("🤔 학생용 탐구 질문"):
        st.markdown("""
        1. 산점도를 보았을 때, 선택한 두 변수는 어떤 관계(양/음/없음)가 있다고 생각하나요?
        2. 상관계수가 0.9인 두 변수를 발견했습니다. 이것을 "A가 변해서 B가 변했다"는 인과관계의 증거로 쓸 수 있을까요?
        """)

# ==========================================
# 탭 4: 단순선형회귀
# ==========================================
with tab4:
    st.header("4. 단순선형회귀 실험")
    
    if st.session_state.df is None:
        st.warning("데이터를 먼저 업로드해주세요.")
    else:
        df = st.session_state.df
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if len(num_cols) < 2:
            st.error("선형회귀를 실행하려면 숫자형 변수가 최소 2개 이상이어야 합니다.")
        else:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                x_var = st.selectbox("독립변수 (X, 원인) 선택:", num_cols, index=0, key='sim_x')
            with col2:
                y_idx = 1 if len(num_cols) > 1 else 0
                y_var = st.selectbox("종속변수 (y, 예측할 결과) 선택:", num_cols, index=y_idx, key='sim_y')
            with col3:
                test_size = st.slider("테스트 데이터 비율 (%)", 10, 40, 20, step=5) / 100.0
                
            if x_var == y_var:
                st.error("X와 y는 서로 다른 변수여야 합니다.")
            else:
                # 1. 결측치 처리 및 데이터 준비
                valid_df = df[[x_var, y_var]].dropna()
                if len(valid_df) < 10:
                    st.error("유효한 데이터가 너무 적어 모델을 학습할 수 없습니다 (최소 10행 이상 필요).")
                else:
                    if len(valid_df) < 30:
                        st.warning("데이터가 30행 미만입니다. 결과 해석에 주의하세요.")
                        
                    X = valid_df[[x_var]]
                    y = valid_df[y_var]
                    
                    # 2. 데이터 분할
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    
                    # 3. 모델 학습
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    # 4. 예측 및 평가
                    y_pred = model.predict(X_test)
                    mae, mse, rmse, r2, adj_r2 = calculate_metrics(y_test, y_pred, len(X_test), 1)
                    
                    coef = model.coef_[0]
                    intercept = model.intercept_
                    
                    # 세션에 저장
                    st.session_state.simple_model_res = {
                        'x_var': [x_var], 'y_var': y_var,
                        'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2, 'adj_r2': adj_r2
                    }
                    
                    st.markdown("---")
                    st.subheader("💡 학습 결과 및 회귀식")
                    
                    st.write(f"학습에 사용된 데이터: **{len(X_train)}개**, 테스트 데이터: **{len(X_test)}개**")
                    
                    # 회귀식 표시
                    st.latex(f"예측\\ {y_var} = ({coef:.4f}) \\times {x_var} + ({intercept:.4f})")
                    
                    # 계수 해석
                    direction = "증가" if coef > 0 else "감소"
                    st.info(f"**기울기 해석:** {x_var} 값이 1만큼 증가할 때, {y_var} 예측값은 평균적으로 약 **{abs(coef):.2f}만큼 {direction}**하는 경향이 있습니다. (단, 인과관계를 의미하지는 않습니다)")
                    
                    # 5. 시각화 (산점도 + 회귀선 + 잔차)
                    st.subheader("📈 회귀선 및 잔차(오차) 확인")
                    
                    # 예측선 생성을 위한 X범위
                    x_range = np.linspace(X.min()[0], X.max()[0], 100).reshape(-1, 1)
                    y_range_pred = model.predict(pd.DataFrame(x_range, columns=[x_var]))
                    
                    fig = go.Figure()
                    
                    # 테스트 데이터 포인트
                    fig.add_trace(go.Scatter(x=X_test[x_var], y=y_test, mode='markers', name='실제값(테스트)', marker=dict(color='blue')))
                    
                    # 회귀선
                    fig.add_trace(go.Scatter(x=x_range.flatten(), y=y_range_pred, mode='lines', name='학습된 회귀선', line=dict(color='red', width=3)))
                    
                    # 잔차 선 (데이터 포인트에서 회귀선까지의 수직선, 20개까지만 표시하여 복잡함 방지)
                    subset_n = min(20, len(X_test))
                    for i in range(subset_n):
                        x_val = X_test.iloc[i, 0]
                        y_true_val = y_test.iloc[i]
                        y_pred_val = y_pred[i]
                        fig.add_trace(go.Scatter(
                            x=[x_val, x_val], 
                            y=[y_true_val, y_pred_val], 
                            mode='lines', 
                            line=dict(color='gray', dash='dot'),
                            showlegend=False
                        ))
                        
                    fig.update_layout(title="실제 데이터와 모델의 회귀선 (점선은 '잔차'를 의미함)", xaxis_title=x_var, yaxis_title=y_var)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 6. 인터랙티브 예측
                    st.markdown("---")
                    st.subheader("🔮 내가 만든 모델로 예측해보기")
                    user_x = st.number_input(f"새로운 {x_var} 값을 입력하세요:", value=float(X[x_var].mean()), step=1.0)
                    user_pred = model.predict(pd.DataFrame([[user_x]], columns=[x_var]))[0]
                    
                    if user_pred < 0:
                         st.warning(f"예측 결과: **{user_pred:.2f}**\n\n(참고: 선형회귀는 직선의 방정식을 따르므로 논리적으로 불가능한 음수 예측값이 나올 수 있습니다. 이는 선형회귀 모델의 한계입니다.)")
                    else:
                        st.success(f"예측 결과: **{user_pred:.2f}**")
                    st.caption("※ 이 값은 데이터에서 학습한 선형적인 경향을 이용한 예측값이며 실제값과 다를 수 있습니다.")

    with st.expander("🤔 학생용 탐구 질문"):
        st.markdown("""
        1. 회귀선은 모든 데이터 점(파란색 점)을 지나가나요? 지나가지 않는다면 그 이유는 무엇일까요?
        2. 잔차(회색 점선)가 양수인 경우, 실제값과 예측값 중 어느 것이 더 큰가요?
        3. 이상치(뚝 떨어져 있는 데이터)를 제거하면 빨간색 회귀선은 어떻게 변할까요?
        """)

# ==========================================
# 탭 5: 다중선형회귀
# ==========================================
with tab5:
    st.header("5. 다중선형회귀 실험")
    st.write("단순선형회귀와 달리, 여러 개의 변수를 동시에 사용하여 예측 성능을 높여봅니다.")
    
    if st.session_state.df is None:
        st.warning("데이터를 먼저 업로드해주세요.")
    else:
        df = st.session_state.df
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        y_var = st.selectbox("종속변수 (y, 예측할 결과) 선택:", num_cols, index=len(num_cols)-1, key='mul_y')
        
        # 종속변수를 제외한 나머지 변수들을 독립변수 후보로 설정
        x_candidates = [col for col in num_cols if col != y_var]
        x_vars = st.multiselect("독립변수 (X, 원인) 다중 선택 (최소 2개 이상):", x_candidates, default=x_candidates[:2])
        
        if len(x_vars) < 2:
            st.warning("다중선형회귀를 실행하려면 최소 2개 이상의 독립변수를 선택하세요.")
        else:
            use_scaler = st.checkbox("데이터 표준화(Standard Scaling) 적용하기", value=True, 
                                     help="단위가 서로 다른 변수들의 크기를 맞추어, 회귀계수로 변수의 영향력을 공정하게 비교할 수 있게 합니다.")
            
            # 1. 결측치 처리 및 데이터 분할
            valid_cols = x_vars + [y_var]
            valid_df = df[valid_cols].dropna()
            
            if len(valid_df) < 10:
                st.error("유효한 데이터가 너무 적습니다.")
            else:
                X = valid_df[x_vars]
                y = valid_df[y_var]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # 2. 모델 학습 (파이프라인 활용)
                if use_scaler:
                    model = Pipeline([
                        ('scaler', StandardScaler()),
                        ('regressor', LinearRegression())
                    ])
                    st.info("데이터 표준화를 적용했습니다. 이제 회귀계수의 절댓값 크기로 변수의 상대적 중요도를 가늠해볼 수 있습니다.")
                else:
                    model = LinearRegression()
                    st.warning("표준화를 적용하지 않았습니다. 변수마다 측정 단위가 다르기 때문에 회귀계수의 크기만으로 중요도를 직접 비교하면 안 됩니다.")
                
                model.fit(X_train, y_train)
                
                # 3. 모델 평가
                y_pred = model.predict(X_test)
                mae, mse, rmse, r2, adj_r2 = calculate_metrics(y_test, y_pred, len(X_test), len(x_vars))
                
                # 세션에 저장
                st.session_state.multi_model_res = {
                    'x_var': x_vars, 'y_var': y_var,
                    'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2, 'adj_r2': adj_r2
                }
                
                # 계수 추출
                if use_scaler:
                    coefs = model.named_steps['regressor'].coef_
                    intercept = model.named_steps['regressor'].intercept_
                else:
                    coefs = model.coef_
                    intercept = model.intercept_
                
                st.markdown("---")
                st.subheader("💡 학습 결과 및 회귀계수")
                
                st.write(f"절편 (Intercept): {intercept:.4f}")
                coef_df = pd.DataFrame({'독립변수': x_vars, '회귀계수': coefs})
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.dataframe(coef_df)
                with col2:
                    fig_bar = px.bar(coef_df, x='회귀계수', y='독립변수', orientation='h', 
                                     title="변수별 회귀계수 시각화", color='회귀계수',
                                     color_continuous_scale=px.colors.diverging.RdBu)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                st.warning("⚠️ **다중선형회귀의 회귀계수 해석:** 다른 모든 입력 변수들이 일정하다고 가정했을 때, 해당 변수가 1단위(표준화 시 1표준편차) 변할 때 종속변수 예측값의 변화량을 의미합니다.")
                
                st.markdown("---")
                st.subheader("🔮 다중 변수 예측 테스트")
                st.write("각 변수의 값을 입력하여 새로운 결과를 예측해 보세요.")
                
                user_inputs = {}
                cols = st.columns(len(x_vars))
                for idx, col_name in enumerate(x_vars):
                    with cols[idx]:
                        user_inputs[col_name] = st.number_input(f"{col_name}", value=float(X[col_name].mean()))
                
                input_df = pd.DataFrame([user_inputs])
                m_pred = model.predict(input_df)[0]
                
                st.success(f"종합 예측 결과: **{m_pred:.2f}**")

    with st.expander("🤔 학생용 탐구 질문"):
        st.markdown("""
        1. 독립변수(X)를 추가했을 때, 예측의 정확도는 무조건 좋아질까요?
        2. 회귀계수가 음수(-)로 나온 변수는 종속변수에 어떤 영향을 미치고 있나요?
        3. 이 모델의 예측 결과를 실제 일기예보나 미세먼지 예보로 사용해도 될까요? 부족한 점은 무엇일까요?
        """)

# ==========================================
# 탭 6: 평가 및 비교
# ==========================================
with tab6:
    st.header("6. 모델 평가 및 비교")
    
    st.markdown("""
    **평가지표(Metric) 설명**
    * **MAE (평균 절대 오차):** 실제값과 예측값 차이의 절댓값 평균. 직관적으로 오차 크기를 알 수 있습니다. (작을수록 좋음)
    * **MSE (평균 제곱 오차):** 오차를 제곱하여 평균 낸 값. 큰 오차에 더 큰 벌점을 줍니다. (작을수록 좋음)
    * **RMSE (평균 제곱근 오차):** MSE에 루트를 씌워 원래 y와 같은 단위로 되돌린 값입니다. (작을수록 좋음)
    * **R² (결정계수):** 모델이 데이터의 변화를 얼마나 잘 설명하는지 나타냅니다. 최대 1이며 높을수록 좋습니다.
    * **조정된 R²:** 무의미한 독립변수를 무조건 많이 추가하여 R²가 가짜로 높아지는 것을 방지(보정)한 값입니다.
    """)
    st.markdown("---")
    
    sim_res = st.session_state.simple_model_res
    mul_res = st.session_state.multi_model_res
    
    if sim_res is None or mul_res is None:
        st.warning("탭 4(단순선형회귀)와 탭 5(다중선형회귀)에서 모델을 먼저 학습시킨 후 이 탭을 확인하세요.")
    else:
        st.subheader("📊 지표 비교표")
        
        comp_df = pd.DataFrame({
            '모델': ['단순선형회귀', '다중선형회귀'],
            '사용된 독립변수': [", ".join(sim_res['x_var']), ", ".join(mul_res['x_var'])],
            'R²': [sim_res['r2'], mul_res['r2']],
            '조정된 R²': [sim_res['adj_r2'], mul_res['adj_r2']],
            'MAE': [sim_res['mae'], mul_res['mae']],
            'MSE': [sim_res['mse'], mul_res['mse']],
            'RMSE': [sim_res['rmse'], mul_res['rmse']]
        })
        
        st.dataframe(comp_df.style.format({
            'R²': '{:.4f}', '조정된 R²': '{:.4f}', 'MAE': '{:.4f}', 'MSE': '{:.4f}', 'RMSE': '{:.4f}'
        }))
        
        st.info("""
        **해석 가이드:**
        * 다중선형회귀의 R²가 더 높더라도, MAE와 RMSE가 함께 줄어들었는지(오차가 작아졌는지) 확인해야 합니다.
        * 변수를 많이 넣었는데 '조정된 R²'가 오히려 떨어졌다면, 불필요한 변수가 포함되었을 수 있습니다.
        * 성능 차이가 미미하다면, 설명하기 쉽고 계산이 가벼운 단순한 모델(단순선형회귀)이 더 좋은 선택일 수 있습니다.
        """)
        
        # 다중선형회귀 모델의 잔차 분석을 위한 재계산 (코드 간소화를 위해 세션의 설정을 다시 불러와 계산)
        df = st.session_state.df
        x_vars = mul_res['x_var']
        y_var = mul_res['y_var']
        valid_df = df[x_vars + [y_var]].dropna()
        X = valid_df[x_vars]
        y = valid_df[y_var]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        st.markdown("---")
        st.subheader("📉 다중선형회귀 모델 잔차 분석 (Residual Analysis)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_ap = px.scatter(x=y_test, y=y_pred, labels={'x': '실제값', 'y': '예측값'},
                                title="1. 실제값 vs 예측값 산점도")
            # 기준선 추가 (y=x)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig_ap.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', 
                                        name='완벽한 예측 기준선', line=dict(color='red', dash='dash')))
            st.plotly_chart(fig_ap, use_container_width=True)
            st.caption("점들이 빨간 점선(기준선) 가까이에 모여있을수록 예측이 실제값과 비슷하다는 의미입니다.")
            
        with col2:
            fig_res = px.scatter(x=y_pred, y=residuals, labels={'x': '예측값', 'y': '잔차(오차)'},
                                 title="2. 예측값 vs 잔차 산점도")
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_res, use_container_width=True)
            st.caption("잔차가 0(빨간 점선)을 중심으로 특정한 패턴(U자형 등) 없이 무작위로 흩어져 있어야 좋은 선형 모델입니다.")
            
        fig_hist = px.histogram(x=residuals, nbins=20, title="3. 잔차 분포 히스토그램",
                                labels={'x': '잔차', 'count': '빈도'})
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption("오차가 0 근처에 가장 많이 몰려있고(종 모양), 좌우 대칭을 이루면 가장 이상적입니다.")

    with st.expander("🤔 학생용 탐구 질문"):
        st.markdown("""
        1. 단순선형회귀와 다중선형회귀 중 어떤 모델이 오차(RMSE)가 더 작게 나왔나요? 그 이유는 무엇일까요?
        2. R²가 높아졌는데 RMSE도 같이 커지는 경우가 발생할 수 있을까요?
        3. 독립변수를 무작정 많이 100개씩 넣으면 무조건 최고의 인공지능 모델이 될까요?
        """)
