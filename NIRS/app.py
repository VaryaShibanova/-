"""
Streamlit web app: Programmer Salary Prediction
Stack Overflow Developer Survey 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Предсказание зарплаты программиста",
    page_icon="💻",
    layout="wide",
)

st.title("💻 Предсказание зарплаты программиста")
st.markdown(
    "**Датасет:** Stack Overflow Developer Survey 2024 (65 437 строк)  \n"
    "Изменяйте гиперпараметры модели и наблюдайте за изменением качества предсказания."
)

# ─── Загрузка и подготовка данных ────────────────────────────────────────────
@st.cache_data
def load_and_prepare():
    train = pd.read_csv("data/datatraining.csv")
    test1 = pd.read_csv("data/datatest.csv")

    feature_cols = [
        "YearsCodePro_n", "YearsCode_n", "WorkExp_n",
        "EdLevel_n", "OrgSize_n", "Remote_n", "Country_n", "DevType_n",
        "lang_count", "lang_Python", "lang_JavaScript", "lang_TypeScript",
        "lang_SQL", "lang_Java", "lang_CSharp", "lang_CPlusPlus",
        "lang_PHP", "lang_Go", "lang_Rust", "IsFullTime",
    ]
    target_col = "ConvertedCompYearly"

    scaler = MinMaxScaler()
    scaler.fit(train[feature_cols])  # fit только на train, как в ноутбуке

    X_train = scaler.transform(train[feature_cols])
    X_test  = scaler.transform(test1[feature_cols])
    y_train = train[target_col].values
    y_test  = test1[target_col].values

    return X_train, X_test, y_train, y_test, scaler, feature_cols

X_train, X_test, y_train, y_test, scaler, feature_cols = load_and_prepare()

# ─── Боковая панель: гиперпараметры ──────────────────────────────────────────
st.sidebar.header("⚙️ Гиперпараметры модели")

model_choice = st.sidebar.selectbox(
    "Алгоритм",
    ["Random Forest", "Gradient Boosting"],
)

n_estimators = st.sidebar.slider(
    "n_estimators (число деревьев)",
    min_value=10, max_value=300, value=100, step=10,
)

max_depth = st.sidebar.select_slider(
    "max_depth (глубина дерева)",
    options=[3, 5, 7, 10, 15, 20, None],
    value=5,
)

if model_choice == "Gradient Boosting":
    learning_rate = st.sidebar.slider(
        "learning_rate",
        min_value=0.01, max_value=0.5, value=0.1, step=0.01,
    )
else:
    learning_rate = None

min_samples_split = st.sidebar.slider(
    "min_samples_split",
    min_value=2, max_value=20, value=2, step=1,
)

train_btn = st.sidebar.button("🚀 Обучить модель", width="stretch")

if "results" not in st.session_state:
    st.session_state.results = []

# ─── Обучение ────────────────────────────────────────────────────────────────
if train_btn:
    with st.spinner("Обучение модели..."):
        if model_choice == "Random Forest":
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=-1,
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth if max_depth else 5,
                learning_rate=learning_rate,
                min_samples_split=min_samples_split,
                random_state=42,
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        st.session_state.model  = model
        st.session_state.y_pred = y_pred
        st.session_state.metrics = dict(MAE=mae, RMSE=rmse, R2=r2, MAPE=mape)
        st.session_state.results.append({
            "Алгоритм": model_choice,
            "n_estimators": n_estimators,
            "max_depth": str(max_depth),
            "MAE": round(mae, 0),
            "RMSE": round(rmse, 0),
            "R²": round(r2, 4),
            "MAPE, %": round(mape, 2),
        })

# ─── Результаты ──────────────────────────────────────────────────────────────
if "metrics" in st.session_state:
    m = st.session_state.metrics

    st.subheader("📊 Метрики качества на тестовой выборке (3 000 записей)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE ($)", f"{m['MAE']:,.0f}")
    col2.metric("RMSE ($)", f"{m['RMSE']:,.0f}")
    col3.metric("R²", f"{m['R2']:.4f}")
    col4.metric("MAPE (%)", f"{m['MAPE']:.2f}")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Предсказание vs Реальность")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, st.session_state.y_pred, alpha=0.3, s=6, color="steelblue")
        mn, mx = y_test.min(), y_test.max()
        ax.plot([mn, mx], [mn, mx], "r--", lw=2, label="Идеальное предсказание")
        ax.set_xlabel("Реальная зарплата (USD)")
        ax.set_ylabel("Предсказанная зарплата (USD)")
        ax.set_title("Предсказанная vs реальная зарплата")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.subheader("Распределение остатков")
        residuals = y_test - st.session_state.y_pred
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.hist(residuals, bins=50, color="coral", edgecolor="white")
        ax2.axvline(0, color="black", linestyle="--", lw=2)
        ax2.set_xlabel("Остаток (USD)")
        ax2.set_ylabel("Частота")
        ax2.set_title("Распределение остатков предсказания")
        st.pyplot(fig2)
        plt.close()

    if hasattr(st.session_state.model, "feature_importances_"):
        st.subheader("🔍 Важность признаков")
        feat_labels = [
            "YearsCodePro", "YearsCode", "WorkExp",
            "EdLevel", "OrgSize", "Remote", "Country", "DevType",
            "lang_count", "Python", "JavaScript", "TypeScript",
            "SQL", "Java", "C#", "C++", "PHP", "Go", "Rust", "IsFullTime",
        ]
        importances = st.session_state.model.feature_importances_
        imp_df = pd.DataFrame({"Признак": feat_labels, "Важность": importances})
        imp_df = imp_df.sort_values("Важность", ascending=True)

        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.barh(imp_df["Признак"], imp_df["Важность"], color="mediumpurple", edgecolor="white")
        ax3.set_xlabel("Важность")
        ax3.set_title("Важность признаков (Feature Importance)")
        st.pyplot(fig3)
        plt.close()

# ─── История запусков ─────────────────────────────────────────────────────────
if st.session_state.results:
    st.divider()
    st.subheader("📋 История экспериментов")
    history_df = pd.DataFrame(st.session_state.results)
    st.dataframe(history_df, width="stretch")

    if len(st.session_state.results) > 1:
        fig4, axes = plt.subplots(1, 2, figsize=(12, 4))
        x = range(len(history_df))
        axes[0].plot(x, history_df["MAE"], marker="o", color="steelblue")
        axes[0].set_title("MAE по экспериментам")
        axes[0].set_xlabel("Эксперимент №")
        axes[0].set_ylabel("MAE (USD)")
        axes[0].set_xticks(x)

        axes[1].plot(x, history_df["R²"], marker="o", color="coral")
        axes[1].set_title("R² по экспериментам")
        axes[1].set_xlabel("Эксперимент №")
        axes[1].set_ylabel("R²")
        axes[1].set_xticks(x)

        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

# ─── Прогноз для конкретного программиста ────────────────────────────────────
if "model" in st.session_state:
    st.divider()
    st.subheader("🔮 Предсказать зарплату для конкретного программиста")

    # Коды соответствуют LabelEncoder(alphabetical):
    # Brazil=0, Canada=1, France=2, Germany=3, India=4, Netherlands=5,
    # Other=6, Spain=7, Ukraine=8, UK=9, USA=10
    COUNTRY_MAP = {
        "Brazil": 0,
        "Canada": 1,
        "France": 2,
        "Germany": 3,
        "India": 4,
        "Netherlands": 5,
        "Other": 6,
        "Spain": 7,
        "Ukraine": 8,
        "United Kingdom": 9,
        "United States of America": 10,
    }
    DEVTYPE_MAP = {
        "Developer, full-stack": 0,
        "Developer, back-end": 1,
        "Developer, front-end": 2,
        "Data scientist / ML specialist": 3,
        "Data engineer": 4,
        "DevOps specialist": 5,
        "Engineering manager": 6,
        "Developer, mobile": 7,
        "Developer, desktop/enterprise": 8,
        "Developer, embedded": 9,
        "Other": 10,
    }
    ED_MAP = {
        "Primary school": 1,
        "Secondary school": 2,
        "Some college (no degree)": 3,
        "Associate degree": 4,
        "Bachelor's degree": 5,
        "Master's degree": 6,
        "Professional degree": 7,
        "Doctoral degree": 8,
    }
    ORG_MAP = {
        "Freelancer / just me": 1,
        "2–9 employees": 2,
        "10–19 employees": 3,
        "20–99 employees": 4,
        "100–499 employees": 5,
        "500–999 employees": 6,
        "1,000–4,999 employees": 7,
        "5,000–9,999 employees": 8,
        "10,000+ employees": 9,
    }

    c1, c2, c3 = st.columns(3)
    with c1:
        inp_country  = st.selectbox("Страна", list(COUNTRY_MAP.keys()))
        inp_devtype  = st.selectbox("Тип разработчика", list(DEVTYPE_MAP.keys()))
        inp_ed       = st.selectbox("Уровень образования", list(ED_MAP.keys()), index=4)
        inp_org      = st.selectbox("Размер компании", list(ORG_MAP.keys()), index=3)
    with c2:
        inp_ycp      = st.slider("Лет профессионального опыта", 0, 40, 5)
        inp_yc       = st.slider("Лет программирования (всего)", 0, 50, 8)
        inp_we       = st.slider("Лет общего рабочего опыта", 0, 40, 6)
        inp_remote   = st.selectbox("Формат работы", ["Remote", "Hybrid", "In-person"])
    with c3:
        inp_fulltime = st.checkbox("Полная занятость", value=True)
        inp_langs    = st.multiselect(
            "Языки программирования (выберите все, которые знаете)",
            ["Python", "JavaScript", "TypeScript", "SQL", "Java", "C#", "C++", "PHP", "Go", "Rust"],
            default=["Python", "SQL"],
        )

    if st.button("💰 Рассчитать зарплату"):
        remote_enc = {"Remote": 0, "Hybrid": 1, "In-person": 2}[inp_remote]
        lang_flags = {
            "Python": 0, "JavaScript": 0, "TypeScript": 0, "SQL": 0, "Java": 0,
            "C#": 0, "C++": 0, "PHP": 0, "Go": 0, "Rust": 0,
        }
        for l in inp_langs:
            lang_flags[l] = 1

        raw = pd.DataFrame([[
            inp_ycp, inp_yc, inp_we,
            ED_MAP[inp_ed], ORG_MAP[inp_org], remote_enc,
            COUNTRY_MAP[inp_country], DEVTYPE_MAP[inp_devtype],
            len(inp_langs),
            lang_flags["Python"], lang_flags["JavaScript"], lang_flags["TypeScript"],
            lang_flags["SQL"], lang_flags["Java"], lang_flags["C#"], lang_flags["C++"],
            lang_flags["PHP"], lang_flags["Go"], lang_flags["Rust"],
            int(inp_fulltime),
        ]], columns=feature_cols)

        scaled = scaler.transform(raw)
        pred_salary = st.session_state.model.predict(scaled)[0]
        st.success(f"**Предсказанная годовая зарплата: ${pred_salary:,.0f} USD**")
