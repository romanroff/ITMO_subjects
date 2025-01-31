import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from scipy.stats import pearsonr

# Функция для фильтрации значимых факторов
def filter_significant_features(X, y, significance_level):
    results = []
    for column in X.columns:
        _, p_value = pearsonr(X[column], y)
        correlation_with_y = X[column].corr(y)
        results.append({
            'Feature': column,
            'p-value': p_value,
            'Correlation with Y': correlation_with_y
        })
    results_df = pd.DataFrame(results)
    results_df['Significant'] = results_df['p-value'] < significance_level
    return results_df

# Функция для создания лагов
def create_lags(data, features, max_lag):
    lagged_data = data.copy()
    for feature in features:
        for lag in range(1, max_lag + 1):
            lagged_data[f"{feature}_lag{lag}"] = lagged_data[feature].shift(lag)
    return lagged_data.dropna()

# Функция для тюнинга по лагам
def tune_lags(data, target_variable, features, max_lag):
    best_rmse = np.inf
    best_lags = {}

    for feature in features:
        for lag in range(0, max_lag + 1):
            # Создаем лаги для текущего фактора
            lagged_data = data.copy()
            if lag > 0:
                lagged_data[f"{feature}_lag{lag}"] = lagged_data[feature].shift(lag)
            lagged_data = lagged_data.dropna()

            # Разделяем данные на обучающую и тестовую выборки
            split_index = int(len(lagged_data) * 0.8)
            train_data = lagged_data.iloc[:split_index]
            test_data = lagged_data.iloc[split_index:]

            # Выбираем фичи для текущего лага
            current_features = [f for f in features if f != feature] + ([f"{feature}_lag{lag}"] if lag > 0 else [feature])
            X_train = train_data[current_features]
            y_train = train_data[target_variable]
            X_test = test_data[current_features]
            y_test = test_data[target_variable]

            # Добавляем константу и строим модель
            X_train_const = sm.add_constant(X_train)
            X_test_const = sm.add_constant(X_test)
            model_sm = sm.OLS(y_train, X_train_const).fit()
            y_pred = model_sm.predict(X_test_const)

            # Вычисляем RMSE
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Обновляем лучшие параметры
            if rmse < best_rmse:
                best_rmse = rmse
                best_lags = {feature: lag}

    return best_lags, best_rmse

# Интерфейс Streamlit
st.title("Алгоритм ЛМФМ")

# Загрузка данных
uploaded_file = st.file_uploader("Загрузите EXCEL файл с данными", type=["xlsx"])
if uploaded_file:
    data = pd.read_excel(uploaded_file, index_col=0)
    st.write("Данные:")
    st.dataframe(data)

    # Выбор целевой переменной и факторов
    st.header("Выбор переменных")
    target_variable = st.selectbox("Выберите целевую переменную:", data.columns)
    feature_variables = st.multiselect(
        "Выберите факторы:", [col for col in data.columns if col != target_variable]
    )

    if target_variable and feature_variables:
        # Настройка разбиения данных
        st.header("Настройки разбиения данных")
        test_size = st.slider("Процент тестовых данных (%):", 10, 50, 20) / 100
        split_index = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_index]
        test_data = data.iloc[split_index:]

        # Отбор значимых факторов
        st.header("Отбор значимых факторов")
        significance_level = st.slider("Уровень значимости (p-value):", 0.01, 0.10, 0.05)
        correlation_threshold = st.slider("Порог корреляции:", 0.0, 1.0, 0.5)

        significant_features_df = filter_significant_features(
            train_data[feature_variables], train_data[target_variable], significance_level
        )

        st.write("Таблица значимых факторов (p-value и корреляция с Y):")
        st.dataframe(significant_features_df)

        significant_features = significant_features_df[significant_features_df['Significant']]['Feature'].tolist()
        st.write("Значимые факторы:", significant_features)

        # Вычисление корреляций
        feature_correlations = train_data.corr()

        high_corr_pairs = [
            {"Feature 1": i, "Feature 2": j, "Correlation": feature_correlations.loc[i, j]}
            for i in significant_features for j in significant_features
            if i != j and abs(feature_correlations.loc[i, j]) > correlation_threshold
        ]

        high_corr_pairs_df = pd.DataFrame(high_corr_pairs)
        st.write("Факторы с высокой корреляцией между собой:")
        st.dataframe(high_corr_pairs_df)

        # Исключение факторов
        excluded_features = st.multiselect(
            "Исключите факторы с высокой корреляцией или незначимые:",
            significant_features
        )
        final_features = [f for f in significant_features if f not in excluded_features]
        st.write("Оставшиеся факторы:", final_features)

        # Тюнинг по лагам
        st.header("Тюнинг по лагам")
        max_lag = st.slider("Максимальное количество лагов (0-6):", 0, 6, 3)

        if max_lag > 0:
            best_lags, best_rmse = tune_lags(data, target_variable, final_features, max_lag)
            st.write(f"Лучшие лаги: {best_lags}")
            st.write(f"Минимальный RMSE: {best_rmse:.3f}")

            # Применение лучших лагов
            lagged_data = data.copy()
            for feature, lag in best_lags.items():
                if lag > 0:
                    lagged_data[f"{feature}_lag{lag}"] = lagged_data[feature].shift(lag)
            lagged_data = lagged_data.dropna()

            train_data = lagged_data.iloc[:split_index]
            test_data = lagged_data.iloc[split_index:]

            # Построение модели
            st.header("Построение модели")
            lagged_features = [f"{feature}_lag{lag}" if lag > 0 else feature for feature, lag in best_lags.items()]
            X_train = train_data[lagged_features]
            y_train = train_data[target_variable]
            X_test = test_data[lagged_features]
            y_test = test_data[target_variable]


        # Добавление константы
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)

        # Модель OLS
        model_sm = sm.OLS(y_train, X_train_const).fit()
        y_pred = model_sm.predict(X_test_const)

        # Оценка модели
        st.header("Оценка модели")
        r2 = model_sm.rsquared
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        e_metric = np.mean(np.abs((y_test - y_pred) / y_test))
        f_stat = model_sm.fvalue
        f_p_value = model_sm.f_pvalue

        st.write(f"Коэффициент детерминации (R²): {r2:.3f}")
        st.write(f"Среднеквадратичная ошибка (RMSE): {rmse:.3f}")
        st.write(f"Ошибка E: {e_metric:.3f}")
        st.write(f"F-статистика: {f_stat:.3f}")
        st.write(f"P-значение: {f_p_value:.3f}")
        st.write("Модель адекватна" if f_p_value < 0.05 else "Модель неадекватна")

        st.header("Итоговая модель")
        st.code(model_sm.summary())


        # Визуализация
        st.header("Визуализация результатов")
        results = pd.DataFrame({"Истинные значения": y_test, "Прогноз": y_pred})
        st.write(results)
        st.line_chart(results)


        # Предсказания на новых данных
        st.header("Предсказания на новых данных")
        new_uploaded_file = st.file_uploader("Загрузите новый EXCEL файл для предсказаний", type=["xlsx"])
        if new_uploaded_file:
            new_data = pd.read_excel(new_uploaded_file, index_col=0)
            st.write("Новый набор данных:")
            st.dataframe(new_data)

            # Фильтрация только значимых колонок
            new_data_filtered = new_data[final_features]

            # Создание лаговых переменных на новом наборе данных
            for feature in final_features:
                for lag in range(1, max_lag + 1):
                    new_data_filtered[f"{feature}_lag{lag}"] = new_data[feature].shift(lag)

            new_data_filtered = new_data_filtered[list(X_train.columns)].dropna()  # Удаляем строки с пропущенными значениями

            # Прогнозирование
            X_new = sm.add_constant(new_data_filtered)
            y_new_pred = model_sm.predict(X_new)
            y_test_new = new_data[target_variable]

            # Вывод предсказаний
            results_new = pd.DataFrame({"Истинные значения": y_test_new, "Прогноз": y_new_pred})
            st.write(results_new)
            st.line_chart(results_new)