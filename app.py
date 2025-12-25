import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import io

# Настройка страницы
st.set_page_config(
    page_title="Визуализация данных с маргинальными распределениями",
    page_icon="📊",
    layout="wide"
)

# Устанавливаем стиль seaborn
sns.set_style("whitegrid")

# Функция для форматирования подписей осей
def format_axis_label(text):
    """Преобразует текст с -1 в степенной формат"""
    replacements = {
        '-1': '⁻¹',
        '-2': '⁻²',
        '-3': '⁻³',
        '-4': '⁻⁴',
        '-5': '⁻⁵',
        '-6': '⁻⁶',
        '-7': '⁻⁷',
        '-8': '⁻⁸',
        '-9': '⁻⁹',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

# Функция для парсинга данных
def parse_data(text, dataset_name):
    """Преобразует текстовые данные в DataFrame"""
    lines = text.strip().split('\n')
    data = []
    for line in lines:
        if line.strip():
            try:
                parts = line.split('\t')
                if len(parts) >= 2:
                    x = float(parts[0].strip())
                    y = float(parts[1].strip())
                    data.append([x, y])
            except:
                continue
    
    if data:
        df = pd.DataFrame(data, columns=['x', 'y'])
        df['group'] = dataset_name
        return df
    return pd.DataFrame()

# Функция для оценки плотности
def estimate_density(data, extend_range=True, padding_factor=0.2):
    """Оценивает плотность распределения"""
    if len(data) > 1:
        kde = gaussian_kde(data)
        
        data_min, data_max = data.min(), data.max()
        data_range = data_max - data_min
        
        if extend_range and data_range > 0:
            x_vals = np.linspace(data_min - padding_factor*data_range, 
                                data_max + padding_factor*data_range, 500)
        else:
            x_vals = np.linspace(data_min, data_max, 500)
        
        density = kde(x_vals)
        
        # Нормируем плотность
        if density.max() > 0:
            density = density / density.max()
        
        return x_vals, density
    return None, None

# Основной заголовок
st.title("📊 Визуализация данных с маргинальными распределениями")
st.markdown("---")

# Инициализация состояния сессии
if 'datasets' not in st.session_state:
    st.session_state.datasets = [
        {
            'name': 'Sample x',
            'data': '0\t-5\n0.2\t-7\n0.1\t-7\n0.15\t-7.5',
            'color': '#E41A1C',
            'marker': 'circle',
            'active': True
        },
        {
            'name': 'Sample y',
            'data': '0.05\t-5\n0.2\t-7\n0.15\t-5.5\n0.15\t-6\n0.15\t-7.5\n0.3\t-5.5',
            'color': '#377EB8',
            'marker': 'square',
            'active': True
        },
        {
            'name': 'Sample z',
            'data': '0.05\t-7\n0.15\t-5\n0.2\t-7.5\n0.2\t-6\n0.1\t-4.5',
            'color': '#4DAF4A',
            'marker': 'triangle-up',
            'active': True
        }
    ]

if 'x_axis_label' not in st.session_state:
    st.session_state.x_axis_label = 'Temperature (°C)'

if 'y_axis_label' not in st.session_state:
    st.session_state.y_axis_label = 'Conductivity (S cm⁻¹)'

# Доступные маркеры Plotly
plotly_markers = {
    'circle': 'circle',
    'square': 'square',
    'triangle-up': 'triangle-up',
    'triangle-down': 'triangle-down',
    'diamond': 'diamond',
    'pentagon': 'pentagon',
    'hexagon': 'hexagon',
    'star': 'star',
    'cross': 'cross',
    'x': 'x'
}

# Цвета по умолчанию
default_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33', '#A65628', '#F781BF', '#999999']

# Боковая панель для настроек
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Названия осей
    st.subheader("Настройка осей")
    st.session_state.x_axis_label = st.text_input(
        "Название оси X",
        value=st.session_state.x_axis_label
    )
    st.session_state.y_axis_label = st.text_input(
        "Название оси Y",
        value=st.session_state.y_axis_label
    )
    
    # Управление осями
    st.subheader("Управление границами осей")
    
    col1, col2 = st.columns(2)
    with col1:
        x_manual = st.checkbox("Настроить ось X", value=False)
    with col2:
        y_manual = st.checkbox("Настроить ось Y", value=False)
    
    if x_manual:
        col1, col2, col3 = st.columns(3)
        with col1:
            x_min = st.number_input("X мин", value=0.0, step=0.1)
        with col2:
            x_max = st.number_input("X макс", value=1.0, step=0.1)
        with col3:
            x_step = st.number_input("X шаг", value=0.1, step=0.1, min_value=0.01)
    else:
        x_min = x_max = x_step = None
    
    if y_manual:
        col1, col2, col3 = st.columns(3)
        with col1:
            y_min = st.number_input("Y мин", value=-10.0, step=0.1)
        with col2:
            y_max = st.number_input("Y макс", value=0.0, step=0.1)
        with col3:
            y_step = st.number_input("Y шаг", value=1.0, step=0.1, min_value=0.01)
    else:
        y_min = y_max = y_step = None
    
    # Управление наборами данных
    st.subheader("Управление наборами данных")
    
    if st.button("➕ Добавить новый набор данных"):
        idx = len(st.session_state.datasets)
        new_dataset = {
            'name': f'Sample {chr(97 + idx)}',
            'data': '',
            'color': default_colors[idx % len(default_colors)],
            'marker': 'circle',
            'active': True
        }
        st.session_state.datasets.append(new_dataset)
    
    if st.button("➖ Удалить последний набор") and len(st.session_state.datasets) > 1:
        st.session_state.datasets.pop()

# Основная область
tab1, tab2, tab3 = st.tabs(["📁 Данные", "📊 Графики", "📈 Статистика"])

with tab1:
    st.header("Настройка наборов данных")
    st.markdown("Введите данные в формате: **X_value<tab>Y_value**")
    st.markdown("Пример: `0.1\t-5.5`")
    
    # Отображение и редактирование наборов данных
    all_data_frames = []
    
    for i, dataset in enumerate(st.session_state.datasets):
        with st.expander(f"Набор данных {i+1}: {dataset['name']}", expanded=True):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                new_name = st.text_input(
                    f"Название набора {i+1}",
                    value=dataset['name'],
                    key=f"name_{i}"
                )
                st.session_state.datasets[i]['name'] = new_name
                
                data_text = st.text_area(
                    "Данные (X\\tY)",
                    value=dataset['data'],
                    height=150,
                    key=f"data_{i}"
                )
                st.session_state.datasets[i]['data'] = data_text
            
            with col2:
                color = st.color_picker(
                    "Цвет",
                    value=dataset['color'],
                    key=f"color_{i}"
                )
                st.session_state.datasets[i]['color'] = color
            
            with col3:
                marker = st.selectbox(
                    "Маркер",
                    options=list(plotly_markers.keys()),
                    index=list(plotly_markers.keys()).index(dataset['marker']),
                    key=f"marker_{i}"
                )
                st.session_state.datasets[i]['marker'] = marker
                
                active = st.checkbox(
                    "Активен",
                    value=dataset['active'],
                    key=f"active_{i}"
                )
                st.session_state.datasets[i]['active'] = active
            
            # Парсим данные для предварительного просмотра
            if data_text.strip():
                df = parse_data(data_text, new_name)
                if not df.empty:
                    all_data_frames.append(df)
                    
                    # Предпросмотр данных
                    st.markdown(f"**Предпросмотр ({len(df)} точек):**")
                    st.dataframe(df[['x', 'y']].head(), use_container_width=True)
    
    # Собираем все данные
    if all_data_frames:
        all_data = pd.concat(all_data_frames, ignore_index=True)
        
        # Обновляем автоматические значения осей
        if not x_manual:
            x_min_val = all_data['x'].min()
            x_max_val = all_data['x'].max()
            x_range = x_max_val - x_min_val
            x_min = max(0, x_min_val - 0.1 * x_range) if x_range > 0 else x_min_val - 0.1
            x_max = x_max_val + 0.1 * x_range if x_range > 0 else x_max_val + 0.1
            x_step = max(x_range / 10, 0.1)
        
        if not y_manual:
            y_min_val = all_data['y'].min()
            y_max_val = all_data['y'].max()
            y_range = y_max_val - y_min_val
            y_min = y_min_val - 0.1 * y_range if y_range > 0 else y_min_val - 0.1
            y_max = y_max_val + 0.1 * y_range if y_range > 0 else y_max_val + 0.1
            y_step = max(y_range / 10, 0.1)

with tab2:
    st.header("Визуализация данных")
    
    # Кнопка для построения графиков
    if st.button("🚀 Построить графики", type="primary"):
        if 'all_data' in locals() and not all_data.empty:
            # Основной график с маргинальными распределениями
            st.subheader("Scatter Plot с маргинальными распределениями")
            
            # Создаем фигуру Matplotlib
            fig, (ax_top, ax_main) = plt.subplots(
                2, 2, 
                figsize=(12, 10),
                gridspec_kw={'height_ratios': [1, 3], 'width_ratios': [3, 1]},
                constrained_layout=True
            )
            
            # Убираем лишние оси
            ax_right = ax_main[1]
            ax_main = ax_main[0]
            ax_top[1].axis('off')
            ax_top = ax_top[0]
            
            # Рисуем точки на основном графике
            for i, dataset in enumerate(st.session_state.datasets):
                if dataset['active']:
                    df = parse_data(dataset['data'], dataset['name'])
                    if not df.empty:
                        ax_main.scatter(
                            df['x'], df['y'],
                            color=dataset['color'],
                            label=dataset['name'],
                            marker=dataset['marker'][0] if dataset['marker'] in ['circle', 'square', 'triangle-up'] else 'o',
                            s=50,
                            alpha=0.7
                        )
            
            # Настройки основного графика
            ax_main.set_xlabel(format_axis_label(st.session_state.x_axis_label), fontsize=12)
            ax_main.set_ylabel(format_axis_label(st.session_state.y_axis_label), fontsize=12)
            ax_main.legend(title='Наборы данных')
            ax_main.grid(True, alpha=0.3)
            
            # Применяем границы осей
            if x_min is not None and x_max is not None:
                ax_main.set_xlim(x_min, x_max)
                ax_top.set_xlim(x_min, x_max)
            if y_min is not None and y_max is not None:
                ax_main.set_ylim(y_min, y_max)
                ax_right.set_ylim(y_min, y_max)
            
            # Рисуем маргинальные распределения
            for i, dataset in enumerate(st.session_state.datasets):
                if dataset['active']:
                    df = parse_data(dataset['data'], dataset['name'])
                    if not df.empty and len(df) > 1:
                        color = dataset['color']
                        
                        # Распределение по X (верхний график)
                        x_vals, density = estimate_density(df['x'].values)
                        if x_vals is not None and density is not None:
                            ax_top.fill_between(x_vals, 0, density, color=color, alpha=0.3)
                            ax_top.plot(x_vals, density, color=color, linewidth=1.5)
                        
                        # Распределение по Y (правый график)
                        y_vals, density = estimate_density(df['y'].values)
                        if y_vals is not None and density is not None:
                            ax_right.fill_betweenx(y_vals, 0, density, color=color, alpha=0.3)
                            ax_right.plot(density, y_vals, color=color, linewidth=1.5)
            
            # Настройки маргинальных графиков
            ax_top.set_ylabel('Density', fontsize=10)
            ax_top.set_ylim(0, 1.1)
            ax_top.tick_params(axis='x', labelbottom=False)
            ax_top.grid(True, alpha=0.3)
            
            ax_right.set_xlabel('Density', fontsize=10)
            ax_right.set_xlim(0, 1.1)
            ax_right.tick_params(axis='y', labelleft=False)
            ax_right.grid(True, alpha=0.3)
            
            # Заголовок
            fig.suptitle('Scatter Plot with Marginal Densities', fontsize=14, fontweight='bold')
            
            st.pyplot(fig)
            
            # Альтернативные графики
            st.subheader("Альтернативное представление")
            
            fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
            
            # 1. Основной scatter plot
            for i, dataset in enumerate(st.session_state.datasets):
                if dataset['active']:
                    df = parse_data(dataset['data'], dataset['name'])
                    if not df.empty:
                        ax1.scatter(df['x'], df['y'], 
                                  color=dataset['color'], 
                                  label=dataset['name'],
                                  marker=dataset['marker'][0] if dataset['marker'] in ['circle', 'square', 'triangle-up'] else 'o',
                                  s=100, alpha=0.7)
            
            ax1.set_title('Scatter Plot: Все образцы')
            ax1.set_xlabel(format_axis_label(st.session_state.x_axis_label))
            ax1.set_ylabel(format_axis_label(st.session_state.y_axis_label))
            ax1.legend(title='Группа')
            ax1.grid(True, alpha=0.3)
            
            # Применяем границы осей
            if x_min is not None and x_max is not None:
                ax1.set_xlim(x_min, x_max)
                ax3.set_xlim(x_min, x_max)
            if y_min is not None and y_max is not None:
                ax1.set_ylim(y_min, y_max)
                ax4.set_ylim(y_min, y_max)
            
            # 2. Второй scatter plot
            for i, dataset in enumerate(st.session_state.datasets):
                if dataset['active']:
                    df = parse_data(dataset['data'], dataset['name'])
                    if not df.empty:
                        ax2.scatter(df['x'], df['y'], 
                                  color=dataset['color'], 
                                  label=dataset['name'],
                                  marker=dataset['marker'][0] if dataset['marker'] in ['circle', 'square', 'triangle-up'] else 'o',
                                  s=100, alpha=0.7)
            
            ax2.set_title('Scatter Plot')
            ax2.set_xlabel(format_axis_label(st.session_state.x_axis_label))
            ax2.set_ylabel(format_axis_label(st.session_state.y_axis_label))
            ax2.legend(title='Группа')
            ax2.grid(True, alpha=0.3)
            
            # 3. KDE для X
            for i, dataset in enumerate(st.session_state.datasets):
                if dataset['active']:
                    df = parse_data(dataset['data'], dataset['name'])
                    if not df.empty and len(df) > 1:
                        color = dataset['color']
                        x_vals, density = estimate_density(df['x'].values)
                        if x_vals is not None and density is not None:
                            ax3.fill_between(x_vals, 0, density, color=color, alpha=0.3)
                            ax3.plot(x_vals, density, color=color, linewidth=2, label=dataset['name'])
            
            ax3.set_title('Распределение по X')
            ax3.set_xlabel(format_axis_label(st.session_state.x_axis_label))
            ax3.set_ylabel('Нормированная плотность')
            ax3.legend(title='Группа')
            ax3.grid(True, alpha=0.3)
            
            # 4. KDE для Y
            for i, dataset in enumerate(st.session_state.datasets):
                if dataset['active']:
                    df = parse_data(dataset['data'], dataset['name'])
                    if not df.empty and len(df) > 1:
                        color = dataset['color']
                        y_vals, density = estimate_density(df['y'].values)
                        if y_vals is not None and density is not None:
                            ax4.fill_between(y_vals, 0, density, color=color, alpha=0.3)
                            ax4.plot(y_vals, density, color=color, linewidth=2, label=dataset['name'])
            
            ax4.set_title('Распределение по Y')
            ax4.set_xlabel(format_axis_label(st.session_state.y_axis_label))
            ax4.set_ylabel('Нормированная плотность')
            ax4.legend(title='Группа')
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle('Анализ данных с маргинальными распределениями', fontsize=16, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig2)
            
            # Интерактивный график Plotly
            st.subheader("Интерактивный график (Plotly)")
            
            fig_plotly = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Scatter Plot: Все образцы', 'Scatter Plot',
                               'Распределение по X', 'Распределение по Y'),
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            # Добавляем scatter plots
            for i, dataset in enumerate(st.session_state.datasets):
                if dataset['active']:
                    df = parse_data(dataset['data'], dataset['name'])
                    if not df.empty:
                        # Scatter plot 1
                        fig_plotly.add_trace(
                            go.Scatter(
                                x=df['x'],
                                y=df['y'],
                                mode='markers',
                                name=dataset['name'],
                                marker=dict(
                                    color=dataset['color'],
                                    symbol=plotly_markers[dataset['marker']],
                                    size=10,
                                    opacity=0.7
                                ),
                                showlegend=True
                            ),
                            row=1, col=1
                        )
                        
                        # Scatter plot 2
                        fig_plotly.add_trace(
                            go.Scatter(
                                x=df['x'],
                                y=df['y'],
                                mode='markers',
                                name=dataset['name'],
                                marker=dict(
                                    color=dataset['color'],
                                    symbol=plotly_markers[dataset['marker']],
                                    size=10,
                                    opacity=0.7
                                ),
                                showlegend=False
                            ),
                            row=1, col=2
                        )
            
            # Обновляем layout
            fig_plotly.update_xaxes(title_text=format_axis_label(st.session_state.x_axis_label), row=1, col=1)
            fig_plotly.update_yaxes(title_text=format_axis_label(st.session_state.y_axis_label), row=1, col=1)
            fig_plotly.update_xaxes(title_text=format_axis_label(st.session_state.x_axis_label), row=1, col=2)
            fig_plotly.update_yaxes(title_text=format_axis_label(st.session_state.y_axis_label), row=1, col=2)
            
            # Применяем границы осей
            if x_min is not None and x_max is not None:
                fig_plotly.update_xaxes(range=[x_min, x_max], row=1, col=1)
                fig_plotly.update_xaxes(range=[x_min, x_max], row=1, col=2)
            if y_min is not None and y_max is not None:
                fig_plotly.update_yaxes(range=[y_min, y_max], row=1, col=1)
                fig_plotly.update_yaxes(range=[y_min, y_max], row=1, col=2)
            
            fig_plotly.update_layout(
                height=800,
                title_text="Интерактивная визуализация данных",
                showlegend=True,
                hovermode='closest'
            )
            
            st.plotly_chart(fig_plotly, use_container_width=True)
            
        else:
            st.warning("Нет данных для отображения! Пожалуйста, введите данные во вкладке 'Данные'.")

with tab3:
    st.header("Статистика данных")
    
    if 'all_data' in locals() and not all_data.empty:
        # Общая статистика
        st.subheader("Общая статистика")
        
        stats_data = []
        for i, dataset in enumerate(st.session_state.datasets):
            if dataset['active']:
                df = parse_data(dataset['data'], dataset['name'])
                if not df.empty:
                    stats = {
                        'Набор данных': dataset['name'],
                        'Количество точек': len(df),
                        'X мин': f"{df['x'].min():.3f}",
                        'X макс': f"{df['x'].max():.3f}",
                        'X среднее': f"{df['x'].mean():.3f}",
                        'X std': f"{df['x'].std():.3f}",
                        'Y мин': f"{df['y'].min():.3f}",
                        'Y макс': f"{df['y'].max():.3f}",
                        'Y среднее': f"{df['y'].mean():.3f}",
                        'Y std': f"{df['y'].std():.3f}"
                    }
                    stats_data.append(stats)
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
            # Экспорт данных
            st.subheader("Экспорт данных")
            
            # CSV
            csv = stats_df.to_csv(index=False, sep='\t').encode('utf-8')
            st.download_button(
                label="📥 Скачать статистику (CSV)",
                data=csv,
                file_name="data_statistics.csv",
                mime="text/csv"
            )
            
            # Экспорт всех данных
            if 'all_data' in locals():
                all_data_csv = all_data.to_csv(index=False, sep='\t').encode('utf-8')
                st.download_button(
                    label="📥 Скачать все данные (CSV)",
                    data=all_data_csv,
                    file_name="all_data.csv",
                    mime="text/csv"
                )
        
        # Настройки осей
        st.subheader("Настройки осей")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Ось X:** {format_axis_label(st.session_state.x_axis_label)}")
            st.write(f"Минимум: {x_min:.3f}" if x_min is not None else "Автоопределение")
            st.write(f"Максимум: {x_max:.3f}" if x_max is not None else "Автоопределение")
            st.write(f"Шаг: {x_step:.3f}" if x_step is not None else "Автоопределение")
        
        with col2:
            st.info(f"**Ось Y:** {format_axis_label(st.session_state.y_axis_label)}")
            st.write(f"Минимум: {y_min:.3f}" if y_min is not None else "Автоопределение")
            st.write(f"Максимум: {y_max:.3f}" if y_max is not None else "Автоопределение")
            st.write(f"Шаг: {y_step:.3f}" if y_step is not None else "Автоопределение")
    
    else:
        st.info("Постройте графики во вкладке 'Графики' для отображения статистики.")

# Футер
st.markdown("---")
st.markdown("### Инструкция по использованию:")
st.markdown("""
1. **Вкладка 'Данные'**: Настройте наборы данных, введите значения X и Y через табуляцию
2. **Боковая панель**: Задайте названия осей и границы (опционально)
3. **Вкладка 'Графики'**: Нажмите кнопку "Построить графики" для визуализации
4. **Вкладка 'Статистика'**: Просмотрите статистику данных и экспортируйте результаты
""")