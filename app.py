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
import json
import base64

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

# Функция для экспорта всех данных с настройками
def export_all_data_with_settings(datasets, x_label, y_label, x_manual, y_manual, 
                                 x_min_val, x_max_val, x_step_val, 
                                 y_min_val, y_max_val, y_step_val):
    """Создает CSV файл с данными и настройками"""
    
    # Создаем структуру для экспорта
    export_dict = {
        'metadata': {
            'version': '1.0',
            'x_axis_label': x_label,
            'y_axis_label': y_label,
            'num_datasets': len(datasets),
            'export_timestamp': pd.Timestamp.now().isoformat()
        },
        'axis_settings': {
            'x_manual': x_manual,
            'y_manual': y_manual,
            'x_min': x_min_val if x_min_val is not None else '',
            'x_max': x_max_val if x_max_val is not None else '',
            'x_step': x_step_val if x_step_val is not None else '',
            'y_min': y_min_val if y_min_val is not None else '',
            'y_max': y_max_val if y_max_val is not None else '',
            'y_step': y_step_val if y_step_val is not None else ''
        },
        'settings': [],
        'data': []
    }
    
    # Сохраняем настройки каждого набора данных
    for i, dataset in enumerate(datasets):
        dataset_settings = {
            'dataset_index': i,
            'name': dataset['name'],
            'color': dataset['color'],
            'marker': dataset['marker'],
            'active': dataset['active']
        }
        export_dict['settings'].append(dataset_settings)
        
        # Сохраняем данные набора
        if dataset['data'].strip():
            df = parse_data(dataset['data'], dataset['name'])
            if not df.empty:
                for _, row in df.iterrows():
                    data_point = {
                        'dataset_index': i,
                        'dataset_name': dataset['name'],
                        'x': row['x'],
                        'y': row['y']
                    }
                    export_dict['data'].append(data_point)
    
    # Создаем CSV-совместимую структуру
    lines = []
    
    # 1. Метаданные
    lines.append("# META DATA SECTION")
    lines.append(f"x_axis_label: {x_label}")
    lines.append(f"y_axis_label: {y_label}")
    lines.append(f"num_datasets: {len(datasets)}")
    lines.append(f"export_timestamp: {export_dict['metadata']['export_timestamp']}")
    lines.append("")
    
    # 2. Настройки осей
    lines.append("# AXIS SETTINGS SECTION")
    lines.append("setting,value")
    lines.append(f"x_manual,{export_dict['axis_settings']['x_manual']}")
    lines.append(f"y_manual,{export_dict['axis_settings']['y_manual']}")
    lines.append(f"x_min,{export_dict['axis_settings']['x_min']}")
    lines.append(f"x_max,{export_dict['axis_settings']['x_max']}")
    lines.append(f"x_step,{export_dict['axis_settings']['x_step']}")
    lines.append(f"y_min,{export_dict['axis_settings']['y_min']}")
    lines.append(f"y_max,{export_dict['axis_settings']['y_max']}")
    lines.append(f"y_step,{export_dict['axis_settings']['y_step']}")
    lines.append("")
    
    # 3. Настройки наборов данных
    lines.append("# DATASET SETTINGS SECTION")
    lines.append("index,name,color,marker,active")
    for settings in export_dict['settings']:
        lines.append(f"{settings['dataset_index']},{settings['name']},{settings['color']},{settings['marker']},{settings['active']}")
    lines.append("")
    
    # 4. Данные
    lines.append("# DATA POINTS SECTION")
    lines.append("dataset_index,dataset_name,x,y")
    for data_point in export_dict['data']:
        lines.append(f"{data_point['dataset_index']},{data_point['dataset_name']},{data_point['x']},{data_point['y']}")
    
    return "\n".join(lines)

# Функция для импорта данных с настройками
def import_data_with_settings(file_content):
    """Импортирует данные и настройки из CSV файла"""
    
    lines = file_content.strip().split('\n')
    
    # Инициализируем переменные
    x_axis_label = "Temperature (°C)"
    y_axis_label = "Conductivity (S cm⁻¹)"
    x_manual = False
    y_manual = False
    x_min = None
    x_max = None
    x_step = None
    y_min = None
    y_max = None
    y_step = None
    datasets_settings = []
    data_points = []
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        # Пропускаем пустые строки
        if not line:
            continue
            
        # Определяем секцию
        if line.startswith("# META DATA SECTION"):
            current_section = "metadata"
            continue
        elif line.startswith("# AXIS SETTINGS SECTION"):
            current_section = "axis_settings"
            continue
        elif line.startswith("# DATASET SETTINGS SECTION"):
            current_section = "settings"
            continue
        elif line.startswith("# DATA POINTS SECTION"):
            current_section = "data"
            continue
        elif line.startswith("#"):
            continue
        
        # Обрабатываем метаданные
        if current_section == "metadata":
            if line.startswith("x_axis_label:"):
                x_axis_label = line.split(":", 1)[1].strip()
            elif line.startswith("y_axis_label:"):
                y_axis_label = line.split(":", 1)[1].strip()
        
        # Обрабатываем настройки осей
        elif current_section == "axis_settings":
            if line.startswith("setting,value"):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                setting_name = parts[0].strip()
                setting_value = parts[1].strip()
                
                try:
                    if setting_name == "x_manual":
                        x_manual = setting_value.lower() == 'true'
                    elif setting_name == "y_manual":
                        y_manual = setting_value.lower() == 'true'
                    elif setting_name == "x_min" and setting_value:
                        x_min = float(setting_value)
                    elif setting_name == "x_max" and setting_value:
                        x_max = float(setting_value)
                    elif setting_name == "x_step" and setting_value:
                        x_step = float(setting_value)
                    elif setting_name == "y_min" and setting_value:
                        y_min = float(setting_value)
                    elif setting_name == "y_max" and setting_value:
                        y_max = float(setting_value)
                    elif setting_name == "y_step" and setting_value:
                        y_step = float(setting_value)
                except:
                    continue
        
        # Обрабатываем настройки
        elif current_section == "settings":
            if line.startswith("index,name,color,marker,active"):
                continue
            parts = line.split(',')
            if len(parts) >= 5:
                try:
                    dataset_setting = {
                        'index': int(parts[0]),
                        'name': parts[1],
                        'color': parts[2],
                        'marker': parts[3],
                        'active': parts[4].lower() == 'true'
                    }
                    datasets_settings.append(dataset_setting)
                except:
                    continue
        
        # Обрабатываем данные
        elif current_section == "data":
            if line.startswith("dataset_index,dataset_name,x,y"):
                continue
            parts = line.split(',')
            if len(parts) >= 4:
                try:
                    data_point = {
                        'dataset_index': int(parts[0]),
                        'dataset_name': parts[1],
                        'x': float(parts[2]),
                        'y': float(parts[3])
                    }
                    data_points.append(data_point)
                except:
                    continue
    
    # Восстанавливаем наборы данных
    datasets = []
    
    # Группируем данные по наборам
    data_by_dataset = {}
    for dp in data_points:
        idx = dp['dataset_index']
        if idx not in data_by_dataset:
            data_by_dataset[idx] = []
        data_by_dataset[idx].append(f"{dp['x']}\t{dp['y']}")
    
    # Создаем структуру datasets
    for setting in datasets_settings:
        idx = setting['index']
        data_text = ""
        if idx in data_by_dataset:
            data_text = "\n".join(data_by_dataset[idx])
        
        dataset = {
            'name': setting['name'],
            'data': data_text,
            'color': setting['color'],
            'marker': setting['marker'],
            'active': setting['active']
        }
        datasets.append(dataset)
    
    axis_settings = {
        'x_manual': x_manual,
        'y_manual': y_manual,
        'x_min': x_min,
        'x_max': x_max,
        'x_step': x_step,
        'y_min': y_min,
        'y_max': y_max,
        'y_step': y_step
    }
    
    return datasets, x_axis_label, y_axis_label, axis_settings

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

# Инициализация состояния для настроек осей
if 'x_manual' not in st.session_state:
    st.session_state.x_manual = False

if 'y_manual' not in st.session_state:
    st.session_state.y_manual = False

if 'x_min' not in st.session_state:
    st.session_state.x_min = None

if 'x_max' not in st.session_state:
    st.session_state.x_max = None

if 'x_step' not in st.session_state:
    st.session_state.x_step = None

if 'y_min' not in st.session_state:
    st.session_state.y_min = None

if 'y_max' not in st.session_state:
    st.session_state.y_max = None

if 'y_step' not in st.session_state:
    st.session_state.y_step = None

# Инициализация состояния для импортированных данных
if 'imported_file_content' not in st.session_state:
    st.session_state.imported_file_content = None

if 'imported_datasets' not in st.session_state:
    st.session_state.imported_datasets = None

if 'imported_x_label' not in st.session_state:
    st.session_state.imported_x_label = None

if 'imported_y_label' not in st.session_state:
    st.session_state.imported_y_label = None

if 'imported_axis_settings' not in st.session_state:
    st.session_state.imported_axis_settings = None

# Флаг для применения импортированных данных
if 'apply_imported_data' not in st.session_state:
    st.session_state.apply_imported_data = False

# Доступные маркеры для matplotlib и Plotly
matplotlib_markers = {
    'circle': 'o',
    'square': 's',
    'triangle-up': '^',
    'triangle-down': 'v',
    'diamond': 'D',
    'pentagon': 'p',
    'hexagon': 'h',
    'star': '*',
    'plus': '+',
    'x': 'x',
    'point': '.'
}

plotly_markers = {
    'circle': 'circle',
    'square': 'square',
    'triangle-up': 'triangle-up',
    'triangle-down': 'triangle-down',
    'diamond': 'diamond',
    'pentagon': 'pentagon',
    'hexagon': 'hexagon',
    'star': 'star',
    'plus': 'cross',
    'x': 'x',
    'point': 'circle-open'
}

# Цвета по умолчанию
default_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33', '#A65628', '#F781BF', '#999999']

# Боковая панель для настроек
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Кнопка для импорта данных
    st.subheader("Импорт/Экспорт")
    
    uploaded_file = st.file_uploader(
        "Загрузить данные с настройками",
        type=['csv', 'txt'],
        help="Загрузите файл, ранее экспортированный из этого приложения"
    )
    
    if uploaded_file is not None:
        try:
            file_content = uploaded_file.getvalue().decode('utf-8')
            imported_datasets, imported_x_label, imported_y_label, imported_axis_settings = import_data_with_settings(file_content)
            
            if imported_datasets:
                st.session_state.imported_file_content = file_content
                st.session_state.imported_datasets = imported_datasets
                st.session_state.imported_x_label = imported_x_label
                st.session_state.imported_y_label = imported_y_label
                st.session_state.imported_axis_settings = imported_axis_settings
                
                st.success(f"Файл загружен! Обнаружено {len(imported_datasets)} наборов данных.")
                st.info("Нажмите кнопку 'Применить загруженные данные' ниже, чтобы использовать эти настройки.")
        except Exception as e:
            st.error(f"Ошибка при загрузке файла: {str(e)}")
    
    # Кнопка для применения загруженных данных
    if st.session_state.imported_datasets is not None:
        if st.button("✅ Применить загруженные данные", type="primary"):
            # Устанавливаем флаг для применения данных
            st.session_state.apply_imported_data = True
            st.rerun()
    
    # Названия осей
    st.subheader("Настройка осей")
    st.session_state.x_axis_label = st.text_input(
        "Название оси X",
        value=st.session_state.x_axis_label,
        key="x_axis_label_input"
    )
    st.session_state.y_axis_label = st.text_input(
        "Название оси Y",
        value=st.session_state.y_axis_label,
        key="y_axis_label_input"
    )
    
    # Управление осями
    st.subheader("Управление границами осей")
    
    col1, col2 = st.columns(2)
    with col1:
        x_manual = st.checkbox("Настроить ось X", 
                              value=st.session_state.x_manual,
                              key="x_manual_checkbox")
        st.session_state.x_manual = x_manual
    with col2:
        y_manual = st.checkbox("Настроить ось Y", 
                              value=st.session_state.y_manual,
                              key="y_manual_checkbox")
        st.session_state.y_manual = y_manual
    
    if x_manual:
        col1, col2, col3 = st.columns(3)
        with col1:
            x_min = st.number_input("X мин", 
                                   value=float(st.session_state.x_min) if st.session_state.x_min is not None else 0.0, 
                                   step=0.1,
                                   key="x_min_input")
            st.session_state.x_min = x_min
        with col2:
            x_max = st.number_input("X макс", 
                                   value=float(st.session_state.x_max) if st.session_state.x_max is not None else 1.0, 
                                   step=0.1,
                                   key="x_max_input")
            st.session_state.x_max = x_max
        with col3:
            x_step = st.number_input("X шаг", 
                                    value=float(st.session_state.x_step) if st.session_state.x_step is not None else 0.1, 
                                    step=0.1, 
                                    min_value=0.01,
                                    key="x_step_input")
            st.session_state.x_step = x_step
    else:
        # Сбрасываем значения при отключении ручной настройки
        st.session_state.x_min = None
        st.session_state.x_max = None
        st.session_state.x_step = None
    
    if y_manual:
        col1, col2, col3 = st.columns(3)
        with col1:
            y_min = st.number_input("Y мин", 
                                   value=float(st.session_state.y_min) if st.session_state.y_min is not None else -10.0, 
                                   step=0.1,
                                   key="y_min_input")
            st.session_state.y_min = y_min
        with col2:
            y_max = st.number_input("Y макс", 
                                   value=float(st.session_state.y_max) if st.session_state.y_max is not None else 0.0, 
                                   step=0.1,
                                   key="y_max_input")
            st.session_state.y_max = y_max
        with col3:
            y_step = st.number_input("Y шаг", 
                                    value=float(st.session_state.y_step) if st.session_state.y_step is not None else 1.0, 
                                    step=0.1, 
                                    min_value=0.01,
                                    key="y_step_input")
            st.session_state.y_step = y_step
    else:
        # Сбрасываем значения при отключении ручной настройки
        st.session_state.y_min = None
        st.session_state.y_max = None
        st.session_state.y_step = None
    
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

# Применяем импортированные данные (если установлен флаг)
if st.session_state.apply_imported_data and st.session_state.imported_datasets is not None:
    # ПОЛНОСТЬЮ заменяем datasets на импортированные
    st.session_state.datasets = st.session_state.imported_datasets.copy()
    
    st.session_state.x_axis_label = st.session_state.imported_x_label
    st.session_state.y_axis_label = st.session_state.imported_y_label
    
    # Применяем настройки осей
    if st.session_state.imported_axis_settings:
        st.session_state.x_manual = st.session_state.imported_axis_settings['x_manual']
        st.session_state.y_manual = st.session_state.imported_axis_settings['y_manual']
        st.session_state.x_min = st.session_state.imported_axis_settings['x_min']
        st.session_state.x_max = st.session_state.imported_axis_settings['x_max']
        st.session_state.x_step = st.session_state.imported_axis_settings['x_step']
        st.session_state.y_min = st.session_state.imported_axis_settings['y_min']
        st.session_state.y_max = st.session_state.imported_axis_settings['y_max']
        st.session_state.y_step = st.session_state.imported_axis_settings['y_step']
    
    # Сбрасываем состояние импорта
    st.session_state.imported_file_content = None
    st.session_state.imported_datasets = None
    st.session_state.imported_x_label = None
    st.session_state.imported_y_label = None
    st.session_state.imported_axis_settings = None
    st.session_state.apply_imported_data = False
    
    st.success("Данные успешно применены!")
    st.rerun()

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
                    options=list(matplotlib_markers.keys()),
                    index=list(matplotlib_markers.keys()).index(dataset['marker']),
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
        
        # Обновляем автоматические значения осей, если не заданы вручную
        if not st.session_state.x_manual:
            x_min_val = all_data['x'].min()
            x_max_val = all_data['x'].max()
            x_range = x_max_val - x_min_val
            auto_x_min = max(0, x_min_val - 0.1 * x_range) if x_range > 0 else x_min_val - 0.1
            auto_x_max = x_max_val + 0.1 * x_range if x_range > 0 else x_max_val + 0.1
            auto_x_step = max(x_range / 10, 0.1)
        else:
            auto_x_min = st.session_state.x_min
            auto_x_max = st.session_state.x_max
            auto_x_step = st.session_state.x_step
        
        if not st.session_state.y_manual:
            y_min_val = all_data['y'].min()
            y_max_val = all_data['y'].max()
            y_range = y_max_val - y_min_val
            auto_y_min = y_min_val - 0.1 * y_range if y_range > 0 else y_min_val - 0.1
            auto_y_max = y_max_val + 0.1 * y_range if y_range > 0 else y_max_val + 0.1
            auto_y_step = max(y_range / 10, 0.1)
        else:
            auto_y_min = st.session_state.y_min
            auto_y_max = st.session_state.y_max
            auto_y_step = st.session_state.y_step

with tab2:
    st.header("Визуализация данных")
    
    # Кнопка для построения графиков
    if st.button("🚀 Построить графики", type="primary"):
        # Собираем все данные для проверки
        all_data_frames_local = []
        for dataset in st.session_state.datasets:
            if dataset['active']:
                df = parse_data(dataset['data'], dataset['name'])
                if not df.empty:
                    all_data_frames_local.append(df)
        
        if all_data_frames_local:
            all_data = pd.concat(all_data_frames_local, ignore_index=True)
            
            # Обновляем автоматические значения осей, если не заданы вручную
            if not st.session_state.x_manual:
                x_min_val = all_data['x'].min()
                x_max_val = all_data['x'].max()
                x_range = x_max_val - x_min_val
                auto_x_min = max(0, x_min_val - 0.1 * x_range) if x_range > 0 else x_min_val - 0.1
                auto_x_max = x_max_val + 0.1 * x_range if x_range > 0 else x_max_val + 0.1
                auto_x_step = max(x_range / 10, 0.1)
            else:
                auto_x_min = st.session_state.x_min
                auto_x_max = st.session_state.x_max
                auto_x_step = st.session_state.x_step
            
            if not st.session_state.y_manual:
                y_min_val = all_data['y'].min()
                y_max_val = all_data['y'].max()
                y_range = y_max_val - y_min_val
                auto_y_min = y_min_val - 0.1 * y_range if y_range > 0 else y_min_val - 0.1
                auto_y_max = y_max_val + 0.1 * y_range if y_range > 0 else y_max_val + 0.1
                auto_y_step = max(y_range / 10, 0.1)
            else:
                auto_y_min = st.session_state.y_min
                auto_y_max = st.session_state.y_max
                auto_y_step = st.session_state.y_step
            
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
                            marker=matplotlib_markers[dataset['marker']],
                            s=50,
                            alpha=0.7
                        )
            
            # Настройки основного графика
            ax_main.set_xlabel(format_axis_label(st.session_state.x_axis_label), fontsize=12)
            ax_main.set_ylabel(format_axis_label(st.session_state.y_axis_label), fontsize=12)
            ax_main.legend(title='Наборы данных')
            ax_main.grid(True, alpha=0.3)
            
            # Применяем границы осей
            if st.session_state.x_manual and st.session_state.x_min is not None and st.session_state.x_max is not None:
                ax_main.set_xlim(st.session_state.x_min, st.session_state.x_max)
                ax_top.set_xlim(st.session_state.x_min, st.session_state.x_max)
            elif 'auto_x_min' in locals() and 'auto_x_max' in locals():
                ax_main.set_xlim(auto_x_min, auto_x_max)
                ax_top.set_xlim(auto_x_min, auto_x_max)
            
            if st.session_state.y_manual and st.session_state.y_min is not None and st.session_state.y_max is not None:
                ax_main.set_ylim(st.session_state.y_min, st.session_state.y_max)
                ax_right.set_ylim(st.session_state.y_min, st.session_state.y_max)
            elif 'auto_y_min' in locals() and 'auto_y_max' in locals():
                ax_main.set_ylim(auto_y_min, auto_y_max)
                ax_right.set_ylim(auto_y_min, auto_y_max)
            
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
                                  marker=matplotlib_markers[dataset['marker']],
                                  s=100, alpha=0.7)
            
            ax1.set_title('Scatter Plot: Все образцы')
            ax1.set_xlabel(format_axis_label(st.session_state.x_axis_label))
            ax1.set_ylabel(format_axis_label(st.session_state.y_axis_label))
            ax1.legend(title='Группа')
            ax1.grid(True, alpha=0.3)
            
            # Применяем границы осей
            if st.session_state.x_manual and st.session_state.x_min is not None and st.session_state.x_max is not None:
                ax1.set_xlim(st.session_state.x_min, st.session_state.x_max)
                ax3.set_xlim(st.session_state.x_min, st.session_state.x_max)
            elif 'auto_x_min' in locals() and 'auto_x_max' in locals():
                ax1.set_xlim(auto_x_min, auto_x_max)
                ax3.set_xlim(auto_x_min, auto_x_max)
            
            if st.session_state.y_manual and st.session_state.y_min is not None and st.session_state.y_max is not None:
                ax1.set_ylim(st.session_state.y_min, st.session_state.y_max)
                ax4.set_ylim(st.session_state.y_min, st.session_state.y_max)
            elif 'auto_y_min' in locals() and 'auto_y_max' in locals():
                ax1.set_ylim(auto_y_min, auto_y_max)
                ax4.set_ylim(auto_y_min, auto_y_max)
            
            # 2. Второй scatter plot
            for i, dataset in enumerate(st.session_state.datasets):
                if dataset['active']:
                    df = parse_data(dataset['data'], dataset['name'])
                    if not df.empty:
                        ax2.scatter(df['x'], df['y'], 
                                  color=dataset['color'], 
                                  label=dataset['name'],
                                  marker=matplotlib_markers[dataset['marker']],
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
                                    symbol=plotly_markers.get(dataset['marker'], 'circle'),
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
            if st.session_state.x_manual and st.session_state.x_min is not None and st.session_state.x_max is not None:
                fig_plotly.update_xaxes(range=[st.session_state.x_min, st.session_state.x_max], row=1, col=1)
                fig_plotly.update_xaxes(range=[st.session_state.x_min, st.session_state.x_max], row=1, col=2)
            elif 'auto_x_min' in locals() and 'auto_x_max' in locals():
                fig_plotly.update_xaxes(range=[auto_x_min, auto_x_max], row=1, col=1)
                fig_plotly.update_xaxes(range=[auto_x_min, auto_x_max], row=1, col=2)
            
            if st.session_state.y_manual and st.session_state.y_min is not None and st.session_state.y_max is not None:
                fig_plotly.update_yaxes(range=[st.session_state.y_min, st.session_state.y_max], row=1, col=1)
                fig_plotly.update_yaxes(range=[st.session_state.y_min, st.session_state.y_max], row=1, col=2)
            elif 'auto_y_min' in locals() and 'auto_y_max' in locals():
                fig_plotly.update_yaxes(range=[auto_y_min, auto_y_max], row=1, col=1)
                fig_plotly.update_yaxes(range=[auto_y_min, auto_y_max], row=1, col=2)
            
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
    
    # Собираем все данные для статистики
    stats_data_frames = []
    for dataset in st.session_state.datasets:
        if dataset['active']:
            df = parse_data(dataset['data'], dataset['name'])
            if not df.empty:
                stats_data_frames.append(df)
    
    if stats_data_frames:
        all_data = pd.concat(stats_data_frames, ignore_index=True)
        
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
            
            # CSV со статистикой
            csv_stats = stats_df.to_csv(index=False, sep='\t').encode('utf-8')
            st.download_button(
                label="📥 Скачать статистику (CSV)",
                data=csv_stats,
                file_name="data_statistics.csv",
                mime="text/csv"
            )
            
            # Экспорт всех данных с настройками
            all_data_with_settings = export_all_data_with_settings(
                st.session_state.datasets,
                st.session_state.x_axis_label,
                st.session_state.y_axis_label,
                st.session_state.x_manual,
                st.session_state.y_manual,
                st.session_state.x_min,
                st.session_state.x_max,
                st.session_state.x_step,
                st.session_state.y_min,
                st.session_state.y_max,
                st.session_state.y_step
            )
            
            st.download_button(
                label="📥 Скачать ВСЕ данные с настройками (CSV)",
                data=all_data_with_settings.encode('utf-8'),
                file_name="all_data_with_settings.csv",
                mime="text/csv",
                help="Этот файл содержит все данные и настройки. Его можно загрузить обратно в приложение."
            )
            
            # Предпросмотр экспортируемых данных
            with st.expander("Предпросмотр экспортируемых данных с настройками"):
                st.code(all_data_with_settings[:2000], language='text')
                if len(all_data_with_settings) > 2000:
                    st.info(f"И ещё {len(all_data_with_settings) - 2000} символов...")
        
        # Настройки осей
        st.subheader("Настройки осей")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Ось X:** {format_axis_label(st.session_state.x_axis_label)}")
            if st.session_state.x_manual:
                st.write(f"Ручная настройка: ВКЛ")
                st.write(f"Минимум: {st.session_state.x_min:.3f}" if st.session_state.x_min is not None else "Не задано")
                st.write(f"Максимум: {st.session_state.x_max:.3f}" if st.session_state.x_max is not None else "Не задано")
                st.write(f"Шаг: {st.session_state.x_step:.3f}" if st.session_state.x_step is not None else "Не задано")
            else:
                st.write("Ручная настройка: ВЫКЛ")
                if 'auto_x_min' in locals() and 'auto_x_max' in locals():
                    st.write(f"Автоопределение: от {auto_x_min:.3f} до {auto_x_max:.3f}")
        
        with col2:
            st.info(f"**Ось Y:** {format_axis_label(st.session_state.y_axis_label)}")
            if st.session_state.y_manual:
                st.write(f"Ручная настройка: ВКЛ")
                st.write(f"Минимум: {st.session_state.y_min:.3f}" if st.session_state.y_min is not None else "Не задано")
                st.write(f"Максимум: {st.session_state.y_max:.3f}" if st.session_state.y_max is not None else "Не задано")
                st.write(f"Шаг: {st.session_state.y_step:.3f}" if st.session_state.y_step is not None else "Не задано")
            else:
                st.write("Ручная настройка: ВЫКЛ")
                if 'auto_y_min' in locals() and 'auto_y_max' in locals():
                    st.write(f"Автоопределение: от {auto_y_min:.3f} до {auto_y_max:.3f}")
        
        # Информация о наборах данных
        st.subheader("Информация о наборах данных")
        for i, dataset in enumerate(st.session_state.datasets):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"**{dataset['name']}**")
            with col2:
                color_box = f'<span style="display: inline-block; width: 20px; height: 20px; background-color: {dataset["color"]}; border-radius: 3px;"></span>'
                st.markdown(f"Цвет: {color_box}", unsafe_allow_html=True)
            with col3:
                st.markdown(f"Маркер: {dataset['marker']}")
            with col4:
                status = "✅ Активен" if dataset['active'] else "❌ Не активен"
                st.markdown(status)
                
    else:
        st.info("Постройте графики во вкладке 'Графики' для отображения статистики.")

# Футер
st.markdown("---")
st.markdown("### Инструкция по использованию:")
st.markdown("""
1. **Вкладка 'Данные'**: Настройте наборы данных, введите значения X и Y через табуляцию
2. **Боковая панель**: 
   - Задайте названия осей и границы (опционально)
   - Загрузите ранее экспортированные данные с настройками
   - Нажмите кнопку "Применить загруженные данные" для использования импортированных настроек
3. **Вкладка 'Графики'**: Нажмите кнопку "Построить графики" для визуализации
4. **Вкладка 'Статистика'**: 
   - Просмотрите статистику данных
   - Экспортируйте статистику отдельно
   - Экспортируйте ВСЕ данные с настройками для последующей загрузки

**Важно**: Файл "Скачать ВСЕ данные с настройками" содержит все параметры и может быть загружен обратно через боковую панель.
""")
