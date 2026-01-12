import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import re
import io
import json
import base64
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Data Visualization with Marginal Distributions",
    page_icon="📊",
    layout="wide"
)

# Set seaborn style
sns.set_style("whitegrid")

# Function to format axis labels
def format_axis_label(text):
    """Converts text with -1 to superscript format"""
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

# Function to parse data with multiple separator support
def parse_data(text, dataset_name):
    """Converts text data to DataFrame with support for multiple separators"""
    lines = text.strip().split('\n')
    data = []
    
    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Skip comment lines starting with #
        if line.startswith('#'):
            continue
            
        # Try different separators
        separators = ['\t', ';', ',', ' ']
        parts = None
        
        for sep in separators:
            if sep in line:
                # Special handling for space - need to avoid splitting single numbers with spaces
                if sep == ' ':
                    # Check if this is likely space-separated data (not just spaces within numbers)
                    temp_parts = line.split()
                    if len(temp_parts) >= 2:
                        try:
                            # Try to convert first two parts to float
                            float(temp_parts[0])
                            float(temp_parts[1])
                            parts = temp_parts
                            break
                        except:
                            continue
                else:
                    parts = line.split(sep)
                    if len(parts) >= 2:
                        # Clean up each part
                        parts = [p.strip() for p in parts]
                        break
        
        if parts and len(parts) >= 2:
            try:
                # Clean the values - remove any non-numeric characters except decimal point and minus
                x_val = parts[0].strip()
                y_val = parts[1].strip()
                
                # Replace comma decimal separator with dot if needed
                if ',' in x_val and '.' not in x_val:
                    x_val = x_val.replace(',', '.')
                if ',' in y_val and '.' not in y_val:
                    y_val = y_val.replace(',', '.')
                
                # Remove any non-numeric characters except decimal point, minus, and 'e' for scientific notation
                x_val = ''.join(c for c in x_val if c.isdigit() or c in '.-eE')
                y_val = ''.join(c for c in y_val if c.isdigit() or c in '.-eE')
                
                x = float(x_val) if x_val else None
                y = float(y_val) if y_val else None
                
                if x is not None and y is not None:
                    data.append([x, y])
            except Exception as e:
                # If conversion fails, try alternative approach
                try:
                    # Try direct conversion after removing whitespace
                    x = float(parts[0].replace(',', '.').strip())
                    y = float(parts[1].replace(',', '.').strip())
                    data.append([x, y])
                except:
                    # Skip this line if it can't be parsed
                    continue
        else:
            # Try to extract numbers from the line
            import re
            numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
            if len(numbers) >= 2:
                try:
                    x = float(numbers[0].replace(',', '.'))
                    y = float(numbers[1].replace(',', '.'))
                    data.append([x, y])
                except:
                    continue
    
    if data:
        df = pd.DataFrame(data, columns=['x', 'y'])
        df['group'] = dataset_name
        return df
    return pd.DataFrame()


# Function to format data for display in text area - converts to tab-separated
def format_data_for_display(data_text):
    """Formats data for display in text area - converts to tab-separated"""
    lines = data_text.strip().split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip comment lines
        if line.startswith('#'):
            formatted_lines.append(line)
            continue
            
        # Try different separators
        separators = ['\t', ';', ',', ' ']
        parts = None
        
        for sep in separators:
            if sep in line:
                # Special handling for space
                if sep == ' ':
                    temp_parts = line.split()
                    if len(temp_parts) >= 2:
                        try:
                            float(temp_parts[0])
                            float(temp_parts[1])
                            parts = temp_parts
                            break
                        except:
                            continue
                else:
                    parts = line.split(sep)
                    if len(parts) >= 2:
                        parts = [p.strip() for p in parts]
                        break
        
        if parts and len(parts) >= 2:
            # Format as tab-separated
            formatted_lines.append(f"{parts[0]}\t{parts[1]}")
        else:
            # If no separator found, keep original line
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)


# Function to detect the most likely separator used in the data
def detect_separator(text):
    """Detects the most likely separator used in the data"""
    lines = text.strip().split('\n')
    
    # Count occurrences of each separator in non-empty lines
    separator_counts = {'\t': 0, ';': 0, ',': 0}
    space_as_separator = 0
    
    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Check for tab
        if '\t' in line:
            separator_counts['\t'] += 1
        
        # Check for semicolon
        if ';' in line:
            separator_counts[';'] += 1
        
        # Check for comma
        if ',' in line:
            separator_counts[','] += 1
        
        # Check if space is likely separator (has at least 2 numbers separated by spaces)
        parts = line.split()
        if len(parts) >= 2:
            try:
                float(parts[0].replace(',', '.'))
                float(parts[1].replace(',', '.'))
                space_as_separator += 1
            except:
                pass
    
    # Determine which separator is most common
    max_count = 0
    likely_separator = '\t'  # Default
    
    for sep, count in separator_counts.items():
        if count > max_count:
            max_count = count
            likely_separator = sep
    
    # Check if space is more common
    if space_as_separator > max_count:
        likely_separator = ' '
    
    return likely_separator

# Improved density estimation function with smooth transition to zero
def estimate_density(data, extend_range=True, padding_factor=0.3, normalize=True):
    """Estimates density distribution with smooth transition to zero"""
    if len(data) > 1:
        kde = gaussian_kde(data)
        
        data_min, data_max = data.min(), data.max()
        data_range = data_max - data_min
        
        if extend_range and data_range > 0:
            # Extend range for smooth transition to zero
            extended_min = data_min - padding_factor * data_range
            extended_max = data_max + padding_factor * data_range
            
            # Create points for density estimation
            x_vals = np.linspace(extended_min, extended_max, 1000)
            
            # Estimate density
            density = kde(x_vals)
            
            # Apply window function for smooth transition at edges
            # Use cosine window for smooth transition
            window_size = padding_factor * data_range
            left_window = (x_vals - extended_min) / window_size
            right_window = (extended_max - x_vals) / window_size
            
            # Create window function (cosine taper)
            window = np.ones_like(x_vals)
            
            # Smooth reduction on left edge
            mask_left = left_window < 1.0
            if np.any(mask_left):
                window[mask_left] = 0.5 * (1 - np.cos(np.pi * left_window[mask_left]))
            
            # Smooth reduction on right edge
            mask_right = right_window < 1.0
            if np.any(mask_right):
                window[mask_right] = np.minimum(window[mask_right], 
                                               0.5 * (1 - np.cos(np.pi * right_window[mask_right])))
            
            # Apply window function
            density = density * window
            
            # Normalize density for display (0-1) if requested
            if normalize and density.max() > 0:
                density = density / density.max()
            
            return x_vals, density
        else:
            # If not extending range, simply estimate density
            x_vals = np.linspace(data_min, data_max, 500)
            density = kde(x_vals)
            
            # Normalize density if requested
            if normalize and density.max() > 0:
                density = density / density.max()
            
            return x_vals, density
    return None, None

# Function to configure axes with customizable borders
def set_custom_axes(ax, border_color='black', border_width=2.5, grid_color='lightgray', 
                    grid_visible=True, background_color='white', tick_fontsize=10, 
                    label_fontsize=12, title_fontsize=14):
    """Sets customizable borders, grid, and font sizes for plot axes"""
    # Set background color
    ax.set_facecolor(background_color)
    
    # Make all borders (spines) with custom color and width
    for spine in ax.spines.values():
        spine.set_linewidth(border_width)
        spine.set_color(border_color)
    
    # Also make tick marks with custom color
    ax.tick_params(axis='both', which='major', width=2, color=border_color)
    ax.tick_params(axis='both', which='minor', width=2, color=border_color)
    
    # Set tick label font size
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize-2)
    
    # Set axis label color and font size
    ax.xaxis.label.set_color(border_color)
    ax.yaxis.label.set_color(border_color)
    ax.xaxis.label.set_fontsize(label_fontsize)
    ax.yaxis.label.set_fontsize(label_fontsize)
    
    # Set title color and font size (if any)
    if ax.get_title():
        ax.title.set_color(border_color)
        ax.title.set_fontsize(title_fontsize)
    
    # Configure grid
    if grid_visible:
        ax.grid(True, linewidth=1, alpha=0.3, color=grid_color)
    else:
        ax.grid(False)
    
    return ax

# Function for automatic axis limits detection
def auto_detect_axis_limits(datasets):
    """Automatically detects minimum and maximum values for X and Y axes"""
    x_min = None
    x_max = None
    y_min = None
    y_max = None
    
    for dataset in datasets:
        if dataset['active'] and dataset['data'].strip():
            df = parse_data(dataset['data'], dataset['name'])
            if not df.empty:
                if x_min is None or df['x'].min() < x_min:
                    x_min = df['x'].min()
                if x_max is None or df['x'].max() > x_max:
                    x_max = df['x'].max()
                if y_min is None or df['y'].min() < y_min:
                    y_min = df['y'].min()
                if y_max is None or df['y'].max() > y_max:
                    y_max = df['y'].max()
    
    # Add small margin at edges (10%)
    if x_min is not None and x_max is not None and x_min != x_max:
        x_range = x_max - x_min
        x_min_auto = x_min - 0.1 * x_range
        x_max_auto = x_max + 0.1 * x_range
    elif x_min is not None and x_max is not None:
        x_min_auto = x_min - 0.1
        x_max_auto = x_max + 0.1
    else:
        x_min_auto = 0
        x_max_auto = 1
    
    if y_min is not None and y_max is not None and y_min != y_max:
        y_range = y_max - y_min
        y_min_auto = y_min - 0.1 * y_range
        y_max_auto = y_max + 0.1 * y_range
    elif y_min is not None and y_max is not None:
        y_min_auto = y_min - 0.1
        y_max_auto = y_max + 0.1
    else:
        y_min_auto = 0
        y_max_auto = 1
    
    # Calculate step (about 10 divisions)
    x_step_auto = max((x_max_auto - x_min_auto) / 10, 0.1)
    y_step_auto = max((y_max_auto - y_min_auto) / 10, 0.1)
    
    return {
        'x_min': round(x_min_auto, 3),
        'x_max': round(x_max_auto, 3),
        'x_step': round(x_step_auto, 3),
        'y_min': round(y_min_auto, 3),
        'y_max': round(y_max_auto, 3),
        'y_step': round(y_step_auto, 3)
    }

# Function to create marginal plots with different types
def create_marginal_plot(datasets, x_label, y_label, plot_title, legend_title,
                        x_min_val, x_max_val, y_min_val, y_max_val,
                        normalize_density, graph_settings, 
                        marginal_type='density',  # 'density', 'histogram', 'boxplot', 'violin'
                        marker_size=50, legend_fontsize=10,
                        axis_label_fontsize=12, tick_fontsize=10,
                        show_legend=True):
    """
    Creates a scatter plot with marginal distributions of specified type
    """
    # Create figure with gridspec
    fig = plt.figure(figsize=(12, 10))
    gs = plt.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Create axes
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    
    # Apply custom axes settings
    set_custom_axes(ax_top, 
                   border_color=graph_settings['border_color'], 
                   border_width=graph_settings['border_width'],
                   grid_color=graph_settings['grid_color'],
                   grid_visible=graph_settings['grid_visible'],
                   background_color=graph_settings['background_color'],
                   tick_fontsize=tick_fontsize,
                   label_fontsize=axis_label_fontsize)
    
    set_custom_axes(ax_main, 
                   border_color=graph_settings['border_color'], 
                   border_width=graph_settings['border_width'],
                   grid_color=graph_settings['grid_color'],
                   grid_visible=graph_settings['grid_visible'],
                   background_color=graph_settings['background_color'],
                   tick_fontsize=tick_fontsize,
                   label_fontsize=axis_label_fontsize)
    
    set_custom_axes(ax_right, 
                   border_color=graph_settings['border_color'], 
                   border_width=graph_settings['border_width'],
                   grid_color=graph_settings['grid_color'],
                   grid_visible=graph_settings['grid_visible'],
                   background_color=graph_settings['background_color'],
                   tick_fontsize=tick_fontsize,
                   label_fontsize=axis_label_fontsize)
    
    # Hide x-axis labels for top plot and y-axis labels for right plot
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    
    # Collect all data for plotting
    all_data = []
    for i, dataset in enumerate(datasets):
        if dataset['active']:
            df = parse_data(dataset['data'], dataset['name'])
            if not df.empty:
                # Add scatter points to main plot
                ax_main.scatter(
                    df['x'], df['y'],
                    color=dataset['color'],
                    label=dataset['name'],
                    marker=matplotlib_markers[dataset['marker']],
                    s=marker_size,
                    alpha=0.7
                )
                all_data.append((df, dataset))
    
    # Set main plot labels
    ax_main.set_xlabel(format_axis_label(x_label), fontsize=axis_label_fontsize)
    ax_main.set_ylabel(format_axis_label(y_label), fontsize=axis_label_fontsize)
    
    # Set axis limits
    ax_main.set_xlim(x_min_val, x_max_val)
    ax_main.set_ylim(y_min_val, y_max_val)
    ax_top.set_xlim(x_min_val, x_max_val)
    ax_right.set_ylim(y_min_val, y_max_val)
    
    # Add legend if requested
    if show_legend and len(datasets) > 0:
        ax_main.legend(title=legend_title, fontsize=legend_fontsize, 
                      title_fontsize=legend_fontsize, loc='upper right')
    
    # Create marginal plots based on type
    if marginal_type == 'density':
        # Density plots
        max_x_density = 0
        max_y_density = 0
        
        for df, dataset in all_data:
            color = dataset['color']
            
            # X density (top plot)
            if len(df) > 1:
                x_vals, density = estimate_density(df['x'].values, normalize=normalize_density)
                if x_vals is not None and density is not None:
                    if not normalize_density and density.max() > max_x_density:
                        max_x_density = density.max()
                    ax_top.fill_between(x_vals, 0, density, color=color, alpha=0.3)
                    ax_top.plot(x_vals, density, color=color, linewidth=1.5)
            
            # Y density (right plot)
            if len(df) > 1:
                y_vals, density = estimate_density(df['y'].values, normalize=normalize_density)
                if y_vals is not None and density is not None:
                    if not normalize_density and density.max() > max_y_density:
                        max_y_density = density.max()
                    ax_right.fill_betweenx(y_vals, 0, density, color=color, alpha=0.3)
                    ax_right.plot(density, y_vals, color=color, linewidth=1.5)
        
        # Set density plot limits
        if normalize_density:
            ax_top.set_ylim(0, 1.1)
            ax_right.set_xlim(0, 1.1)
        else:
            ax_top.set_ylim(0, max_x_density * 1.1 if max_x_density > 0 else 1.1)
            ax_right.set_xlim(0, max_y_density * 1.1 if max_y_density > 0 else 1.1)
        
        ax_top.set_ylabel('Density' if not normalize_density else 'Norm. Density', 
                         fontsize=axis_label_fontsize)
        ax_right.set_xlabel('Density' if not normalize_density else 'Norm. Density', 
                           fontsize=axis_label_fontsize)
    
    elif marginal_type == 'histogram':
        # Histogram plots
        bin_width_x = (x_max_val - x_min_val) / 20
        bin_width_y = (y_max_val - y_min_val) / 20
        
        for df, dataset in all_data:
            color = dataset['color']
            
            # X histogram (top plot)
            if len(df) > 0:
                ax_top.hist(df['x'], bins=np.arange(x_min_val, x_max_val + bin_width_x, bin_width_x),
                           color=color, alpha=0.5, density=True, edgecolor='black', linewidth=0.5)
            
            # Y histogram (right plot)
            if len(df) > 0:
                ax_right.hist(df['y'], bins=np.arange(y_min_val, y_max_val + bin_width_y, bin_width_y),
                            color=color, alpha=0.5, density=True, edgecolor='black', linewidth=0.5,
                            orientation='horizontal')
        
        ax_top.set_ylabel('Density', fontsize=axis_label_fontsize)
        ax_right.set_xlabel('Density', fontsize=axis_label_fontsize)
        
        # Adjust limits
        y_top_lim = ax_top.get_ylim()[1]
        ax_top.set_ylim(0, y_top_lim * 1.1)
        x_right_lim = ax_right.get_xlim()[1]
        ax_right.set_xlim(0, x_right_lim * 1.1)
    
    elif marginal_type == 'boxplot':
        # Boxplot plots
        x_data = []
        y_data = []
        colors = []
        labels = []
        
        for df, dataset in all_data:
            if len(df) > 0:
                x_data.append(df['x'].values)
                y_data.append(df['y'].values)
                colors.append(dataset['color'])
                labels.append(dataset['name'])
        
        if x_data:
            # X boxplot (top plot) - horizontal
            bp_x = ax_top.boxplot(x_data, vert=False, patch_artist=True, 
                                 labels=labels if len(labels) <= 5 else None)
            # Color the boxes
            for patch, color in zip(bp_x['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            
            # Customize boxplot appearance
            for element in ['whiskers', 'caps', 'medians']:
                for line in bp_x[element]:
                    line.set_color('black')
                    line.set_linewidth(1)
        
        if y_data:
            # Y boxplot (right plot) - vertical
            bp_y = ax_right.boxplot(y_data, vert=True, patch_artist=True,
                                   labels=labels if len(labels) <= 5 else None)
            # Color the boxes
            for patch, color in zip(bp_y['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            
            # Customize boxplot appearance
            for element in ['whiskers', 'caps', 'medians']:
                for line in bp_y[element]:
                    line.set_color('black')
                    line.set_linewidth(1)
        
        ax_top.set_ylabel('Value', fontsize=axis_label_fontsize)
        ax_right.set_xlabel('Value', fontsize=axis_label_fontsize)
        
        # Hide labels if too many datasets
        if len(labels) > 5:
            ax_top.set_yticklabels([])
            ax_right.set_xticklabels([])
    
    elif marginal_type == 'violin':
        # Violin plots
        x_data = []
        y_data = []
        colors = []
        labels = []
        
        for df, dataset in all_data:
            if len(df) > 0:
                x_data.append(df['x'].values)
                y_data.append(df['y'].values)
                colors.append(dataset['color'])
                labels.append(dataset['name'])
        
        if x_data:
            # X violin plot (top plot) - horizontal
            vp_x = ax_top.violinplot(x_data, vert=False, showmeans=True, showmedians=True)
            
            # Color the violins
            for i, (body, color) in enumerate(zip(vp_x['bodies'], colors)):
                body.set_facecolor(color)
                body.set_alpha(0.5)
                body.set_edgecolor('black')
            
            # Customize appearance
            vp_x['cmeans'].set_color('red')
            vp_x['cmedians'].set_color('blue')
            vp_x['cmins'].set_color('black')
            vp_x['cmaxes'].set_color('black')
            vp_x['cbars'].set_color('black')
        
        if y_data:
            # Y violin plot (right plot) - vertical
            vp_y = ax_right.violinplot(y_data, vert=True, showmeans=True, showmedians=True)
            
            # Color the violins
            for i, (body, color) in enumerate(zip(vp_y['bodies'], colors)):
                body.set_facecolor(color)
                body.set_alpha(0.5)
                body.set_edgecolor('black')
            
            # Customize appearance
            vp_y['cmeans'].set_color('red')
            vp_y['cmedians'].set_color('blue')
            vp_y['cmins'].set_color('black')
            vp_y['cmaxes'].set_color('black')
            vp_y['cbars'].set_color('black')
        
        ax_top.set_ylabel('Value', fontsize=axis_label_fontsize)
        ax_right.set_xlabel('Value', fontsize=axis_label_fontsize)
    
    # Set title
    if plot_title:
        fig.suptitle(f"{plot_title} - {marginal_type.capitalize()} Marginals", 
                    fontsize=axis_label_fontsize * 1.2, fontweight='bold')
    else:
        fig.suptitle(f'Scatter Plot with {marginal_type.capitalize()} Marginals', 
                    fontsize=axis_label_fontsize * 1.2, fontweight='bold')
    
    return fig

# Function to export all data with settings
def export_all_data_with_settings(datasets, x_label, y_label, plot_title, legend_title,
                                 x_manual, y_manual, x_min_val, x_max_val, x_step_val, 
                                 y_min_val, y_max_val, y_step_val,
                                 marker_size, legend_fontsize,
                                 axis_label_fontsize, tick_fontsize,
                                 graph_settings, normalize_density):
    """Creates CSV file with data and all settings"""
    
    # Create export structure
    export_dict = {
        'metadata': {
            'version': '1.4',
            'x_axis_label': x_label,
            'y_axis_label': y_label,
            'plot_title': plot_title,
            'legend_title': legend_title,
            'num_datasets': len(datasets),
            'export_timestamp': pd.Timestamp.now().isoformat(),
            'marker_size': marker_size,
            'legend_fontsize': legend_fontsize,
            'axis_label_fontsize': axis_label_fontsize,
            'tick_fontsize': tick_fontsize,
            'normalize_density': normalize_density
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
        'graph_settings': graph_settings,
        'dataset_settings': [],
        'data': []
    }
    
    # Save each dataset settings
    for i, dataset in enumerate(datasets):
        dataset_settings = {
            'dataset_index': i,
            'name': dataset['name'],
            'color': dataset['color'],
            'marker': dataset['marker'],
            'active': dataset['active']
        }
        export_dict['dataset_settings'].append(dataset_settings)
        
        # Save dataset data
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
    
    # Create CSV-compatible structure
    lines = []
    
    # 1. Metadata
    lines.append("# META DATA SECTION")
    lines.append(f"x_axis_label: {x_label}")
    lines.append(f"y_axis_label: {y_label}")
    lines.append(f"plot_title: {plot_title}")
    lines.append(f"legend_title: {legend_title}")
    lines.append(f"num_datasets: {len(datasets)}")
    lines.append(f"export_timestamp: {export_dict['metadata']['export_timestamp']}")
    lines.append(f"marker_size: {marker_size}")
    lines.append(f"legend_fontsize: {legend_fontsize}")
    lines.append(f"axis_label_fontsize: {axis_label_fontsize}")
    lines.append(f"tick_fontsize: {tick_fontsize}")
    lines.append(f"normalize_density: {normalize_density}")
    lines.append("")
    
    # 2. Axis settings
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
    
    # 3. Graph settings
    lines.append("# GRAPH SETTINGS SECTION")
    lines.append("setting,value")
    for key, value in graph_settings.items():
        lines.append(f"{key},{value}")
    lines.append("")
    
    # 4. Dataset settings
    lines.append("# DATASET SETTINGS SECTION")
    lines.append("index,name,color,marker,active")
    for settings in export_dict['dataset_settings']:
        lines.append(f"{settings['dataset_index']},{settings['name']},{settings['color']},{settings['marker']},{settings['active']}")
    lines.append("")
    
    # 5. Data
    lines.append("# DATA POINTS SECTION")
    lines.append("dataset_index,dataset_name,x,y")
    for data_point in export_dict['data']:
        lines.append(f"{data_point['dataset_index']},{data_point['dataset_name']},{data_point['x']},{data_point['y']}")
    
    return "\n".join(lines)

# Function to import data with settings
# Function to import data with settings - UPDATED FOR MULTIPLE SEPARATORS
def import_data_with_settings(file_content):
    """Imports data and settings from CSV file with multiple separator support"""
    
    lines = file_content.strip().split('\n')
    
    # Initialize variables
    x_axis_label = "X Axis"
    y_axis_label = "Y Axis"
    plot_title = ""
    legend_title = "Datasets"
    x_manual = False
    y_manual = False
    x_min = None
    x_max = None
    x_step = None
    y_min = None
    y_max = None
    y_step = None
    marker_size = 50
    legend_fontsize = 10
    axis_label_fontsize = 12
    tick_fontsize = 10
    normalize_density = True
    graph_settings = {
        'background_color': '#FFFFFF',
        'grid_color': '#CCCCCC',
        'grid_visible': 'True',
        'border_color': '#000000',
        'border_width': '2.5'
    }
    dataset_settings = []
    data_points = []
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Determine section
        if line.startswith("# META DATA SECTION"):
            current_section = "metadata"
            continue
        elif line.startswith("# AXIS SETTINGS SECTION"):
            current_section = "axis_settings"
            continue
        elif line.startswith("# GRAPH SETTINGS SECTION"):
            current_section = "graph_settings"
            continue
        elif line.startswith("# DATASET SETTINGS SECTION"):
            current_section = "settings"
            continue
        elif line.startswith("# DATA POINTS SECTION"):
            current_section = "data"
            continue
        elif line.startswith("#"):
            continue
        
        # Process metadata
        if current_section == "metadata":
            if line.startswith("x_axis_label:"):
                x_axis_label = line.split(":", 1)[1].strip()
            elif line.startswith("y_axis_label:"):
                y_axis_label = line.split(":", 1)[1].strip()
            elif line.startswith("plot_title:"):
                plot_title = line.split(":", 1)[1].strip()
            elif line.startswith("legend_title:"):
                legend_title = line.split(":", 1)[1].strip()
            elif line.startswith("marker_size:"):
                try:
                    marker_size = int(line.split(":", 1)[1].strip())
                except:
                    marker_size = 50
            elif line.startswith("legend_fontsize:"):
                try:
                    legend_fontsize = float(line.split(":", 1)[1].strip())
                except:
                    legend_fontsize = 10
            elif line.startswith("axis_label_fontsize:"):
                try:
                    axis_label_fontsize = float(line.split(":", 1)[1].strip())
                except:
                    axis_label_fontsize = 12
            elif line.startswith("tick_fontsize:"):
                try:
                    tick_fontsize = float(line.split(":", 1)[1].strip())
                except:
                    tick_fontsize = 10
            elif line.startswith("normalize_density:"):
                try:
                    normalize_density = line.split(":", 1)[1].strip().lower() == 'true'
                except:
                    normalize_density = True
        
        # Process axis settings
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
        
        # Process graph settings
        elif current_section == "graph_settings":
            if line.startswith("setting,value"):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                setting_name = parts[0].strip()
                setting_value = parts[1].strip()
                graph_settings[setting_name] = setting_value
        
        # Process dataset settings
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
                    dataset_settings.append(dataset_setting)
                except:
                    continue
        
        # Process data - UPDATED FOR MULTIPLE SEPARATORS
        elif current_section == "data":
            if line.startswith("dataset_index,dataset_name,x,y"):
                continue
            
            # Try different separators for data lines
            separators = [',', ';', '\t']
            parts = None
            
            for sep in separators:
                if sep in line:
                    parts = line.split(sep)
                    if len(parts) >= 4:
                        break
            
            if not parts and ' ' in line:
                # Try space separator
                parts = line.split(' ')
                if len(parts) < 4:
                    parts = None
            
            if parts and len(parts) >= 4:
                try:
                    data_point = {
                        'dataset_index': int(parts[0].strip()),
                        'dataset_name': parts[1].strip(),
                        'x': float(parts[2].strip().replace(',', '.')),
                        'y': float(parts[3].strip().replace(',', '.'))
                    }
                    data_points.append(data_point)
                except:
                    # Try alternative parsing
                    try:
                        import re
                        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
                        if len(numbers) >= 4:
                            data_point = {
                                'dataset_index': int(float(numbers[0])),
                                'dataset_name': ''.join(filter(str.isalpha, line)),
                                'x': float(numbers[2]),
                                'y': float(numbers[3])
                            }
                            data_points.append(data_point)
                    except:
                        continue
    
    # Reconstruct datasets
    datasets = []
    
    # Group data by dataset
    data_by_dataset = {}
    for dp in data_points:
        idx = dp['dataset_index']
        if idx not in data_by_dataset:
            data_by_dataset[idx] = []
        # Store in tab-separated format for consistency
        data_by_dataset[idx].append(f"{dp['x']}\t{dp['y']}")
    
    # Create datasets structure
    for setting in dataset_settings:
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
    
    return datasets, x_axis_label, y_axis_label, plot_title, legend_title, axis_settings, marker_size, legend_fontsize, axis_label_fontsize, tick_fontsize, normalize_density, graph_settings

# Main title
st.title("📊 Data Visualization with Marginal Distributions")
st.markdown("---")

# Initialize session state - NO DEFAULT DATA
if 'datasets' not in st.session_state:
    st.session_state.datasets = []  # Empty list instead of default data

if 'x_axis_label' not in st.session_state:
    st.session_state.x_axis_label = 'X Axis'

if 'y_axis_label' not in st.session_state:
    st.session_state.y_axis_label = 'Y Axis'

if 'plot_title' not in st.session_state:
    st.session_state.plot_title = ''

if 'legend_title' not in st.session_state:
    st.session_state.legend_title = 'Datasets'

if 'normalize_density' not in st.session_state:
    st.session_state.normalize_density = True

# Initialize marginal type state
if 'marginal_type' not in st.session_state:
    st.session_state.marginal_type = 'density'

# Initialize axis settings state
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

# Initialize graph settings state
if 'graph_settings' not in st.session_state:
    st.session_state.graph_settings = {
        'background_color': '#FFFFFF',
        'grid_color': '#CCCCCC',
        'grid_visible': True,
        'border_color': '#000000',
        'border_width': 2.5
    }

# Initialize imported data state
if 'imported_file_content' not in st.session_state:
    st.session_state.imported_file_content = None

if 'imported_datasets' not in st.session_state:
    st.session_state.imported_datasets = None

if 'imported_x_label' not in st.session_state:
    st.session_state.imported_x_label = None

if 'imported_y_label' not in st.session_state:
    st.session_state.imported_y_label = None

if 'imported_plot_title' not in st.session_state:
    st.session_state.imported_plot_title = None

if 'imported_legend_title' not in st.session_state:
    st.session_state.imported_legend_title = None

if 'imported_axis_settings' not in st.session_state:
    st.session_state.imported_axis_settings = None

if 'imported_marker_size' not in st.session_state:
    st.session_state.imported_marker_size = 50

if 'imported_legend_fontsize' not in st.session_state:
    st.session_state.imported_legend_fontsize = 10

if 'imported_axis_label_fontsize' not in st.session_state:
    st.session_state.imported_axis_label_fontsize = 12

if 'imported_tick_fontsize' not in st.session_state:
    st.session_state.imported_tick_fontsize = 10

if 'imported_normalize_density' not in st.session_state:
    st.session_state.imported_normalize_density = True

if 'imported_graph_settings' not in st.session_state:
    st.session_state.imported_graph_settings = None

# Flag for applying imported data
if 'apply_imported_data' not in st.session_state:
    st.session_state.apply_imported_data = False

# Initialize marker size and legend state
if 'marker_size' not in st.session_state:
    st.session_state.marker_size = 50

if 'legend_fontsize' not in st.session_state:
    st.session_state.legend_fontsize = 10

if 'axis_label_fontsize' not in st.session_state:
    st.session_state.axis_label_fontsize = 12

if 'tick_fontsize' not in st.session_state:
    st.session_state.tick_fontsize = 10

# Available markers for matplotlib and Plotly
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

# Default colors
default_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33', '#A65628', '#F781BF', '#999999']

# Function to reset all settings
def reset_all_settings():
    """Resets all settings to default values"""
    st.session_state.datasets = []
    st.session_state.x_axis_label = 'X Axis'
    st.session_state.y_axis_label = 'Y Axis'
    st.session_state.plot_title = ''
    st.session_state.legend_title = 'Datasets'
    st.session_state.marginal_type = 'density'
    st.session_state.x_manual = False
    st.session_state.y_manual = False
    st.session_state.x_min = None
    st.session_state.x_max = None
    st.session_state.x_step = None
    st.session_state.y_min = None
    st.session_state.y_max = None
    st.session_state.y_step = None
    st.session_state.marker_size = 50
    st.session_state.legend_fontsize = 10
    st.session_state.axis_label_fontsize = 12
    st.session_state.tick_fontsize = 10
    st.session_state.normalize_density = True
    st.session_state.graph_settings = {
        'background_color': '#FFFFFF',
        'grid_color': '#CCCCCC',
        'grid_visible': True,
        'border_color': '#000000',
        'border_width': 2.5
    }

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Clear All button at the top
    if st.button("🗑️ Clear All Settings", type="secondary"):
        reset_all_settings()
        st.rerun()
    
    # 1. Dataset Management
    st.subheader("Dataset Management")
    
    if st.button("➕ Add New Dataset", key="add_dataset_button"):
        idx = len(st.session_state.datasets)
        new_dataset = {
            'name': f'Dataset {idx + 1}',
            'data': '',
            'color': default_colors[idx % len(default_colors)],
            'marker': 'circle',
            'active': True
        }
        st.session_state.datasets.append(new_dataset)
        st.rerun()
    
    if st.button("➖ Remove Last Dataset", key="remove_dataset_button") and len(st.session_state.datasets) > 0:
        st.session_state.datasets.pop()
        st.rerun()
    
    # 2. Title and Label Settings
    st.subheader("Title and Label Settings")
    
    st.session_state.plot_title = st.text_input(
        "Plot Title",
        value=st.session_state.plot_title,
        key="plot_title_input"
    )
    
    st.session_state.x_axis_label = st.text_input(
        "X Axis Label",
        value=st.session_state.x_axis_label,
        key="x_axis_label_input"
    )
    
    st.session_state.y_axis_label = st.text_input(
        "Y Axis Label",
        value=st.session_state.y_axis_label,
        key="y_axis_label_input"
    )
    
    st.session_state.legend_title = st.text_input(
        "Legend Title",
        value=st.session_state.legend_title,
        key="legend_title_input"
    )
    
    # 3. Marginal Distributions Settings
    st.subheader("Marginal Distributions")
    
    # Type of marginal plot
    marginal_type = st.selectbox(
        "Type of Marginal Plot",
        options=['density', 'histogram', 'boxplot', 'violin'],
        index=['density', 'histogram', 'boxplot', 'violin'].index(st.session_state.marginal_type),
        key="marginal_type_select",
        help="Select the type of marginal distribution to display"
    )
    st.session_state.marginal_type = marginal_type
    
    # Normalize option (only for density plots)
    if marginal_type == 'density':
        st.session_state.normalize_density = st.checkbox(
            "Normalize Marginal Distributions",
            value=st.session_state.normalize_density,
            help="When checked, marginal distributions are normalized to [0,1]. When unchecked, shows actual density values.",
            key="normalize_density_checkbox"
        )
    else:
        # For non-density plots, normalization doesn't apply
        st.info(f"Normalization only applies to density plots. For {marginal_type} plots, actual distributions are shown.")
    
    # 4. Axis Boundaries Management
    st.subheader("Axis Boundaries Management")
    
    col1, col2 = st.columns(2)
    with col1:
        x_manual = st.checkbox("Configure X Axis", 
                              value=st.session_state.x_manual,
                              key="x_manual_checkbox")
        st.session_state.x_manual = x_manual
    with col2:
        y_manual = st.checkbox("Configure Y Axis", 
                              value=st.session_state.y_manual,
                              key="y_manual_checkbox")
        st.session_state.y_manual = y_manual
    
    # Automatic axis limits detection
    auto_limits = auto_detect_axis_limits(st.session_state.datasets)
    
    # Update session_state values if not manually set
    if st.session_state.x_min is None:
        st.session_state.x_min = auto_limits['x_min']
    if st.session_state.x_max is None:
        st.session_state.x_max = auto_limits['x_max']
    if st.session_state.x_step is None:
        st.session_state.x_step = auto_limits['x_step']
    if st.session_state.y_min is None:
        st.session_state.y_min = auto_limits['y_min']
    if st.session_state.y_max is None:
        st.session_state.y_max = auto_limits['y_max']
    if st.session_state.y_step is None:
        st.session_state.y_step = auto_limits['y_step']
    
    if x_manual:
        col1, col2, col3 = st.columns(3)
        with col1:
            # Show automatically detected value as default
            x_min = st.number_input("X min", 
                                   value=float(st.session_state.x_min) if st.session_state.x_min is not None else float(auto_limits['x_min']), 
                                   step=0.1,
                                   key="x_min_input")
            st.session_state.x_min = x_min
        with col2:
            x_max = st.number_input("X max", 
                                   value=float(st.session_state.x_max) if st.session_state.x_max is not None else float(auto_limits['x_max']), 
                                   step=0.1,
                                   key="x_max_input")
            st.session_state.x_max = x_max
        with col3:
            x_step = st.number_input("X step", 
                                    value=float(st.session_state.x_step) if st.session_state.x_step is not None else float(auto_limits['x_step']), 
                                    step=0.1, 
                                    min_value=0.01,
                                    key="x_step_input")
            st.session_state.x_step = x_step
    else:
        # Reset values when manual configuration is disabled
        st.session_state.x_min = None
        st.session_state.x_max = None
        st.session_state.x_step = None
    
    if y_manual:
        col1, col2, col3 = st.columns(3)
        with col1:
            y_min = st.number_input("Y min", 
                                   value=float(st.session_state.y_min) if st.session_state.y_min is not None else float(auto_limits['y_min']), 
                                   step=0.1,
                                   key="y_min_input")
            st.session_state.y_min = y_min
        with col2:
            y_max = st.number_input("Y max", 
                                   value=float(st.session_state.y_max) if st.session_state.y_max is not None else float(auto_limits['y_max']), 
                                   step=0.1,
                                   key="y_max_input")
            st.session_state.y_max = y_max
        with col3:
            y_step = st.number_input("Y step", 
                                    value=float(st.session_state.y_step) if st.session_state.y_step is not None else float(auto_limits['y_step']), 
                                    step=0.1, 
                                    min_value=0.01,
                                    key="y_step_input")
            st.session_state.y_step = y_step
    else:
        # Reset values when manual configuration is disabled
        st.session_state.y_min = None
        st.session_state.y_max = None
        st.session_state.y_step = None
    
    # Button to reset to automatic values
    if st.button("🔄 Reset to Automatic Values"):
        auto_limits = auto_detect_axis_limits(st.session_state.datasets)
        st.session_state.x_min = auto_limits['x_min']
        st.session_state.x_max = auto_limits['x_max']
        st.session_state.x_step = auto_limits['x_step']
        st.session_state.y_min = auto_limits['y_min']
        st.session_state.y_max = auto_limits['y_max']
        st.session_state.y_step = auto_limits['y_step']
        st.rerun()
    
    # 5. Display Settings
    st.subheader("Display Settings")
    
    # Marker size settings
    marker_size = st.slider(
        "Marker Size",
        min_value=10,
        max_value=200,
        value=st.session_state.marker_size,
        step=5,
        key="marker_size_slider",
        help="Size of markers on plots"
    )
    st.session_state.marker_size = marker_size
    
    # Font size settings
    st.markdown("**Font Sizes**")
    
    legend_fontsize = st.slider(
        "Legend Font Size",
        min_value=6,
        max_value=24,
        value=st.session_state.legend_fontsize,
        step=1,
        key="legend_fontsize_slider",
        help="Font size in plot legends"
    )
    st.session_state.legend_fontsize = legend_fontsize
    
    axis_label_fontsize = st.slider(
        "Axis Label Font Size",
        min_value=8,
        max_value=24,
        value=st.session_state.axis_label_fontsize,
        step=1,
        key="axis_label_fontsize_slider",
        help="Font size for axis labels"
    )
    st.session_state.axis_label_fontsize = axis_label_fontsize
    
    tick_fontsize = st.slider(
        "Tick Label Font Size",
        min_value=6,
        max_value=20,
        value=st.session_state.tick_fontsize,
        step=1,
        key="tick_fontsize_slider",
        help="Font size for tick labels (numbers on axes)"
    )
    st.session_state.tick_fontsize = tick_fontsize
    
    # 6. Graph Appearance Settings
    st.subheader("Graph Appearance")
    
    # Background color
    background_color = st.color_picker(
        "Graph Background Color",
        value=st.session_state.graph_settings['background_color'],
        key="background_color_picker"
    )
    st.session_state.graph_settings['background_color'] = background_color
    
    # Grid settings
    grid_visible = st.checkbox(
        "Show Grid",
        value=st.session_state.graph_settings['grid_visible'],
        key="grid_visible_checkbox"
    )
    st.session_state.graph_settings['grid_visible'] = grid_visible
    
    if grid_visible:
        grid_color = st.color_picker(
            "Grid Color",
            value=st.session_state.graph_settings['grid_color'],
            key="grid_color_picker"
        )
        st.session_state.graph_settings['grid_color'] = grid_color
    
    # Border settings
    border_color = st.color_picker(
        "Border Color",
        value=st.session_state.graph_settings['border_color'],
        key="border_color_picker"
    )
    st.session_state.graph_settings['border_color'] = border_color
    
    border_width = st.slider(
        "Border Width",
        min_value=0.5,
        max_value=10.0,
        value=float(st.session_state.graph_settings['border_width']),
        step=0.5,
        key="border_width_slider"
    )
    st.session_state.graph_settings['border_width'] = border_width
    
    # 7. Import/Export
    st.subheader("Import/Export")
    
    uploaded_file = st.file_uploader(
        "Load Data with Settings",
        type=['csv', 'txt'],
        help="Load a file previously exported from this application"
    )
    
    if uploaded_file is not None:
        try:
            file_content = uploaded_file.getvalue().decode('utf-8')
            imported_datasets, imported_x_label, imported_y_label, imported_plot_title, imported_legend_title, imported_axis_settings, \
            imported_marker_size, imported_legend_fontsize, imported_axis_label_fontsize, imported_tick_fontsize, imported_normalize_density, \
            imported_graph_settings = import_data_with_settings(file_content)
            
            if imported_datasets:
                st.session_state.imported_file_content = file_content
                st.session_state.imported_datasets = imported_datasets
                st.session_state.imported_x_label = imported_x_label
                st.session_state.imported_y_label = imported_y_label
                st.session_state.imported_plot_title = imported_plot_title
                st.session_state.imported_legend_title = imported_legend_title
                st.session_state.imported_axis_settings = imported_axis_settings
                st.session_state.imported_marker_size = imported_marker_size
                st.session_state.imported_legend_fontsize = imported_legend_fontsize
                st.session_state.imported_axis_label_fontsize = imported_axis_label_fontsize
                st.session_state.imported_tick_fontsize = imported_tick_fontsize
                st.session_state.imported_normalize_density = imported_normalize_density
                st.session_state.imported_graph_settings = imported_graph_settings
                
                st.success(f"File loaded! Found {len(imported_datasets)} datasets.")
                st.info("Click 'Apply Loaded Data' button below to use these settings.")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Button to apply loaded data
    if st.session_state.imported_datasets is not None:
        if st.button("✅ Apply Loaded Data", type="primary"):
            # Set flag to apply data
            st.session_state.apply_imported_data = True
            st.rerun()

# Apply imported data (if flag is set)
if st.session_state.apply_imported_data and st.session_state.imported_datasets is not None:
    # COMPLETELY replace datasets with imported ones
    st.session_state.datasets = st.session_state.imported_datasets.copy()
    
    st.session_state.x_axis_label = st.session_state.imported_x_label
    st.session_state.y_axis_label = st.session_state.imported_y_label
    st.session_state.plot_title = st.session_state.imported_plot_title
    st.session_state.legend_title = st.session_state.imported_legend_title
    
    # Apply axis settings
    if st.session_state.imported_axis_settings:
        st.session_state.x_manual = st.session_state.imported_axis_settings['x_manual']
        st.session_state.y_manual = st.session_state.imported_axis_settings['y_manual']
        st.session_state.x_min = st.session_state.imported_axis_settings['x_min']
        st.session_state.x_max = st.session_state.imported_axis_settings['x_max']
        st.session_state.x_step = st.session_state.imported_axis_settings['x_step']
        st.session_state.y_min = st.session_state.imported_axis_settings['y_min']
        st.session_state.y_max = st.session_state.imported_axis_settings['y_max']
        st.session_state.y_step = st.session_state.imported_axis_settings['y_step']
    
    # Apply display settings
    st.session_state.marker_size = st.session_state.imported_marker_size
    st.session_state.legend_fontsize = st.session_state.imported_legend_fontsize
    st.session_state.axis_label_fontsize = st.session_state.imported_axis_label_fontsize
    st.session_state.tick_fontsize = st.session_state.imported_tick_fontsize
    st.session_state.normalize_density = st.session_state.imported_normalize_density
    
    # Apply graph settings
    if st.session_state.imported_graph_settings:
        st.session_state.graph_settings = st.session_state.imported_graph_settings.copy()
        # Ensure boolean conversion
        st.session_state.graph_settings['grid_visible'] = st.session_state.graph_settings['grid_visible'].lower() == 'true' if isinstance(st.session_state.graph_settings['grid_visible'], str) else st.session_state.graph_settings['grid_visible']
        st.session_state.graph_settings['border_width'] = float(st.session_state.graph_settings['border_width'])
    
    # Reset import state
    st.session_state.imported_file_content = None
    st.session_state.imported_datasets = None
    st.session_state.imported_x_label = None
    st.session_state.imported_y_label = None
    st.session_state.imported_plot_title = None
    st.session_state.imported_legend_title = None
    st.session_state.imported_axis_settings = None
    st.session_state.imported_marker_size = 50
    st.session_state.imported_legend_fontsize = 10
    st.session_state.imported_axis_label_fontsize = 12
    st.session_state.imported_tick_fontsize = 10
    st.session_state.imported_normalize_density = True
    st.session_state.imported_graph_settings = None
    st.session_state.apply_imported_data = False
    
    st.success("Data successfully applied!")
    st.rerun()

# Main area
tab1, tab2, tab3 = st.tabs(["📁 Data", "📊 Plots", "📈 Statistics"])

with tab1:
    st.header("Dataset Configuration")
    
    # Instructions with examples
    st.markdown("""
    **Enter data in any of these formats:**
    - **Tab-separated:** `X_value<tab>Y_value` (Example: `0.1\t-5.5`)
    - **Space-separated:** `X_value Y_value` (Example: `0.1 -5.5`)
    - **Comma-separated:** `X_value,Y_value` (Example: `0.1,-5.5`)
    - **Semicolon-separated:** `X_value;Y_value` (Example: `0.1;-5.5`)
    
    **Notes:**
    - Decimal separator can be dot (.) or comma (,)
    - Scientific notation is supported (e.g., `1.23e-4`)
    - Comment lines starting with `#` are ignored
    - Empty lines are ignored
    """)
    
    # Format selector for new data entry
    st.subheader("Data Entry Format")
    entry_format = st.radio(
        "Select format for new data entry:",
        options=["Tab (\\t)", "Space", "Comma (,)", "Semicolon (;)"],
        horizontal=True,
        index=0,
        key="entry_format_radio"
    )
    
    # Map format to separator
    format_map = {
        "Tab (\\t)": "\t",
        "Space": " ",
        "Comma (,)": ",",
        "Semicolon (;)": ";"
    }
    current_separator = format_map[entry_format]
    
    # Show current automatic boundaries
    auto_limits_tab1 = auto_detect_axis_limits(st.session_state.datasets)
    st.info(f"Auto-detected boundaries: X=[{auto_limits_tab1['x_min']:.3f}, {auto_limits_tab1['x_max']:.3f}], Y=[{auto_limits_tab1['y_min']:.3f}, {auto_limits_tab1['y_max']:.3f}]")
    
    if len(st.session_state.datasets) == 0:
        st.info("No datasets. Click '➕ Add New Dataset' button in the sidebar.")
    else:
        # Display and edit datasets
        all_data_frames = []
        
        for i, dataset in enumerate(st.session_state.datasets):
            with st.expander(f"Dataset {i+1}: {dataset['name']}", expanded=True):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    new_name = st.text_input(
                        f"Dataset Name {i+1}",
                        value=dataset['name'],
                        key=f"name_{i}"
                    )
                    st.session_state.datasets[i]['name'] = new_name
                    
                    # Format data for display based on selected format
                    data_for_display = dataset['data']
                    
                    data_text = st.text_area(
                        f"Data (X{current_separator}Y)",
                        value=data_for_display,
                        height=200,
                        key=f"data_{i}",
                        help=f"Enter data with '{current_separator}' as separator"
                    )
                    st.session_state.datasets[i]['data'] = data_text
                
                with col2:
                    color = st.color_picker(
                        "Color",
                        value=dataset['color'],
                        key=f"color_{i}"
                    )
                    st.session_state.datasets[i]['color'] = color
                
                with col3:
                    marker = st.selectbox(
                        "Marker",
                        options=list(matplotlib_markers.keys()),
                        index=list(matplotlib_markers.keys()).index(dataset['marker']) if dataset['marker'] in matplotlib_markers else 0,
                        key=f"marker_{i}"
                    )
                    st.session_state.datasets[i]['marker'] = marker
                    
                    active = st.checkbox(
                        "Active",
                        value=dataset['active'],
                        key=f"active_{i}"
                    )
                    st.session_state.datasets[i]['active'] = active
                
                # Parse data for preview
                if data_text.strip():
                    df = parse_data(data_text, new_name)
                    if not df.empty:
                        all_data_frames.append(df)
                        
                        # Data preview
                        st.markdown(f"**Preview ({len(df)} points):**")
                        
                        col_preview1, col_preview2 = st.columns([2, 1])
                        with col_preview1:
                            st.dataframe(df[['x', 'y']].head(10), use_container_width=True)
                        
                        with col_preview2:
                            # Show detected separator info
                            detected_sep = detect_separator(data_text)
                            sep_names = {
                                '\t': 'Tab',
                                ' ': 'Space',
                                ',': 'Comma',
                                ';': 'Semicolon'
                            }
                            st.info(f"Detected separator: **{sep_names.get(detected_sep, 'Unknown')}**")
                            
                            # Show statistics
                            st.info(f"Valid points: **{len(df)}**")
                            
                            # Quick format converter button
                            if st.button(f"🔧 Format as {entry_format}", key=f"format_btn_{i}"):
                                # Convert data to selected format
                                lines = data_text.strip().split('\n')
                                formatted_lines = []
                                for line in lines:
                                    line = line.strip()
                                    if not line or line.startswith('#'):
                                        formatted_lines.append(line)
                                        continue
                                    
                                    # Parse and reformat
                                    parsed = parse_data(line, "temp")
                                    if not parsed.empty:
                                        formatted_lines.append(f"{parsed['x'].iloc[0]}{current_separator}{parsed['y'].iloc[0]}")
                                    else:
                                        formatted_lines.append(line)
                                
                                st.session_state.datasets[i]['data'] = '\n'.join(formatted_lines)
                                st.rerun()
        
        # Quick data entry helpers
        st.subheader("Quick Data Entry")
        
        col_help1, col_help2, col_help3 = st.columns(3)
        
        with col_help1:
            if st.button("📋 Paste Example Data (Linear)"):
                example_data = """1.2\t5.1
2.1\t7.3
3.4\t9.8
4.0\t11.2
5.1\t13.4"""
                # Convert to current separator
                lines = example_data.split('\n')
                formatted = []
                for line in lines:
                    if '\t' in line:
                        x, y = line.split('\t')
                        formatted.append(f"{x}{current_separator}{y}")
                st.session_state.datasets[-1]['data'] = '\n'.join(formatted) if st.session_state.datasets else ""
                st.rerun()
        
        with col_help2:
            if st.button("📋 Paste Example Data (Two Groups)"):
                example_data = """4.2\t8.1
4.8\t9.3
5.1\t9.8
14.8\t24.3
15.2\t25.1"""
                # Convert to current separator
                lines = example_data.split('\n')
                formatted = []
                for line in lines:
                    if '\t' in line:
                        x, y = line.split('\t')
                        formatted.append(f"{x}{current_separator}{y}")
                st.session_state.datasets[-1]['data'] = '\n'.join(formatted) if st.session_state.datasets else ""
                st.rerun()
        
        with col_help3:
            if st.button("📋 Paste Example Data (Random)"):
                example_data = """7.2\t18.3
8.1\t22.4
9.3\t16.8
10.5\t24.1
11.8\t19.7"""
                # Convert to current separator
                lines = example_data.split('\n')
                formatted = []
                for line in lines:
                    if '\t' in line:
                        x, y = line.split('\t')
                        formatted.append(f"{x}{current_separator}{y}")
                st.session_state.datasets[-1]['data'] = '\n'.join(formatted) if st.session_state.datasets else ""
                st.rerun()
        
        # Collect all data
        if all_data_frames:
            all_data = pd.concat(all_data_frames, ignore_index=True)

with tab2:
    st.header("Data Visualization")
    
    # Show current display settings
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"📏 Marker Size: **{st.session_state.marker_size}**")
    with col2:
        st.info(f"🔤 Legend Font Size: **{st.session_state.legend_fontsize}**")
    with col3:
        st.info(f"📊 Marginal Type: **{st.session_state.marginal_type.upper()}**")
    
    # Show axis label settings
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📝 Axis Labels: **{st.session_state.axis_label_fontsize}**")
    with col2:
        st.info(f"🔢 Tick Labels: **{st.session_state.tick_fontsize}**")
    
    # Show title settings
    if st.session_state.plot_title:
        st.info(f"📋 Plot Title: **{st.session_state.plot_title}**")
    
    # Show marginal distribution setting
    if st.session_state.marginal_type == 'density':
        density_status = "NORMALIZED" if st.session_state.normalize_density else "ACTUAL DENSITY"
        st.info(f"📊 Marginal Distributions: **{density_status}**")
    
    # Check if there is data for plotting
    has_data = False
    for dataset in st.session_state.datasets:
        if dataset['active'] and dataset['data'].strip():
            df = parse_data(dataset['data'], dataset['name'])
            if not df.empty:
                has_data = True
                break
    
    if not has_data:
        st.warning("No data to display! Please add datasets and enter data in the 'Data' tab.")
    else:
        # Button to create plots
        if st.button("🚀 Create Plots", type="primary"):
            # Collect all data for checking
            all_data_frames_local = []
            for dataset in st.session_state.datasets:
                if dataset['active']:
                    df = parse_data(dataset['data'], dataset['name'])
                    if not df.empty:
                        all_data_frames_local.append(df)
            
            if all_data_frames_local:
                all_data = pd.concat(all_data_frames_local, ignore_index=True)
                
                # Automatically determine boundaries if not manually set
                auto_limits = auto_detect_axis_limits(st.session_state.datasets)
                
                if not st.session_state.x_manual:
                    auto_x_min = auto_limits['x_min']
                    auto_x_max = auto_limits['x_max']
                    auto_x_step = auto_limits['x_step']
                else:
                    auto_x_min = st.session_state.x_min
                    auto_x_max = st.session_state.x_max
                    auto_x_step = st.session_state.x_step
                
                if not st.session_state.y_manual:
                    auto_y_min = auto_limits['y_min']
                    auto_y_max = auto_limits['y_max']
                    auto_y_step = auto_limits['y_step']
                else:
                    auto_y_min = st.session_state.y_min
                    auto_y_max = st.session_state.y_max
                    auto_y_step = st.session_state.y_step
                
                # 1. Main plot with selected marginal type
                st.subheader(f"Scatter Plot with {st.session_state.marginal_type.capitalize()} Marginal Distributions")
                
                # Create the main marginal plot
                fig_main = create_marginal_plot(
                    datasets=st.session_state.datasets,
                    x_label=st.session_state.x_axis_label,
                    y_label=st.session_state.y_axis_label,
                    plot_title=st.session_state.plot_title,
                    legend_title=st.session_state.legend_title,
                    x_min_val=auto_x_min,
                    x_max_val=auto_x_max,
                    y_min_val=auto_y_min,
                    y_max_val=auto_y_max,
                    normalize_density=st.session_state.normalize_density,
                    graph_settings=st.session_state.graph_settings,
                    marginal_type=st.session_state.marginal_type,
                    marker_size=st.session_state.marker_size,
                    legend_fontsize=st.session_state.legend_fontsize,
                    axis_label_fontsize=st.session_state.axis_label_fontsize,
                    tick_fontsize=st.session_state.tick_fontsize,
                    show_legend=True
                )
                
                st.pyplot(fig_main)
                
                # 2. Alternative plots - show all marginal types in a grid
                st.subheader("Compare All Marginal Types")
                
                # Create a 2x2 grid of different marginal types
                marginal_types = ['density', 'histogram', 'boxplot', 'violin']
                
                fig_grid, axes = plt.subplots(2, 2, figsize=(14, 12))
                axes = axes.flatten()
                
                for idx, mtype in enumerate(marginal_types):
                    ax = axes[idx]
                    
                    # Create individual marginal plot
                    fig_single = create_marginal_plot(
                        datasets=st.session_state.datasets,
                        x_label=st.session_state.x_axis_label,
                        y_label=st.session_state.y_axis_label,
                        plot_title="",
                        legend_title="",
                        x_min_val=auto_x_min,
                        x_max_val=auto_x_max,
                        y_min_val=auto_y_min,
                        y_max_val=auto_y_max,
                        normalize_density=st.session_state.normalize_density,
                        graph_settings=st.session_state.graph_settings,
                        marginal_type=mtype,
                        marker_size=st.session_state.marker_size,
                        legend_fontsize=st.session_state.legend_fontsize,
                        axis_label_fontsize=st.session_state.axis_label_fontsize,
                        tick_fontsize=st.session_state.tick_fontsize,
                        show_legend=(idx == 0)  # Only show legend in first plot
                    )
                    
                    # Extract just the main scatter plot from the figure
                    if fig_single is not None:
                        # We'll create a simple scatter plot for each type
                        ax.clear()
                        set_custom_axes(ax, 
                                      border_color=st.session_state.graph_settings['border_color'], 
                                      border_width=st.session_state.graph_settings['border_width'],
                                      grid_color=st.session_state.graph_settings['grid_color'],
                                      grid_visible=st.session_state.graph_settings['grid_visible'],
                                      background_color=st.session_state.graph_settings['background_color'],
                                      tick_fontsize=st.session_state.tick_fontsize,
                                      label_fontsize=st.session_state.axis_label_fontsize)
                        
                        # Add scatter points
                        for dataset in st.session_state.datasets:
                            if dataset['active']:
                                df = parse_data(dataset['data'], dataset['name'])
                                if not df.empty:
                                    ax.scatter(
                                        df['x'], df['y'],
                                        color=dataset['color'],
                                        label=dataset['name'],
                                        marker=matplotlib_markers[dataset['marker']],
                                        s=st.session_state.marker_size/2,
                                        alpha=0.7
                                    )
                        
                        ax.set_xlim(auto_x_min, auto_x_max)
                        ax.set_ylim(auto_y_min, auto_y_max)
                        ax.set_title(f"{mtype.capitalize()} Marginals", fontsize=st.session_state.axis_label_fontsize * 1.1)
                        
                        if idx == 0:
                            ax.legend(title=st.session_state.legend_title, fontsize=st.session_state.legend_fontsize, 
                                    title_fontsize=st.session_state.legend_fontsize, loc='upper right')
                
                # Adjust layout
                plt.tight_layout()
                if st.session_state.plot_title:
                    fig_grid.suptitle(f"Comparison of Different Marginal Types: {st.session_state.plot_title}", 
                                   fontsize=st.session_state.axis_label_fontsize * 1.3, fontweight='bold')
                else:
                    fig_grid.suptitle("Comparison of Different Marginal Distribution Types", 
                                   fontsize=st.session_state.axis_label_fontsize * 1.3, fontweight='bold')
                plt.subplots_adjust(top=0.92)
                st.pyplot(fig_grid)
                
                # 3. Interactive Plotly plot
                st.subheader("Interactive Plot (Plotly)")
                
                fig_plotly = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Scatter Plot: All Samples', 'Scatter Plot',
                                   'Distribution by X', 'Distribution by Y'),
                    vertical_spacing=0.15,
                    horizontal_spacing=0.15
                )
                
                # Add scatter plots with marker size
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
                                        size=st.session_state.marker_size/2,  # Reduce size for Plotly
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
                                        size=st.session_state.marker_size/2,  # Reduce size for Plotly
                                        opacity=0.7
                                    ),
                                    showlegend=False
                                ),
                                row=1, col=2
                            )
                
                # Update layout with font sizes
                fig_plotly.update_xaxes(
                    title_text=format_axis_label(st.session_state.x_axis_label), 
                    row=1, col=1,
                    title_font=dict(size=st.session_state.axis_label_fontsize),
                    tickfont=dict(size=st.session_state.tick_fontsize)
                )
                fig_plotly.update_yaxes(
                    title_text=format_axis_label(st.session_state.y_axis_label), 
                    row=1, col=1,
                    title_font=dict(size=st.session_state.axis_label_fontsize),
                    tickfont=dict(size=st.session_state.tick_fontsize)
                )
                fig_plotly.update_xaxes(
                    title_text=format_axis_label(st.session_state.x_axis_label), 
                    row=1, col=2,
                    title_font=dict(size=st.session_state.axis_label_fontsize),
                    tickfont=dict(size=st.session_state.tick_fontsize)
                )
                fig_plotly.update_yaxes(
                    title_text=format_axis_label(st.session_state.y_axis_label), 
                    row=1, col=2,
                    title_font=dict(size=st.session_state.axis_label_fontsize),
                    tickfont=dict(size=st.session_state.tick_fontsize)
                )
                
                # Apply axis boundaries
                if st.session_state.x_manual and st.session_state.x_min is not None and st.session_state.x_max is not None:
                    fig_plotly.update_xaxes(range=[st.session_state.x_min, st.session_state.x_max], row=1, col=1)
                    fig_plotly.update_xaxes(range=[st.session_state.x_min, st.session_state.x_max], row=1, col=2)
                else:
                    fig_plotly.update_xaxes(range=[auto_x_min, auto_x_max], row=1, col=1)
                    fig_plotly.update_xaxes(range=[auto_x_min, auto_x_max], row=1, col=2)
                
                if st.session_state.y_manual and st.session_state.y_min is not None and st.session_state.y_max is not None:
                    fig_plotly.update_yaxes(range=[st.session_state.y_min, st.session_state.y_max], row=1, col=1)
                    fig_plotly.update_yaxes(range=[st.session_state.y_min, st.session_state.y_max], row=1, col=2)
                else:
                    fig_plotly.update_yaxes(range=[auto_y_min, auto_y_max], row=1, col=1)
                    fig_plotly.update_yaxes(range=[auto_y_min, auto_y_max], row=1, col=2)
                
                # Add plot title
                plot_title_text = st.session_state.plot_title if st.session_state.plot_title else "Interactive Data Visualization"
                
                fig_plotly.update_layout(
                    height=800,
                    title_text=plot_title_text,
                    title_font=dict(size=st.session_state.axis_label_fontsize * 1.2),
                    showlegend=True,
                    hovermode='closest',
                    plot_bgcolor=st.session_state.graph_settings['background_color'],
                    legend=dict(
                        title_text=st.session_state.legend_title,
                        font=dict(size=st.session_state.legend_fontsize),
                        title_font=dict(size=st.session_state.legend_fontsize)
                    )
                )
                
                st.plotly_chart(fig_plotly, use_container_width=True)

with tab3:
    st.header("Data Statistics")
    
    # Collect all data for statistics
    stats_data_frames = []
    for dataset in st.session_state.datasets:
        if dataset['active']:
            df = parse_data(dataset['data'], dataset['name'])
            if not df.empty:
                stats_data_frames.append(df)
    
    if stats_data_frames:
        all_data = pd.concat(stats_data_frames, ignore_index=True)
        
        # General statistics
        st.subheader("General Statistics")
        
        stats_data = []
        for i, dataset in enumerate(st.session_state.datasets):
            if dataset['active']:
                df = parse_data(dataset['data'], dataset['name'])
                if not df.empty:
                    stats = {
                        'Dataset': dataset['name'],
                        'Number of Points': len(df),
                        'X min': f"{df['x'].min():.3f}",
                        'X max': f"{df['x'].max():.3f}",
                        'X mean': f"{df['x'].mean():.3f}",
                        'X std': f"{df['x'].std():.3f}",
                        'Y min': f"{df['y'].min():.3f}",
                        'Y max': f"{df['y'].max():.3f}",
                        'Y mean': f"{df['y'].mean():.3f}",
                        'Y std': f"{df['y'].std():.3f}"
                    }
                    stats_data.append(stats)
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
            # Data export
            st.subheader("Data Export")
            
            # CSV with statistics
            csv_stats = stats_df.to_csv(index=False, sep='\t').encode('utf-8')
            st.download_button(
                label="📥 Download Statistics (CSV)",
                data=csv_stats,
                file_name="data_statistics.csv",
                mime="text/csv"
            )
            
            # Export all data with settings
            all_data_with_settings = export_all_data_with_settings(
                st.session_state.datasets,
                st.session_state.x_axis_label,
                st.session_state.y_axis_label,
                st.session_state.plot_title,
                st.session_state.legend_title,
                st.session_state.x_manual,
                st.session_state.y_manual,
                st.session_state.x_min,
                st.session_state.x_max,
                st.session_state.x_step,
                st.session_state.y_min,
                st.session_state.y_max,
                st.session_state.y_step,
                st.session_state.marker_size,
                st.session_state.legend_fontsize,
                st.session_state.axis_label_fontsize,
                st.session_state.tick_fontsize,
                st.session_state.graph_settings,
                st.session_state.normalize_density
            )
            
            st.download_button(
                label="📥 Download ALL Data with Settings (CSV)",
                data=all_data_with_settings.encode('utf-8'),
                file_name="all_data_with_settings.csv",
                mime="text/csv",
                help="This file contains all data and settings. It can be loaded back into the application."
            )
            
            # Preview of exported data
            with st.expander("Preview of Exported Data with Settings"):
                st.code(all_data_with_settings[:2000], language='text')
                if len(all_data_with_settings) > 2000:
                    st.info(f"And {len(all_data_with_settings) - 2000} more characters...")
        
        # Title and label settings
        st.subheader("Title and Label Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.plot_title:
                st.info(f"**Plot Title:** {st.session_state.plot_title}")
            else:
                st.info("**Plot Title:** Not set")
            
            st.info(f"**X Axis:** {format_axis_label(st.session_state.x_axis_label)}")
            if st.session_state.x_manual:
                st.write(f"Manual configuration: ON")
                st.write(f"Min: {st.session_state.x_min:.3f}" if st.session_state.x_min is not None else "Not set")
                st.write(f"Max: {st.session_state.x_max:.3f}" if st.session_state.x_max is not None else "Not set")
                st.write(f"Step: {st.session_state.x_step:.3f}" if st.session_state.x_step is not None else "Not set")
            else:
                st.write("Manual configuration: OFF")
                auto_limits = auto_detect_axis_limits(st.session_state.datasets)
                st.write(f"Auto-detection: from {auto_limits['x_min']:.3f} to {auto_limits['x_max']:.3f}")
        
        with col2:
            st.info(f"**Legend Title:** {st.session_state.legend_title}")
            
            st.info(f"**Y Axis:** {format_axis_label(st.session_state.y_axis_label)}")
            if st.session_state.y_manual:
                st.write(f"Manual configuration: ON")
                st.write(f"Min: {st.session_state.y_min:.3f}" if st.session_state.y_min is not None else "Not set")
                st.write(f"Max: {st.session_state.y_max:.3f}" if st.session_state.y_max is not None else "Not set")
                st.write(f"Step: {st.session_state.y_step:.3f}" if st.session_state.y_step is not None else "Not set")
            else:
                st.write("Manual configuration: OFF")
                auto_limits = auto_detect_axis_limits(st.session_state.datasets)
                st.write(f"Auto-detection: from {auto_limits['y_min']:.3f} to {auto_limits['y_max']:.3f}")
        
        # Display settings
        st.subheader("Display Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Marker Size:** {st.session_state.marker_size}")
            st.info(f"**Legend Font Size:** {st.session_state.legend_fontsize}")
        
        with col2:
            st.info(f"**Axis Label Font Size:** {st.session_state.axis_label_fontsize}")
            st.info(f"**Tick Label Font Size:** {st.session_state.tick_fontsize}")
        
        # Difference between axis labels and tick labels
        size_difference = st.session_state.axis_label_fontsize - st.session_state.tick_fontsize
        st.info(f"**Size Difference (Labels - Ticks):** {size_difference} points")
        if size_difference >= 2:
            st.success(f"✓ Labels are at least 2 points larger than ticks (difference: {size_difference} points)")
        else:
            st.warning(f"⚠ Labels are only {size_difference} points larger than ticks. Consider increasing the difference to at least 2 points.")
        
        # Marginal distributions setting
        st.subheader("Marginal Distributions Setting")
        if st.session_state.marginal_type == 'density':
            density_status = "NORMALIZED to [0,1]" if st.session_state.normalize_density else "ACTUAL DENSITY VALUES"
            st.info(f"**Marginal Distributions:** {st.session_state.marginal_type.upper()} - {density_status}")
        else:
            st.info(f"**Marginal Distributions:** {st.session_state.marginal_type.upper()}")
        
        # Graph settings
        st.subheader("Graph Settings")
        col1, col2 = st.columns(2)
        with col1:
            grid_status = "ON" if st.session_state.graph_settings['grid_visible'] else "OFF"
            st.info(f"**Grid:** {grid_status}")
            st.info(f"**Border Width:** {st.session_state.graph_settings['border_width']}")
        
        with col2:
            st.info(f"**Background Color:** {st.session_state.graph_settings['background_color']}")
            st.info(f"**Border Color:** {st.session_state.graph_settings['border_color']}")
        
        # Color indicators
        st.subheader("Color Settings")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Background Color:**")
            color_box = f'<div style="width: 100px; height: 30px; background-color: {st.session_state.graph_settings["background_color"]}; border: 1px solid #ccc; border-radius: 5px;"></div>'
            st.markdown(color_box, unsafe_allow_html=True)
        with col2:
            if st.session_state.graph_settings['grid_visible']:
                st.markdown("**Grid Color:**")
                color_box = f'<div style="width: 100px; height: 30px; background-color: {st.session_state.graph_settings["grid_color"]}; border: 1px solid #ccc; border-radius: 5px;"></div>'
                st.markdown(color_box, unsafe_allow_html=True)
        with col3:
            st.markdown("**Border Color:**")
            color_box = f'<div style="width: 100px; height: 30px; background-color: {st.session_state.graph_settings["border_color"]}; border: 1px solid #ccc; border-radius: 5px;"></div>'
            st.markdown(color_box, unsafe_allow_html=True)
        
        # Dataset information
        st.subheader("Dataset Information")
        for i, dataset in enumerate(st.session_state.datasets):
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown(f"**{dataset['name']}**")
            with col2:
                color_box = f'<span style="display: inline-block; width: 20px; height: 20px; background-color: {dataset["color"]}; border-radius: 3px;"></span>'
                st.markdown(f"Color: {color_box}", unsafe_allow_html=True)
            with col3:
                st.markdown(f"Marker: {dataset['marker']}")
            with col4:
                df = parse_data(dataset['data'], dataset['name'])
                st.markdown(f"Points: {len(df)}")
            with col5:
                status = "✅ Active" if dataset['active'] else "❌ Inactive"
                st.markdown(status)
                
    else:
        st.info("No data to display statistics. Add datasets and enter data in the 'Data' tab, then create plots in the 'Plots' tab.")

# Footer
st.markdown("---")
st.markdown("### Usage Instructions:")
st.markdown("""
1. **Sidebar**: 
   - Configure plot title, axis labels, and legend title in "Title and Label Settings"
   - Choose type of marginal distribution: density, histogram, boxplot, or violin
   - For density plots, choose whether to normalize marginal distributions
   - Configure marker size in "Display Settings"
   - Configure font sizes for legend, axis labels, and tick labels
   - Click "➕ Add New Dataset" to create datasets
   - Set axis labels
   - System automatically selects axis boundaries based on data
   - Enable "Configure X/Y Axis" for manual boundary configuration
   - Use "Reset to Automatic Values" button to return to auto-detection
   - Configure graph appearance (background color, grid, borders) in "Graph Appearance"
   - Click "🗑️ Clear All Settings" to reset everything
   - Load previously exported data with settings (optional)
   - Click "Apply Loaded Data" button to use imported settings
2. **'Data' tab**: Enter X and Y values separated by tabs for each dataset. System shows auto-detected boundaries.
3. **'Plots' tab**: Click "Create Plots" button for visualization. All settings are applied.
4. **'Statistics' tab**: 
   - View data statistics and settings
   - Check if labels are at least 2 points larger than ticks
   - Export statistics separately
   - Export ALL data with settings for subsequent loading

**Important**: The "Download ALL Data with Settings" file contains all parameters and can be loaded back via the sidebar.

**Marginal Distributions Types**:
- **Density**: Kernel density estimation plots (can be normalized or actual density)
- **Histogram**: Histograms showing frequency distributions
- **Boxplot**: Box-and-whisker plots showing quartiles and outliers
- **Violin**: Combination of boxplot and density plot, showing full distribution

**developed by @daM, @CTA, https://chimicatechnoacta.ru **.
""")

