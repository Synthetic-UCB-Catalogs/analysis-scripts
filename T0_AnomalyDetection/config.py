import pandas as pd
import numpy as np
import plotly.graph_objects as go

import os

class Params:

	FILENAME = './SEVN/MIST/setA/Z0.02/sevn_mist'
	MODELS_FOLDER = os.path.join('./models/', '-'.join(FILENAME.split('/')[1:]))

	SEQ_LEN = 5
	UNITS = 128
	EPOCHS = 20
	STEPS_PER_EPOCH = 100


# --- Component Type Map & Description Function ---
component_type_map = {
    1: "Star: undergoing some form of nuclear fusion", 
    11: "Protostar",
    12: "Star with a hydrogen envelope", 
    121: "Main sequence (H-core burning)",
    122: "Hertzsprung gap", 
    123: "First giant branch (H-shell burning)",
    124: "Core helium burning", 
    125: "Asymptotic giant branch (He-shell burning)",
    1251: "Early AGB", 
    1252: "Thermally pulsing AGB",
    13: "Helium star", 
    131: "Helium main sequence", 
    132: "Helium Hertzsprung gap",
    133: "Helium first giant branch", 
    14: "Carbon star", 
    15: "Chemically homogeneous star",
    2: "White dwarf", 
    21: "Helium WD", 
    22: "Carbon-Oxygen WD", 
    23: "Oxygen-Neon WD",
    3: "Neutron star", 
    4: "Black hole", 
    5: "Planet", 
    6: "Brown dwarf",
    7: "Thorne-Żytkow Object", 
    9: "Other star type",
    -1: "Massless remnant/Destroyed", 
    -2: "Unknown type",
}

# --- Codify Events (from Section 2.3.3) ---
# The '*' character stands for 0 (not specified), 1 (component 1), 2 (component 2), or 3 (both).
# We will create entries for each possibility based on the doc.
# Event codes are strings in the data, as they can be like "2*1" (meaning e.g. "211", "221", "231").

event_map = {
    # 1*: Component * changes type
    "10": "Component (not specified) changes type",
    "11": "Component 1 changes type",
    "12": "Component 2 changes type",
    "13": "Both components change type",

    # 2*: Component * goes supernova
    ## 2*1: runaway thermonuclear explosion (type Ia supernova)
    "201": "Component (not specified) undergoes Type Ia SN", # Ambiguous, assuming * is for component
    "211": "Component 1 undergoes Type Ia SN (single degenerate, accretor)",
    "221": "Component 2 undergoes Type Ia SN (single degenerate, accretor)",
    "231": "Double degenerate merger leading to Type Ia SN", # 231 for double degenerate
    ## 2*2: Core collapse supernova
    "202": "Component (not specified) undergoes core collapse supernova",
    "212": "Component 1 undergoes core collapse supernova",
    "222": "Component 2 undergoes core collapse supernova",
    "232": "Both components undergo core collapse supernova (unlikely simultaneously)", # Or system event
    ## 2*3: Electron capture supernova
    "203": "Component (not specified) undergoes electron capture supernova",
    "213": "Component 1 undergoes electron capture supernova",
    "223": "Component 2 undergoes electron capture supernova",
    "233": "Both components undergo electron capture supernova (unlikely)", # Or system event
    ## 2*4: Pair-instability supernova
    "204": "Component (not specified) undergoes pair-instability supernova",
    "214": "Component 1 undergoes pair-instability supernova",
    "224": "Component 2 undergoes pair-instability supernova",
    "234": "Both components undergo pair-instability supernova (unlikely)", # Or system event
    ## 2*5: Pulsational pair-instability supernova
    "205": "Component (not specified) undergoes pulsational pair-instability supernova",
    "215": "Component 1 undergoes pulsational pair-instability supernova",
    "225": "Component 2 undergoes pulsational pair-instability supernova",
    "235": "Both components undergo pulsational pair-instability supernova (unlikely)", # Or system event
    ## 2*6: Failed supernova (direct collapse into a black hole)
    "206": "Component (not specified) undergoes failed supernova (direct collapse to BH)",
    "216": "Component 1 undergoes failed supernova (direct collapse to BH)",
    "226": "Component 2 undergoes failed supernova (direct collapse to BH)",
    "236": "Both components undergo failed supernova (unlikely)", # Or system event

    # 3*: Component * overflows its Roche lobe
    "30": "Component (not specified) overflows its Roche lobe",
    "31": "Component 1 overflows its Roche lobe",
    "32": "Component 2 overflows its Roche lobe",
    "33": "Both components overflow their Roche lobes (contact binary)",

    # 4*: Component * goes back into its Roche lobe (end of stable mass transfer or end of a contact phase)
    "40": "Component (not specified) goes back into its Roche lobe",
    "41": "Component 1 goes back into its Roche lobe",
    "42": "Component 2 goes back into its Roche lobe",
    "43": "Both components go back into their Roche lobes (contact ends)",

    # 5: the surface of the two components touch. (This is a general event category)
    # More specific sub-events:
    ## 51*: Component * engulfs the companion, triggering a common envelope
    "510": "Component (not specified) engulfs companion (common envelope)",
    "511": "Component 1 engulfs component 2 (common envelope)",
    "512": "Component 2 engulfs component 1 (common envelope)",
    "513": "Double common envelope initiated (both engulf each other or system event)",
    ## 52: the two components merge.
    "52": "The two components merge", # This is a system event, no * needed.
    ## 53: the system initiates a contact phase.
    "53": "The system initiates a contact phase",
    ## 54: the two components collide at periastron.
    "54": "The two components collide at periastron",

    # 8: terminating condition reached, evolution is stopped.
    # (Events starting with 8 are only for the last line of a system's evolution)
    "81": "Termination: Max time reached",
    "82": "Termination: Both components are compact remnants",
    "83": "Termination: The binary system is dissociated",
    "84": "Termination: Only one object is left (e.g. due to merger/disruption)",
    "85": "Termination: Nothing left (both components are massless remnants)",
    # "88": "Termination: The evolution failed. (and anything starting with 88) can be used as an error code."
    # This will be handled by startswith logic.
    "89": "Termination: Other terminating condition different from any previous one",

    # 9: other: any event that does not fit in any previous category.
    "9": "Other: Any event not fitting previous categories", # General '9'
    "90": "Other event (component not specified)", # If 9* convention applies
    "91": "Other event related to component 1",
    "92": "Other event related to component 2",
    "93": "Other event related to both components",


    # -1: no notable events happened in this time step.
    "-1": "No notable events happened in this time step",
    # -2: unknown: the event is not specified. This information is missing.
    "-2": "Unknown: The event is not specified (missing information)",
}



    # --- Constants ---
L_SUN_W = 3.828e26  # Solar luminosity in Watts
R_SUN_M = 6.957e8   # Solar radius in meters
SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann constant in W m^-2 K^-4
M_BOL_SUN = 4.74 # Absolute bolometric magnitude of the Sun


def get_event_description(event_code_str):
    """Gets human-readable description for an event code string."""
    event_code_str = str(event_code_str).strip() # Ensure string and remove whitespace
    if event_code_str.startswith("88"):
        return f"Termination: The evolution failed (Error code: {event_code_str})"
    return event_map.get(event_code_str, f"Unrecognized event code: {event_code_str}")

def get_component_description(type_code):
    """Gets human-readable description for a component type code."""
    try:
        type_code = int(type_code)
    except (ValueError, TypeError):
        return f"Invalid component type: {type_code}"
    return component_type_map.get(type_code, f"Unrecognized type code: {type_code}")

# --- Helper Functions for HR Diagram Calculations ---
def calculate_luminosity_solar(radius_solar, teff_k):
    """Calculates luminosity in solar units given radius in solar units and Teff in Kelvin."""
    if pd.isna(radius_solar) or pd.isna(teff_k) or radius_solar <= 0 or teff_k <= 0:
        return np.nan
    L_watts = 4 * np.pi * (radius_solar * R_SUN_M)**2 * SIGMA_SB * teff_k**4
    return L_watts / L_SUN_W

def luminosity_to_Mbol(luminosity_solar):
    """Converts luminosity in solar units to absolute bolometric magnitude."""
    if pd.isna(luminosity_solar) or luminosity_solar <= 0:
        return np.nan
    return M_BOL_SUN - 2.5 * np.log10(luminosity_solar)

def Mbol_to_Mv_approx(Mbol, Teff):
    """
    Approximate Mv from Mbol.
    """
    if pd.isna(Mbol): return np.nan
    return Mbol # Simplest: Mv approx Mbol

# --- Spectral Type Definitions for Top Axis ---
spectral_type_ticks_logT = {
    'O5': np.log10(44500), 'B0': np.log10(30000), 'B5': np.log10(15200),
    'A0': np.log10(9520), 'A5': np.log10(8200), 'F0': np.log10(7200),
    'F5': np.log10(6650), 'G0': np.log10(5940), 'G5': np.log10(5560),
    'K0': np.log10(5150), 'K5': np.log10(4410), 'M0': np.log10(3840),
    'M5': np.log10(3170)
}

# --- Definitions for Special Markers ---
COMPACT_OBJECT_TYPES = [2, 21, 22, 23, 3, 4] # Numerical type codes for WD, NS, BH
MASSLESS_REMNANT_TYPE = -1 # Numerical type code for a destroyed/massless star




def plot_hr_diagram_from_df(df, title="HR Diagram Evolution",
                            mv_tick_step=None,
                            legend_pos='bottom_right',
                            plot_width=800, 
                            font_size_general=12):
    """
    Plots an interactive HR diagram for two components from a DataFrame.

    Left Y-axis is Absolute Magnitude Mv (linear, brighter up).
    Bottom X-axis shows Teff (log scale, powers of 10 for labels) with grid,
    and Spectral Class letters (OBAFGKM) as annotations below the axis.
    Features include:
    - Evolutionary tracks with arrows indicating temporal progression.
    - Special markers: cross for massless remnants, square for compact objects.
    - "START" annotation at the beginning of each component's track.
    - A semi-transparent Main Sequence guideline in the background.
    - Hover information displaying detailed state parameters for each point.

    Args:
        df (pd.DataFrame): DataFrame containing the evolutionary data. Expected columns include
                           'time', 'Teff1', 'type1', 'radius1' (or 'lum1'), 'Teff2', 'type2',
                           'radius2' (or 'lum2').
        title (str, optional): Title of the plot. Defaults to "HR Diagram Evolution".
        mv_tick_step (float, optional): Step for Mv axis ticks. If None, Plotly's
                                       default tick generation is used. Defaults to None.
        legend_pos (str, optional): Specifies the legend position. Defaults to 'bottom_right'.
                                    Available options:
                                    - 'bottom_right', 'bottom_left', 'top_right', 'top_left'
                                    - 'outside_top_right' (may require margin adjustment)
                                    - 'top_center_horizontal' (horizontal legend above plot)
        plot_width (int, optional): Width of the plot in pixels. Defaults to 800.
        font_size_general (int, optional): Base font size for axis titles, ticks, and legend.
                                          Defaults to 12.
    """
    fig = go.Figure()

    # 1. Add Main Sequence Indication
    ms_log_T = np.array([
        np.log10(42000), np.log10(30000), np.log10(19000), np.log10(9520),
        np.log10(7200), np.log10(5940), np.log10(5150), np.log10(3840), np.log10(3170)
    ])
    ms_Mv_for_plot = np.array([-5.7, -4.0, -1.6, 0.65, 2.7, 4.4, 5.9, 8.8, 12.3])
    ms_sorted_indices = np.argsort(ms_log_T)[::-1]

    fig.add_trace(go.Scatter(
        x=ms_log_T[ms_sorted_indices], y=ms_Mv_for_plot, mode='lines',
        line=dict(color='grey', width=2, dash='dash'), name='Main Sequence (Approx.)',
        opacity=0.5, hoverinfo='skip'
    ))

    processed_data_for_axis_ranging_x = []
    processed_data_for_axis_ranging_y = []

    for i_comp in [1, 2]:
        teff_col = f'Teff{i_comp}'; type_col = f'type{i_comp}'; radius_col = f'radius{i_comp}'
        lum_col = f'lum{i_comp}'; time_col = 'time'

        cols_to_select_map_orig_to_temp = {time_col: 'time', teff_col: 'Teff', type_col: 'type_code_orig'}
        if not all(c in df.columns for c in [time_col, teff_col, type_col]): continue
        has_lum_col = lum_col in df.columns; has_radius_col = radius_col in df.columns

        if has_lum_col and pd.notna(df[lum_col]).any():
            cols_to_select_map_orig_to_temp[lum_col] = 'L_solar_orig'
            data_source_cols = list(cols_to_select_map_orig_to_temp.keys())
        elif has_radius_col and pd.notna(df[radius_col]).any() and pd.notna(df[teff_col]).any():
            cols_to_select_map_orig_to_temp[radius_col] = 'radius_solar_orig'
            data_source_cols = list(cols_to_select_map_orig_to_temp.keys())
        else: continue
            
        temp_comp_df = df[data_source_cols].copy()
        temp_comp_df.rename(columns=cols_to_select_map_orig_to_temp, inplace=True)

        if 'L_solar_orig' in temp_comp_df.columns: temp_comp_df['L_solar'] = temp_comp_df['L_solar_orig']
        elif 'radius_solar_orig' in temp_comp_df.columns and 'Teff' in temp_comp_df.columns:
            temp_comp_df['L_solar'] = temp_comp_df.apply(
                lambda row: calculate_luminosity_solar(row['radius_solar_orig'], row['Teff']), axis=1)
        else: continue

        teff_values_float = temp_comp_df['Teff'].astype(float)
        temp_comp_df['log_Teff'] = np.log10(np.where(teff_values_float > 0, teff_values_float, np.nan))
        l_solar_values_float = temp_comp_df['L_solar'].astype(float)
        temp_comp_df['log_L_solar'] = np.log10(np.where(l_solar_values_float > 0, l_solar_values_float, np.nan))
        temp_comp_df['Mbol'] = temp_comp_df['L_solar'].apply(luminosity_to_Mbol)
        temp_comp_df['Mv'] = temp_comp_df.apply(lambda row: Mbol_to_Mv_approx(row['Mbol'], row['Teff']), axis=1)
        temp_comp_df['y_plot_data'] = temp_comp_df['Mv']
        plot_df = temp_comp_df.dropna(subset=['log_Teff', 'y_plot_data']).sort_values(by='time').copy()

        if plot_df.empty:
            if not temp_comp_df.empty:
                 processed_data_for_axis_ranging_x.append(temp_comp_df[['log_Teff']].copy())
                 processed_data_for_axis_ranging_y.append(temp_comp_df[['y_plot_data']].copy())
            continue

        marker_symbols = []; marker_sizes = []
        for type_code_val in plot_df['type_code_orig']:
            tc = int(type_code_val) if pd.notna(type_code_val) else -2
            if tc == MASSLESS_REMNANT_TYPE: marker_symbols.append('x-thin-open'); marker_sizes.append(10)
            elif tc in COMPACT_OBJECT_TYPES: marker_symbols.append('square'); marker_sizes.append(9)
            else: marker_symbols.append('circle'); marker_sizes.append(7)
        plot_df.loc[:, 'marker_symbol'] = marker_symbols; plot_df.loc[:, 'marker_size'] = marker_sizes

        if not temp_comp_df.empty:
            last_original_type_for_comp = int(temp_comp_df.iloc[-1]['type_code_orig'])
            if last_original_type_for_comp == MASSLESS_REMNANT_TYPE and not plot_df.empty:
                plot_df.loc[plot_df.index[-1], 'marker_symbol'] = 'x-thin-open'
                plot_df.loc[plot_df.index[-1], 'marker_size'] = 10
        
        plot_df.loc[:, 'hover_info'] = plot_df.apply(
            lambda row: f"Type: {get_component_description(row['type_code_orig'])}<br>Teff: {row['Teff']:.0f}K<br>log(L/Lsun): {row['log_L_solar']:.2f}<br>Mv: {row['Mv']:.2f}<br>Time: {row['time']:.2e}", axis=1)

        fig.add_trace(go.Scatter(
            x=plot_df['log_Teff'], y=plot_df['y_plot_data'], mode='lines+markers', name=f'Component {i_comp}',
            marker=dict(symbol=plot_df['marker_symbol'], size=plot_df['marker_size'],
                        color=f'rgba(200,0,0,0.8)' if i_comp == 1 else f'rgba(0,0,200,0.8)',
                        line=dict(width=1, color='DarkSlateGrey')),
            line=dict(width=2, color=f'rgba(255,100,100,0.6)' if i_comp == 1 else f'rgba(100,100,255,0.6)'),
            customdata=plot_df[['hover_info']].values, hovertemplate='%{customdata[0]}<extra></extra>',
            legendgroup=f"group{i_comp}",
        ))
        fig.add_annotation(
            x=plot_df['log_Teff'].iloc[0], y=plot_df['y_plot_data'].iloc[0], xref="x", yref="y",
            text="<b>START</b>", showarrow=False, font=dict(family="Arial", size=font_size_general-2, color="black"),
            align="center", xanchor='left', yanchor='middle', xshift=10)
        for j in range(1, len(plot_df)):
            fig.add_annotation(
                x=plot_df['log_Teff'].iloc[j], y=plot_df['y_plot_data'].iloc[j],
                ax=plot_df['log_Teff'].iloc[j-1], ay=plot_df['y_plot_data'].iloc[j-1],
                xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2,
                arrowsize=2.2, arrowwidth=1.5,
                arrowcolor=f'rgba(255,100,100,0.5)' if i_comp == 1 else f'rgba(100,100,255,0.5)')
        processed_data_for_axis_ranging_x.append(plot_df[['log_Teff']].copy())
        processed_data_for_axis_ranging_y.append(plot_df[['y_plot_data']].copy())

    if not processed_data_for_axis_ranging_x or all(df_item.empty for df_item in processed_data_for_axis_ranging_x):
        min_log_Teff_data, max_log_Teff_data = np.log10(2000), np.log10(60000)
    else:
        valid_dfs_x = [df_item for df_item in processed_data_for_axis_ranging_x if not df_item.empty]
        if not valid_dfs_x: min_log_Teff_data, max_log_Teff_data = np.log10(2000), np.log10(60000)
        else:
            all_dfs_concat_x = pd.concat(valid_dfs_x); min_log_Teff_data = all_dfs_concat_x['log_Teff'].min(); max_log_Teff_data = all_dfs_concat_x['log_Teff'].max()
    if not processed_data_for_axis_ranging_y or all(df_item.empty for df_item in processed_data_for_axis_ranging_y):
        min_y_data, max_y_data = 15, -10
    else:
        valid_dfs_y = [df_item for df_item in processed_data_for_axis_ranging_y if not df_item.empty]
        if not valid_dfs_y: min_y_data, max_y_data = 15, -10
        else:
            all_dfs_concat_y = pd.concat(valid_dfs_y); min_y_data = all_dfs_concat_y['y_plot_data'].min(); max_y_data = all_dfs_concat_y['y_plot_data'].max()

    padding_logT = 0.05; log_T_range_calculated = [max_log_Teff_data + padding_logT, min_log_Teff_data - padding_logT]
    padding_y = 0.5; y_range_calculated = [max_y_data + padding_y, min_y_data - padding_y]

    log_T_tickvals_for_axis = []; log_T_ticktext_for_axis = []
    min_T_for_ticks_calc = 10**np.floor(log_T_range_calculated[1]); max_T_for_ticks_calc = 10**np.ceil(log_T_range_calculated[0])
    current_power_T_calc = min_T_for_ticks_calc
    while current_power_T_calc <= max_T_for_ticks_calc:
        log_val_calc = np.log10(current_power_T_calc)
        if log_T_range_calculated[1] <= log_val_calc <= log_T_range_calculated[0]:
            log_T_tickvals_for_axis.append(log_val_calc)
            if abs(round(np.log10(current_power_T_calc)) - np.log10(current_power_T_calc)) < 1e-5 : log_T_ticktext_for_axis.append(f"10<sup>{int(round(np.log10(current_power_T_calc)))}</sup>")
            else: log_T_ticktext_for_axis.append(f"{current_power_T_calc:g}")
        if current_power_T_calc >= max_T_for_ticks_calc or current_power_T_calc <=0 : break
        current_T_str_calc = f"{current_power_T_calc:.0f}"
        if current_T_str_calc.startswith('1'): next_T_calc = current_power_T_calc * 3
        elif current_T_str_calc.startswith('3'): next_T_calc = current_power_T_calc / 3 * 5
        elif current_T_str_calc.startswith('5'): next_T_calc = current_power_T_calc * 2
        else: next_T_calc = 10**(np.floor(np.log10(current_power_T_calc)) + 1)
        if next_T_calc <= current_power_T_calc :
            next_T_calc = 10**(np.floor(np.log10(current_power_T_calc)) + 1)
            if next_T_calc <= current_power_T_calc : next_T_calc = current_power_T_calc * 10
        current_power_T_calc = next_T_calc
    
    axis_title_font_size = font_size_general + 2
    axis_tick_font_size = font_size_general

    xaxis_config = dict(
        title=dict(
            text="T<sub>eff</sub> / K (Spectral Type)",
            font=dict(size=axis_title_font_size) # Axis title font
        ),
        tickfont=dict(size=axis_tick_font_size),   # Axis tick font
        autorange=False, range=log_T_range_calculated, showgrid=True
    )
    if log_T_tickvals_for_axis:
        sorted_indices_x = np.argsort(log_T_tickvals_for_axis); 
        xaxis_config['tickvals'] = np.array(log_T_tickvals_for_axis)[sorted_indices_x]
        xaxis_config['ticktext'] = np.array(log_T_ticktext_for_axis)[sorted_indices_x]
    else: xaxis_config['nticks'] = 7

    yaxis_config = dict(
        title=dict(
            text="Absolute Magnitude (M<sub>V</sub>)",
            font=dict(size=axis_title_font_size) # Axis title font
        ),
        tickfont=dict(size=axis_tick_font_size),   # Axis tick font
        autorange=False, range=y_range_calculated
    )

    if mv_tick_step:
        start_tick = np.ceil(y_range_calculated[1] / mv_tick_step) * mv_tick_step; end_tick = np.floor(y_range_calculated[0] / mv_tick_step) * mv_tick_step
        mv_primary_ticks = np.arange(start_tick, end_tick + mv_tick_step / 2, mv_tick_step)
        mv_primary_ticks = mv_primary_ticks[(mv_primary_ticks >= y_range_calculated[1]) & (mv_primary_ticks <= y_range_calculated[0])]
        if len(mv_primary_ticks) > 0: yaxis_config['tickvals'] = mv_primary_ticks
        else: yaxis_config['nticks'] = 7
    else: yaxis_config['nticks'] = 7

    legend_options = {
        'bottom_right': dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        'bottom_left':  dict(yanchor="bottom", y=0.01, xanchor="left",  x=0.01),
        'top_right':    dict(yanchor="top",    y=0.99, xanchor="right", x=0.99),
        'top_left':     dict(yanchor="top",    y=0.99, xanchor="left",  x=0.01),
        'outside_top_right': dict(yanchor="top", y=1.02, xanchor="left", x=1.02),
        'top_center_horizontal': dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    }
    selected_legend_config = legend_options.get(legend_pos, legend_options['bottom_right'])
    selected_legend_config['bgcolor'] = "rgba(255,255,255,0.7)"
    selected_legend_config['bordercolor'] = "rgba(0,0,0,0.5)"
    selected_legend_config['borderwidth'] = 1
    selected_legend_config['font'] = dict(size=font_size_general)


    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=axis_title_font_size + 2)
        ),
        xaxis=xaxis_config,
        yaxis=yaxis_config,
        hovermode="closest",
        legend=selected_legend_config,
        width=plot_width,
        margin=dict(t=80,b=80,l=70,r=40)
    )

    spt_class_centers_logT_anno = {'O': np.log10(40000), 'B': np.log10(20000), 'A': np.log10(8700), 'F': np.log10(6700), 'G': np.log10(5600), 'K': np.log10(4400), 'M': np.log10(3200)}
    for spt, logT_val in spt_class_centers_logT_anno.items():
        if log_T_range_calculated[1] <= logT_val <= log_T_range_calculated[0]:
            fig.add_annotation(x=logT_val, y=0, xref="x", yref="paper", text=f"<b>{spt}</b>", showarrow=False,
                               font=dict(family="Arial", size=font_size_general, color="black"), yanchor='top', yshift=-5)
    fig.show()
