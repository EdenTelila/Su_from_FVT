# import all the necessary libraries
import os
import sys
import logging
import platform
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pyodbc
import ezdxf
import tkinter as tk
from tkinter import Toplevel, filedialog, messagebox, ttk
import tksheet
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
pd.set_option('future.no_silent_downcasting', True)
#Declare global variables
df_tt_si = None
df_tt_ne = None
df_tt_no = None
squeezed_columns = None
extracted_data_df = None
su_filter_df = None
cleaned_data_df = None
complete_data_df = None
data_summary_df = None
su_count = None
su_with_atleast_1param_count = None


#file browse function
def browse():
    file_path.set(filedialog.askopenfilename(filetypes=[("tek files", "*.tek")]))

# Configure error logging text
logging.basicConfig(
    filename="error_log.txt",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


# Filter out the FVT data (TT_SI) from the .tek file
def parse_si(file_path):
    """returns a DataFrame of SI results from every test location and depth"""
    #Check if the file exists
    if not os.path.exists(file_path):
        messagebox.showerror("File error", "The selected .tek file was not found")
        logging.error(f"File error - The selected .tek file was not found")
        cleaned_data_df = None
        return pd.DataFrame()
    #Check if the selected tek file is not empty
    if os.path.getsize(file_path) == 0:
        messagebox.showerror("File error", "The selected .tek file is empty")
        logging.error(f"File error - The selected .tek file is empty")
        cleaned_data_df = None
        return pd.DataFrame()

    #Extract test ID, Su measured, Su disturbed measured, and sensitivity parameters from each location and depth
    columns = ['ID_SI', 'X', 'Y', 'Z', 'depth', 'Su_measured', 'Su_disturbed_measured_Suv', 'sensitivity_St']
    data = []
    current_tt = None
    current_id = None
    current_coords = [None, None, None]
    with open(file_path, 'r') as file:
        for line in file:
            line_parts = line.strip().split()
            if line.startswith('-1'):
                current_tt = None
            elif line_parts and line_parts[0] == 'TT': # locate the test type line
                if line_parts[1] == 'SI':   # check if the test type is SI
                    current_tt = 'SI'
                    current_id = line_parts[3] if len(line_parts) > 2 else None
                else:
                    current_tt = None
            elif line_parts and line_parts[0] == 'XY':
                current_coords = [float(line_parts[2]), float(line_parts[1]), float(line_parts[3])]
            elif current_tt == 'SI' and len(line_parts) >= 1:
                try:
                    depth = float(line_parts[0])
                    shear_strength = float(line_parts[1])
                except ValueError as e: # skip lines that can't be converted to floats
                    logging.error(f"There are lines that can't be converted to floats: {e}")
                    continue
                try:
                    disturbed_shear_strength = float(line_parts[2])
                except (ValueError, IndexError):
                    disturbed_shear_strength = 0  # set 0 to values that can't be converted to floats
                try:
                    sensitivity = float(line_parts[3])
                except (ValueError, IndexError):
                    sensitivity = 0    # set 0 to values that can't be converted to floats
                data.append([current_id, *current_coords, depth, shear_strength, disturbed_shear_strength, sensitivity])
    #Check if the tek file is read but no SI test performed
    if len(data) == 0:
        logging.error(f"No data extracted - The file was read, but no valid data could be extracted")
        messagebox.showerror("No data extracted!", "The file was read, but no SI could be extracted")
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=columns)
    return df


# Filter out disturbed and undisturbed sampling data (TT_NE & TT_NO) from the tek file
def parse_no_ne(file_path, tt_type):
    """returns a DataFrame of NE/NO results from every test location and depth"""
    # extract soil type, water content, fineness number, and unit weight parameters from each location and depth
    columns = ['ID', 'X', 'Y', 'Z', 'depth', 'soil_type', 'water content(w)', 'fineness number(F)', 'unit weight(γ)']
    # declare variables
    data = []
    current_tt = None
    current_id = None
    current_coords = [None, None, None]
    current_depth = None
    current_soil_type = None
    current_w = None
    current_f = None
    current_vg = None

    def save_current_row():
        # store one complete row and reset
        nonlocal current_depth, current_soil_type, current_w, current_f, current_vg
        if current_depth is not None and current_w is not None:
            data.append([
                current_id,
                *current_coords,
                current_depth,
                current_soil_type,
                current_w,
                current_f,
                current_vg
            ])
            # reset values except coords & ID
            current_depth = None
            current_soil_type = None
            current_w = None
            current_f = None
            current_vg = None

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line_parts = line.strip().split()
            if line.startswith('-1'):
                save_current_row()
                current_tt = None
            elif line_parts and line_parts[0] == 'TT': # locate the test type line
                save_current_row()
                if line_parts[1] == tt_type: # check if the test sampling type is NE/NO
                    current_tt = tt_type
                    current_id = line_parts[2] if len(line_parts) > 2 else None
                else:
                    current_tt = None
            elif line_parts and line_parts[0] == 'XY' and len(line_parts) >= 4:
                current_coords = [float(line_parts[2]), float(line_parts[1]), float(line_parts[3])]
            elif current_tt == tt_type and len(line_parts) > 0:
                # New depth line
                try:
                    depth = float(line_parts[0])
                    save_current_row()
                    current_depth = depth
                    if not any(char.isdigit() for char in line_parts[-1]):
                        current_soil_type = line_parts[-1]
                    else:
                        current_soil_type = None
                    continue
                except ValueError as e:
                    pass
                # LB lines
                if line_parts[0] == 'LB' and len(line_parts) >= 3:
                    label = line_parts[1]
                    value = line_parts[2]
                    try:
                        if label == 'w': # check if the test is w
                            current_w = float(value)
                        elif label == 'F': # check if the test is F
                            current_f = float(value)
                        elif label == 'VG': # check if the test is VG (Unit weight)
                            current_vg = float(value)
                    except ValueError as e:
                        logging.error(f"Value error in extracting parameters - {e}")
                        continue

    save_current_row()  # Final save after reading all lines

    df = pd.DataFrame(data, columns=columns)
    return df


# Squeeze the merged df by filtering out similar test results from disturbed and undisturbed sampling tests
def squeeze_columns(df):
    """returns a DataFrame containing raw extracted data from the SI along with NE or NO, priority given to NE"""
    df = df.copy()  # copy the df to avoid chained assignment
    # assign parameters given priority to NE test types
    df['unit weight(γ)'] = df.apply(
        lambda row: row['unit weight(γ)_NE'] if pd.notnull(row['unit weight(γ)_NE']) else row['unit weight(γ)_NO'],
        axis=1)
    df['fineness number(F)'] = df.apply(
        lambda row: row['fineness number(F)_NE'] if pd.notnull(row['fineness number(F)_NE']) else row[
            'fineness number(F)_NO'], axis=1)
    df['water content(w)'] = df.apply(
        lambda row: row['water content(w)_NE'] if pd.notnull(row['water content(w)_NE']) else row[
            'water content(w)_NO'], axis=1)
    df['soil_type'] = df.apply(
        lambda row: row['soil_type_NE'] if pd.notnull(row['soil_type_NE']) else row['soil_type_NO'], axis=1)
    # arrange the df containing the raw extracted data
    selected_columns = ['ID_SI', 'X', 'Y', 'Z', 'depth', 'soil_type', 'Su_measured', 'Su_disturbed_measured_Suv',
                        'sensitivity_St', 'unit weight(γ)',
                        'fineness number(F)', 'water content(w)']
    return df[selected_columns]


# Filter out locations where there is Su that has no corresponding w or F tests
def filter_Su(df):
    """returns a DataFrame containing only SI test locations which has a corresponding w or F tests"""
    numeric_cols = ['X', 'Y', 'Z', 'depth', 'Su_measured', 'Su_disturbed_measured_Suv', 'sensitivity_St', 'unit weight(γ) measured',
                        'fineness number(F) measured','water content(w) measured']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') # convert values into numbers
    # Define input parameters
    input_parameters = ['fineness number(F) measured', 'water content(w) measured']
    # Create location groups_ collect similar(x,y)
    df['X'] = df['X'].round(1)
    df['Y'] = df['Y'].round(1)
    grouped_by_location = df.groupby(['X', 'Y'])
    su_count = 0
    su_with_atleast_1param_count = 0
    processed_dfs = []
    df["Su_measured"] = np.where(df["Su_measured"] == 0, np.nan, df["Su_measured"])
    for (x, y), group in grouped_by_location:
        group = group.sort_values('depth').copy()
        has_su = group['Su_measured'].notna().any()     #Check if location has at least one Su value
        has_parameters = group[input_parameters].notna().any().any()    #Check if a location has at least one parameter
        if has_su:
            su_count += 1
            if has_parameters:
                su_with_atleast_1param_count += 1
        if not has_su or not has_parameters:
            continue
        if has_su:
            if has_parameters:
                su_depths = group.loc[group['Su_measured'].notna(), 'depth']
                # define the min and max depth of Su measurements
                min_su_depth = su_depths.min()
                max_su_depth = su_depths.max()
                # Keep all rows where Depth_m is within Su depth range
                mask_within_su_range = (group['depth'] >= min_su_depth) & (group['depth'] <= max_su_depth)
                filtered = group[mask_within_su_range]
                processed_dfs.append(filtered)
    if processed_dfs:
        return pd.concat(processed_dfs).sort_values(['X', 'Y', 'depth']), su_count, su_with_atleast_1param_count
    else:
        return pd.DataFrame(columns=df.columns), su_count, su_with_atleast_1param_count


# Interpolate missing parameters by default after data merging and filtering
def interpolate_data(df):
    """returns a DataFrame with interpolated values of Su, w, F, and unit weight between known parameters"""
    df = df.copy()
    df['Su_measured'] = df['Su_measured'].replace(0, np.nan)
    grouped = df.groupby(['X', 'Y'])
    processed_dfs = []
    for (x, y), group in grouped:
        # Interpolate Su, w, unit_wt, and F values
        group['Su + interpolation'] = (
            group['Su_measured'].interpolate(method='linear', limit_direction='both', limit_area='inside'))
        group['Suv + interpolation'] = (
            group['Su_disturbed_measured_Suv'].interpolate(method='linear', limit_direction='both', limit_area='inside'))

        group['water content(w) + interpolation'] = (
            group['water content(w) measured'].interpolate(method='linear', limit_direction='both'))

        if group['unit weight(γ) measured'].notna().sum() >= 3: # interpolate if there are at least 3 non-zero values per group
            group['unit weight(γ) + interpolation'] = (
                group['unit weight(γ) measured'].interpolate(method='linear', limit_direction='both'))
        else:
            group['unit weight(γ) + interpolation'] = group['unit weight(γ) measured']
        if group[
            'fineness number(F) measured'].notna().sum() >= 2:  # interpolate if there are at least 2 non-zero values per group
            group['fineness number(F) + interpolation'] = (
                group['fineness number(F) measured'].interpolate(method='linear', limit_direction='both'))
        else:
            group['fineness number(F) + interpolation'] = group['fineness number(F) measured']
        processed_dfs.append(group)

    # Combine processed groups
    if processed_dfs:
        result_df = pd.concat(processed_dfs).sort_values(['X', 'Y', 'depth'])
    else:
        result_df = pd.DataFrame(columns=df.columns)
    return result_df


# Parse, extract, merge and clean the data from tek file
def extract_data_and_preprocessing():
    """returns a DataFrame of cleaned data merging and filtering the valid SI, NE, and NO test results"""
    try:
        global cleaned_data_df, extracted_data_df, su_filter_df, df_tt_si, df_tt_no, df_tt_ne, su_count, su_with_atleast_1param_count, squeezed_columns
        # compute the sensitivity on missing lines
        def compute_sensitivity(row):
            """returns sensitivity by dividing the undisturbed by disturbed Su values"""
            if pd.notnull(row['sensitivity_St']):
                return row['sensitivity_St']
            try:
                return row['Su + interpolation'] / row['Suv + interpolation']

            except ZeroDivisionError:
                return 0

        df_tt_si = parse_si(file_path.get())    # parse SI values
        if df_tt_si.empty:
            return

        # Parse NO and NE values
        df_tt_no = parse_no_ne(file_path.get(), "NO")   # parse NO values
        df_tt_ne = parse_no_ne(file_path.get(), "NE")   # parse NE values
        # Check if no and ne dataframes are empty, and notify user that there are no sampling data
        if df_tt_no.empty or df_tt_ne.empty == "0":
            messagebox.showerror("No parameters found", f"The file doesnt have sampling tests (NO or NE)\n"
                                                        f"Please use a tek file with Su and sampling parameters")
            cleaned_data_df = None
            return pd.DataFrame()

        # Round the coordinate data to 1 digit and depth to 3 digits
        df_tt_si['X'] = df_tt_si['X'].round(1)
        df_tt_si['Y'] = df_tt_si['Y'].round(1)
        df_tt_si['Z'] = df_tt_si['Z'].round(1)
        df_tt_si['depth'] = df_tt_si['depth'].round(3)
        df_tt_no['Y'] = df_tt_no['Y'].round(1)
        df_tt_no['X'] = df_tt_no['X'].round(1)
        df_tt_no['Z'] = df_tt_no['Z'].round(1)
        df_tt_no['depth'] = df_tt_no['depth'].round(3)
        df_tt_ne['Y'] = df_tt_ne['Y'].round(1)
        df_tt_ne['X'] = df_tt_ne['X'].round(1)
        df_tt_ne['Z'] = df_tt_ne['Z'].round(1)
        df_tt_ne['depth'] = df_tt_ne['depth'].round(3)

        # Merge data based on location and depth
        merged_df = pd.merge(df_tt_si, df_tt_ne, on=['X', 'Y', 'Z', 'depth'], how='outer', suffixes=('', '_NE'))
        merged_df = pd.merge(merged_df, df_tt_no, on=['X', 'Y', 'Z', 'depth'], how='outer', suffixes=('', '_NO'))
        merged_df = merged_df.rename(columns={'ID': 'ID_NE', 'water content(w)': 'water content(w)_NE',
                                                'fineness number(F)': 'fineness number(F)_NE',
                                                'unit weight(γ)': 'unit weight(γ)_NE', 'soil_type': 'soil_type_NE'})
        merged_df = merged_df.drop(columns=['ID_NE', 'ID_NO'])

        # Squeeze data given priority to ne sample tests
        squeezed_columns = squeeze_columns(merged_df)

        # rename and rearrange squeezed data
        extracted_data_df = squeezed_columns.copy()
        extracted_data_df['Su_measured'] = extracted_data_df['Su_measured'].replace(0, np.nan) #replace nan values by 0
        extracted_data_df = extracted_data_df.rename(
            columns={'unit weight(γ)': 'unit weight(γ) measured', 'fineness number(F)': 'fineness number(F) measured',
                        'water content(w)': 'water content(w) measured'}) # rename parameters

        # Filter su corresponding rows, filtering out those outside min and max su measurement depths
        su_filter_df, su_count, su_with_atleast_1param_count = filter_Su(extracted_data_df)

        # Clean data - interpolating missing values when enough data is available
        cleaned_data_df = interpolate_data(su_filter_df)

        # Filter out the data out of the water content interpolation range as no w means no data to use for correlation.
        cleaned_data_df = cleaned_data_df[cleaned_data_df['water content(w) + interpolation'].notnull()]

        # Compute sensitivity
        cleaned_data_df["sensitivity_St"] = cleaned_data_df.apply(compute_sensitivity, axis=1)
        # Compute F using only correlation with w
        cleaned_data_df['fineness number(F) all correlation'] = (0.7433 * cleaned_data_df[
            'water content(w) + interpolation'] + 11.358).clip(lower=22, upper=200)
        # Create complete F column equal to interpolated F until user fills it either manually or system default methods.
        cleaned_data_df['fineness number(F) complete'] = cleaned_data_df['fineness number(F) + interpolation']
        # Create complete uw column equal to interpolated uw until user fills it either manually or system default methods.
        cleaned_data_df['unit weight(γ) complete'] = cleaned_data_df['unit weight(γ) + interpolation']
        # Define global ID for all the test points using their coordinates as X_Y.
        cleaned_data_df["Global_ID"] = cleaned_data_df["X"].astype(str) + '_' + cleaned_data_df["Y"].astype(str)
        # Assign Point id from test id of SI
        cleaned_data_df["Point_ID"] = cleaned_data_df["ID_SI"]
        # Compute default PI from correlation
        cleaned_data_df['Plasticity index (PI)'] = (0.8343 * cleaned_data_df['fineness number(F) + interpolation'].
                                                    fillna( cleaned_data_df['water content(w) + interpolation']) - 15.531).clip(lower=0)
        # Assign default phi as 21 which can later be updated by the user
        cleaned_data_df["friction angle (φ)"] = 21

        cleaned_data_df = cleaned_data_df[
            ["Global_ID", "Point_ID", "X", "Y", "Z", "depth", "soil_type", "Su_measured", "Su_disturbed_measured_Suv",
                'Su + interpolation', 'Suv + interpolation', "sensitivity_St", "water content(w) measured",
                "water content(w) + interpolation", "fineness number(F) measured", 'fineness number(F) all correlation',
                "fineness number(F) + interpolation", 'fineness number(F) complete', "unit weight(γ) measured",
                "unit weight(γ) + interpolation", 'unit weight(γ) complete', "friction angle (φ)",
                "Plasticity index (PI)"]].round(5)

        # Pickle the cleaned data for missing parameter estimation, save it where the python file is.
        cleaned_data_df.to_pickle("preprocessing.pkl")

        # Data analysis of the extracted, cleaned and interpolated data
        grouped_by_location = extracted_data_df.groupby(['X', 'Y'])
        # Count the boreholes
        w_count = 0
        w_and_su_count = 0
        F_count = 0
        F_and_su_count = 0
        uw_count = 0
        uw_and_su_count = 0
        for (x, y), group in grouped_by_location:
            group = group.sort_values('depth').copy()
            # Check if location has at least one Su value
            has_su = group['Su_measured'].notna().any()
            has_w = group['water content(w) measured'].notna().any()
            has_F = group['fineness number(F) measured'].notna().any()
            has_uw = group['unit weight(γ) measured'].notna().any()
            if has_w:
                w_count += 1
                if has_su:
                    w_and_su_count += 1
            if has_F:
                F_count += 1
                if has_su:
                    F_and_su_count += 1
            if has_uw:
                uw_count += 1
                if has_su:
                    uw_and_su_count += 1
        # Create a DataFrame of data summary
        data_summary_df1 = pd.DataFrame(
            {"Undrained shear": ["strength (Su)", "--------------------",
                                    extracted_data_df['Su_measured'].notna().sum(), su_count,
                                    su_filter_df['Su_measured'].notna().sum(),
                                    su_with_atleast_1param_count, cleaned_data_df['Su + interpolation'].notna().sum(),
                                    0],
                "Water": ["content (w)", "--------------------",
                        extracted_data_df['water content(w) measured'].notna().sum(), w_count,
                        su_filter_df['water content(w) measured'].notna().sum(), w_and_su_count,
                        cleaned_data_df['water content(w) + interpolation'].notna().sum(), 0],
                "Fineness": ["number (F)", "--------------------------",
                            extracted_data_df['fineness number(F) measured'].notna().sum(), F_count,
                            su_filter_df['fineness number(F) measured'].notna().sum(), F_and_su_count,
                            cleaned_data_df['fineness number(F) + interpolation'].notna().sum(), 0],
                "Unit": ["weight (γ)", "------------------------",
                        extracted_data_df['unit weight(γ) measured'].notna().sum(), uw_count,
                        su_filter_df['unit weight(γ) measured'].notna().sum(), uw_and_su_count,
                        cleaned_data_df['unit weight(γ) + interpolation'].notna().sum(), 0],
                "Friction": ["angle (φ)", "------------------------", 0, 0, 0, 0, 0, 0],
                "Plasticity": ["Index (PI)", "-----------------------", 0, 0, 0, 0, 0, 0]
                }, index=[" ", " ", "Total measured test points", "No. of measurement boreholes",
                        "Su test points with other parameters along the depth",
                        "Su Boreholes containing other parameters along the depth",
                        "Total points after interpolation",
                        "Total points after manual entry or system default"]

        )
        data_summary_df1.loc[""] = ["-----------------------"] * data_summary_df1.shape[1]
        data_summary_df1.loc["Total Filled values with interpolation"] = ["0"] * data_summary_df1.shape[1]
        data_summary_df1.loc["Total Filled values (manual entry/system default)"] = ["0"] * data_summary_df1.shape[
            1]
        data_summary_df1.loc["Current Missing parameters"] = data_summary_df1.loc[
                                                                        "Total points after interpolation", "Undrained shear"] - \
                                                                    data_summary_df1.loc[
                                                                        "Total points after interpolation"]
        # Presenting initial data summary table on the GUI
        # Customize column widths
        column_widths = {
            "Parameter": 315,
            "Undrained shear": 100,
            "Water": 70,
            "Fineness": 70,
            "Unit": 65,
            "friction": 65,
            "Plasticity": 65
        }
        tk.Label(text="Data summary", font=("Helvetica", 10, "bold"), anchor="center", justify="center", ).place(
            relx=0.58, rely=0.54)
        tree = ttk.Treeview(root, height=12)
        tree.place(relx=0.54, rely=0.57)
        # Define columns
        tree["columns"] = ["Parameter"] + list(data_summary_df1.columns)
        tree["show"] = "headings"
        # Create columns
        for col in tree["columns"]:
            tree.heading(col, text=col)
            if col == "Parameter":
                tree.column(col, anchor='w', width=column_widths.get(col, 350)) # use custom width or default 350
            else:
                tree.column(col, anchor='center', width=column_widths.get(col, 90))  # use custom width or default 90
        # Insert data rows
        for idx, row in data_summary_df1.iterrows():
            values = [idx] + list(row)
            tree.insert("", "end", values=values)

        # Data summary of only parameters to message box
        su_rows = (cleaned_data_df[cleaned_data_df["fineness number(F) + interpolation"].notna()]
                .drop(columns=["Point_ID","Global_ID",'X','Y','Z','depth', 'Su_measured', 'Su + interpolation', 'soil_type',
                                "Su_disturbed_measured_Suv", 'Suv + interpolation', "sensitivity_St",'fineness number(F) all correlation',
                'fineness number(F) complete', 'unit weight(γ) complete', "friction angle (φ)", "Plasticity index (PI)" ],axis=1))
        nan_counts = su_rows.isna().sum()
        # checking if the SI and NE/NO counts are equal, all data is available
        if w_and_su_count == uw_and_su_count or w_and_su_count == F_and_su_count:
            messagebox.showinfo("Status",
                                f"Data extracted successfully!\nNo missing values/required parameters were reasonably interpolated")
        # present figures of each parameter data via a message box
        else:
            message = (f"Data extracted successfully\n\nBut there are missing parameters out of the {su_rows.shape[0]} "
                        f"FVT Su points:\n\n- Friction angle: {su_rows.shape[0]} missing\n- Plasticity Index (PI): {su_rows.shape[0]} missing\n")
            for col, count in nan_counts.items():
                message += f"- {col}: {count} missing\n"
            messagebox.showwarning("Status",
                                    message + f"\nPlease enter the missing parameters manually or using system default")
    except Exception as e:
        messagebox.showerror("Status",
                               "Data could not be extracted!\nPlease complete the previous step")
        logging.error(f"Error in extracting the data - {e}")


# System default parameter estimations
# Unit weight estimation from correlation
def estimate_unit_weight(row):
    """returns correlated values of uw if it is still empty after the interpolation logic"""
    if pd.notnull(row['unit weight(γ) + interpolation']):
        return row['unit weight(γ) + interpolation']
    if pd.notnull(row['water content(w) + interpolation']):
        uw = ((row['water content(w) + interpolation'] + 100) * 10) / (37.7 + row['water content(w) + interpolation'])
    else:
        return np.nan
    return min(max(uw, 10), 22) # cutoff values of uw, 10 - 22

# Estimating friction angle from PI, sin(phi) = 0.8 - (0.094 * ln(PI)) - [cutoff limit 18-25]
def estimate_friction_angle(row):
    """returns correlated values of phi"""
    if pd.notnull(row['water content(w) + interpolation']):
        plasticity_index = 0.8343 * row['water content(w) + interpolation'] - 15.531  # PI from water content
    else:
        return 21

    if plasticity_index <= 0:
        return 25
    sin_phi = 0.8 - 0.094 * np.log(plasticity_index)
    sin_phi = max(min(sin_phi, 1), -1)
    phi = np.degrees(np.arcsin(sin_phi))
    return min(max(phi, 18), 25)    # cutoff values of uw, 18 - 25


# Display editable spreadsheet like widget to accept user inputs
def open_manual_editor(df):
    """returns a spreadsheet like widget customized with color codes for easier parameter visualization"""
    root.withdraw() # temporarily hide the main window
    df = df.drop(columns="Global_ID")   # drop the global id as it is no use to display for a user input

    # Create a toplevel window
    edit_window = Toplevel(root)
    # Customize the top level window size and title
    edit_window.geometry("1800x900+50+30")
    edit_window.title("Missing parameter estimation_manual entry")
    # Create a frame for the color code legends
    legend_frame = tk.Frame(edit_window)
    legend_frame.pack(anchor="e", pady=(0, 10), padx=(0, 15))
    # Add labels which mentions that the friction angle and PI are set to default values and color legends
    ttk.Label(edit_window,
              text=f"NB: to simplify manual data entry, the friction angle is set to a default value of 21 \n"
                   f"and the PI is prefilled with correlations but you can change it!").place(relx=0.01,
                                                                                              rely=0.02)
    tk.Label(legend_frame, text="Color Legend").grid(row=0, column=0, sticky="e")
    # Put color codes in the legend
    tk.Label(legend_frame, text="Blue - Interpolated values", fg="blue").grid(row=1, column=0,
                                                                              sticky="e")
    tk.Label(legend_frame, text="Red - Correlated values", fg="red").grid(row=2, column=0, sticky="e")
    tk.Label(legend_frame, text="Yellow highlight - from correlation cutoff", bg='#FFEEBD',
             fg="red").grid(row=3, column=0, sticky="e")
    tk.Label(legend_frame,
             text="Green highlight - Please modify here to control results of reduced undrained shear strength, Su",
             bg='#B7FDDF', fg="black").grid(row=4, column=0, sticky="e")

    #Add the data from the DataFrame
    data = df.values.tolist()
    headers = list(df.columns)
    MAX_LEN = 15 # set max width of the columns
    first_row_headers = [] # put length limit for the header
    second_row_headers = [] # put the remaining
    for h in headers:
        if len(h) <= MAX_LEN:
            first_row_headers.append(h)
            second_row_headers.append("")
        else:
            first_row_headers.append(h[:MAX_LEN])
            second_row_headers.append(h[MAX_LEN:])

    # Define units for all the parameters
    units = (["-"] + ["(m)"] * 4 + ["-"] + ["(kPa)", "(kPa)", "(kPa)", "(kPa)", "-"] +
             ["(%)"] * 6 + ["(kN/m3)", "(kN/m3)", "(kN/m3)", "(°)", "(%)"])
    # Insert the extra header row into the table as top rows
    data.insert(0, units)
    data.insert(0, second_row_headers)

    sheet = tksheet.Sheet(edit_window, data=data, headers=first_row_headers)

    sheet.enable_bindings("single_select", "row_select", "column_select", "arrowkeys", "right_click_popup_menu", "copy",
                           "cut", "paste", "delete", "undo", "edit_cell", "column_width_resize")
    sheet.pack(fill="both", expand=True)

    # Apply the color codes defined
    def highlight_if_interpolated_correlated():
        """returns color coded columns, those containing interpolation and correlation data"""
        c1 = df.columns.get_loc("Su + interpolation")
        c2 = df.columns.get_loc("Su_measured")
        c3 = df.columns.get_loc("water content(w) + interpolation")
        c4 = df.columns.get_loc("water content(w) measured")
        c5 = df.columns.get_loc("fineness number(F) measured")
        c6 = df.columns.get_loc("fineness number(F) all correlation")
        c7 = df.columns.get_loc("fineness number(F) + interpolation")
        c8 = df.columns.get_loc("fineness number(F) complete")
        c9 = df.columns.get_loc("unit weight(γ) measured")
        c10 = df.columns.get_loc("unit weight(γ) + interpolation")
        c11 = df.columns.get_loc("unit weight(γ) complete")
        c12 = df.columns.get_loc("friction angle (φ)")
        c13 = df.columns.get_loc("Plasticity index (PI)")

        # loop through the whole rows of specified columns
        for r in range(2, len(df) + 2):
            val1 = sheet.get_cell_data(r, c1)   # Su + interpolation
            val2 = sheet.get_cell_data(r, c2)   # Su_measured
            val3 = sheet.get_cell_data(r, c3)   # water content(w) + interpolation
            val4 = sheet.get_cell_data(r, c4)   # water content(w) measured
            val5 = sheet.get_cell_data(r, c5)   # fineness number(F) measured
            val6 = sheet.get_cell_data(r, c6)   # fineness number(F) all correlation
            val7 = sheet.get_cell_data(r, c7)   # fineness number(F) + interpolation
            val8 = sheet.get_cell_data(r, c8)   # fineness number(F) complete
            val9 = sheet.get_cell_data(r, c9)   # unit weight(γ) measured
            val10 = sheet.get_cell_data(r, c10)   # unit weight(γ) + interpolation
            val11 = sheet.get_cell_data(r, c11)   # unit weight(γ) complete
            val12 = sheet.get_cell_data(r, c12)   # friction angle (φ)
            val13 = sheet.get_cell_data(r, c13)   # Plasticity index (PI)
            if pd.isna(val2) or val2 == "":
                sheet.highlight_cells(row=r, column=c1, fg="blue", bg="#B7FDDF")   #light green bg to give emphasis
            else:
                sheet.highlight_cells(row=r, column=c1, fg="black", bg="#B7FDDF")
            if pd.isna(val4) or val4 == "":
                sheet.highlight_cells(row=r, column=c3, fg="blue")
            if pd.isna(val7) or val7 == "":
                if pd.isna(val5) or val5 == "":
                    try:
                        v8 = float(val8)
                    except (TypeError, ValueError):
                        v8 = 0.0
                        sheet.set_cell_data(r, c8, "0")
                    if v8 <= 22 or v8 >= 200:
                        sheet.highlight_cells(row=r, column=c8, fg='red', bg="#FFE2A3") #yellow bg to give alert
                    else:
                        sheet.highlight_cells(row=r, column=c8, fg="red", bg="#B7FDDF")
                else:
                    sheet.highlight_cells(row=r, column=c8, fg="black", bg="#B7FDDF")
            elif pd.isna(val5) or val5 == "":
                sheet.highlight_cells(row=r, column=c8, fg="blue", bg="#B7FDDF")
                sheet.highlight_cells(row=r, column=c7, fg="blue")
            else:
                sheet.highlight_cells(row=r, column=c8, fg="black", bg="#B7FDDF")
            if pd.isna(val10) or val10 == "":
                if pd.isna(val9) or val9 == "":
                    try:
                        v11 = float(val11)
                    except (TypeError, ValueError):
                        v11 = 0.0
                        sheet.set_cell_data(r, c11, "0")
                    if v11 <= 10 or v11 >= 22: # considering unit weight cutoff 10 - 22
                        sheet.highlight_cells(row=r, column=c11, fg='red', bg="#FFE2A3")
                    else:
                        sheet.highlight_cells(row=r, column=c11, fg='red')
            elif pd.isna(val9) or val9 == "":
                sheet.highlight_cells(row=r, column=c11, fg="blue")
            else:
                sheet.highlight_cells(row=r, column=c11, fg="black")
            try:
                v12 = float(val12)
            except (TypeError, ValueError):
                v12 = 0.0
                sheet.set_cell_data(r, c12, "0")
            if v12 <= 18 or v12 >= 25: # considering friction angle cutoff 18 - 25
                sheet.highlight_cells(row=r, column=c12, fg='red', bg="#FFE2A3")
            else:
                sheet.highlight_cells(row=r, column=c12, fg='red')
            try:
                v6 = float(val6)
            except (TypeError, ValueError):
                v6 = 0.0
                sheet.set_cell_data(r, c6, "0")
            if v6 <= 22 or v6 >= 200: # considering fineness number cutoff 22 - 200
                sheet.highlight_cells(row=r, column=c6, fg='red', bg="#FFE2A3")
            else:
                sheet.highlight_cells(row=r, column=c6, fg="red")
            sheet.highlight_cells(row=r, column=c13, fg="red")

    # call the color coding function for the initial data
    highlight_if_interpolated_correlated()

    def update_coloring_after_edit(event=None):
        """call and apply the color code function"""
        highlight_if_interpolated_correlated()
    # automatically apply color codes when everytime the user edits a cell
    sheet.extra_bindings([("end_edit_cell", update_coloring_after_edit)])  # tksheet event name "end_edit_cell" to detect when editing finishes

    num_cols = len(sheet.headers()) # count the columns (header)
    # Highlight first row white as part of the header
    for c in range(num_cols):
        sheet.highlight_cells(row=0, column=c, fg="#454545", bg="white")
    # Highlight units row grey
    for c in range(num_cols):
        sheet.highlight_cells(row=1, column=c, bg="#F0F0F0")

    # Search the sheet using search box
    def find_value():
        """returns the first matching cell then exit the function"""
        query = entry.get().strip()
        if not query:
            return
        for r, row in enumerate(sheet.get_sheet_data()):    # loop over all cells
            for c, value in enumerate(row):
                if str(query).lower() in str(value).lower():
                    # highlight the cell and scroll to it
                    sheet.see(r, c)  # scroll into view
                    sheet.select_cell(r, c)  # select the cell
                    return
        tk.messagebox.showinfo("Find", f"'{query}' not found")
    # Create a frame for the search option and add the search button, find label and entry functionalities
    search_frame = tk.Frame(edit_window)
    search_frame.pack(fill="x")
    ttk.Label(search_frame, text="Find:").pack(side="left", padx=5)
    entry = ttk.Entry(search_frame)
    entry.pack(side="left", padx=5)
    tk.Button(search_frame, text="Search", command=find_value).pack(side="left")

    # Show warning for an attempt to close the manual editor window without saving
    def when_closing(): messagebox.showwarning("Action Required",
            "Please use the 'Save & Close' button to exit.")

    edit_window.protocol("WM_DELETE_WINDOW", when_closing)  #call the when_closing function by the delete window protocol

    # Create a save and close functions
    def save_and_close():
        global complete_data_df
        updated_data = sheet.get_sheet_data()   # store the updated data
        new_df = pd.DataFrame(updated_data, columns=headers)  #create a new df to store the data from the tksheet
        new_df.drop(index=[0, 1], inplace=True)
        # Bring the global id column to the new df and replace the old pkl file by the new one
        try:
            new_df = new_df.merge(cleaned_data_df[["X", "Y", "Z", "depth", "Global_ID"]], how="left",
                                                  on=["X", "Y", "Z", "depth"])
            no_global_id = new_df.pop('Global_ID')  # remove global id from new_df and name it col
            new_df.insert(0, 'Global_ID', no_global_id)  # put global id back as a first column
            new_df["Su + interpolation"] = pd.to_numeric(new_df["Su + interpolation"], errors="coerce")
            new_df["fineness number(F) complete"] = pd.to_numeric(new_df["fineness number(F) complete"], errors="coerce")
            complete_data_df = new_df   # replace the complete DataFrame
            complete_data_df.to_pickle("preprocessing.pkl") # replace the previously saved pickle data with the updated one
            messagebox.showinfo("Status", "Data updated successfully!")
        except:
            messagebox.showerror("Status", "The data you're seeing is from a previously executed tek file!")

        edit_window.destroy()   # close the edit window

    # Assign the save and close method to the save and close button
    tk.Button(edit_window, text="Save & Close", height=1, width=15, fg= "red", font=("calibri", 11, "bold"),
              command=save_and_close).place(relx=0.01, rely=0.07)
    root.wait_window(edit_window)   # pause the code execution until the edit_window is closed
    root.deiconify()    # bring back the main root window
    return complete_data_df


# Define the logic behind missing parameters estimation methods for both manual and system default options
def missing_parameters_estimation():
    """returns a DataFrame of completed data using either the manual or system default methods"""
    global cleaned_data_df, complete_data_df, data_summary_df, su_count, su_with_atleast_1param_count, su_filter_df, extracted_data_df
    try:
        select = missing_parameter.get() # store user's choice of missing parameter handling option
        complete_data_df = cleaned_data_df.copy()

        # SYSTEM DEFAULT missing parameter estimation
        if select == "default":
            # Estimating F using correlation, F = 0.7433 * w + 11.35 when interpolation is not possible
            complete_data_df['fineness number(F) complete'] = np.where(
                complete_data_df['fineness number(F) + interpolation'].notna(),
                complete_data_df['fineness number(F) + interpolation'],
                complete_data_df['fineness number(F) all correlation'])

            # Estimating unit weight from water content, ((100 + w(%)) * 10) / (37,7 + w(%)) when interpolation is not possible
            complete_data_df['unit weight(γ) complete'] = complete_data_df.apply(estimate_unit_weight, axis=1)

            # Estimating friction angle from PI, sin(phi) = 0.8 - (0.094 * ln(PI)) - [cutoff limit 18-25]
            complete_data_df['friction angle (φ)'] = complete_data_df.apply(estimate_friction_angle, axis=1)
            complete_data_df['Plasticity index (PI)'] = (0.8343 * complete_data_df['fineness number(F) + interpolation'].
                        fillna(complete_data_df['water content(w) + interpolation']) - 15.531).clip(lower=0)

            # Arrange the final complete dataframe
            complete_data_df = complete_data_df[
                ["Global_ID", "Point_ID", "X", "Y", "Z", "depth", "soil_type", "Su_measured", "Su_disturbed_measured_Suv",
                'Su + interpolation', 'Suv + interpolation', "sensitivity_St", "water content(w) measured",
                "water content(w) + interpolation", "fineness number(F) measured",
                "fineness number(F) + interpolation",
                'fineness number(F) all correlation', 'fineness number(F) complete', "unit weight(γ) measured",
                "unit weight(γ) + interpolation", 'unit weight(γ) complete', "friction angle (φ)",
                "Plasticity index (PI)"]].round(5)

            # Replace the old preprocessed pkl file with the system default complete data
            complete_data_df.to_pickle("preprocessing.pkl")

            # data summary after default estimation
            grouped_by_location = extracted_data_df.groupby(['X', 'Y'])
            # counting boreholes
            w_count = 0
            w_and_su_count = 0
            F_count = 0
            F_and_su_count = 0
            uw_count = 0
            uw_and_su_count = 0
            for (x, y), group in grouped_by_location:
                group = group.sort_values('depth').copy()
                # Check if location has at least one Su value
                has_su = group['Su_measured'].notna().any()
                has_w = group['water content(w) measured'].notna().any()
                has_F = group['fineness number(F) measured'].notna().any()
                has_uw = group['unit weight(γ) measured'].notna().any()
                if has_w:
                    w_count += 1
                    if has_su:
                        w_and_su_count += 1
                if has_F:
                    F_count += 1
                    if has_su:
                        F_and_su_count += 1
                if has_uw:
                    uw_count += 1
                    if has_su:
                        uw_and_su_count += 1
            if w_and_su_count == 0:
                messagebox.showerror("No data found!", "No parameters found, please use a tek file with Su and sampling parameters")
            else:
                data_summary_df = pd.DataFrame(
                    {"Undrained shear": ["strength (Su)", "--------------------",extracted_data_df['Su_measured'].notna().sum(), su_count,
                            su_filter_df['Su_measured'].notna().sum(),
                            su_with_atleast_1param_count, cleaned_data_df['Su + interpolation'].notna().sum(),
                            complete_data_df['Su + interpolation'].notna().sum()],
                    "Water":[ "content (w)", "--------------------", extracted_data_df['water content(w) measured'].notna().sum(), w_count, su_filter_df['water content(w) measured'].notna().sum(), w_and_su_count,
                        cleaned_data_df['water content(w) + interpolation'].notna().sum(), complete_data_df['water content(w) + interpolation'].notna().sum()],
                    "Fineness": ["number (F)","--------------------------", extracted_data_df['fineness number(F) measured'].notna().sum(), F_count, su_filter_df['fineness number(F) measured'].notna().sum(), F_and_su_count,
                        cleaned_data_df['fineness number(F) + interpolation'].notna().sum(), complete_data_df['fineness number(F) complete'].notna().sum()],
                    "Unit": ["weight (γ)","------------------------", extracted_data_df['unit weight(γ) measured'].notna().sum(), uw_count,
                                su_filter_df['unit weight(γ) measured'].notna().sum(), uw_and_su_count,
                                cleaned_data_df['unit weight(γ) + interpolation'].notna().sum(),
                                complete_data_df['unit weight(γ) complete'].notna().sum()],
                    "Friction": ["angle (φ)","------------------------", 0, 0, 0, 0, 0, complete_data_df['friction angle (φ)'].notna().sum()],
                    "Plasticity": ["Index (PI)","-----------------------",0, 0, 0, 0, 0, complete_data_df['Plasticity index (PI)'].notna().sum()]
                    }, index=[" ", " ","Total measured test points", "No. of measurement boreholes",
                            "Su test points with other parameters along the depth",
                            "Su Boreholes containing other parameters along the depth",
                            "Total points after interpolation",
                            "Total points after manual entry or system default"]
                )
                data_summary_df.loc[""] = ["-----------------------"] * data_summary_df.shape[1]
                data_summary_df.loc["Total Filled values with interpolation"] = data_summary_df.loc[
                                                                                    "Total points after interpolation"] - \
                                                                                data_summary_df.loc[
                                                                                    "Su test points with other parameters along the depth"]
                data_summary_df.loc["Total Filled values (manual entry/system default)"] = data_summary_df.loc[
                                                                                            "Total points after manual entry or system default"] - \
                                                                                        data_summary_df.loc[
                                                                                            "Total points after interpolation"]
                data_summary_df.loc["Current Missing parameters"] = data_summary_df.loc[
                                                                        "Total points after interpolation", "Undrained shear"] - \
                                                                    data_summary_df.loc[
                                                                        "Total points after manual entry or system default"]

                # # Presenting system default updated data summary table on the GUI
                # Column widths (custom)
                column_widths = {
                    "Parameter": 315,
                    "Undrained shear": 100,
                    "Water": 70,
                    "Fineness": 70,
                    "Unit": 65,
                    "friction": 65,
                    "Plasticity": 65
                }
                tree = ttk.Treeview(root, height=12)
                tree.place(relx=0.54, rely=0.57)
                # Define columns
                tree["columns"] = ["Parameter"] + list(data_summary_df.columns)
                tree["show"] = "headings"
                # Create columns
                for col in tree["columns"]:
                    tree.heading(col, text=col)
                    if col == "Parameter":
                        tree.column(col, anchor='w', width=column_widths.get(col, 350))
                    else:
                        tree.column(col, anchor='center',
                                    width=column_widths.get(col, 90))  # use custom width or default 90
                # Insert data rows
                for idx, row in data_summary_df.iterrows():
                    values = [idx] + list(row)
                    tree.insert("", "end", values=values)

                messagebox.showwarning("Check", f"All missing values are filled with system default method!\n\n"
                                            f"Remember to review system-estimated parameters before proceeding with the calculations.")

        #MANUAL METHOD - Missing parameter estimation
        if select == "manual":
            df = pd.read_pickle("preprocessing.pkl")
            # call the function containing the estimation with the tksheet window for the manual method
            complete_data_df = open_manual_editor(df)

            # data summary after manually estimating missing parameters
            grouped_by_location = extracted_data_df.groupby(['X', 'Y'])
            w_count = 0
            w_and_su_count = 0
            F_count = 0
            F_and_su_count = 0
            uw_count = 0
            uw_and_su_count = 0
            for (x, y), group in grouped_by_location:
                group = group.sort_values('depth').copy()
                # Check if location has at least one Su value
                has_su = group['Su_measured'].notna().any()
                has_w = group['water content(w) measured'].notna().any()
                has_F = group['fineness number(F) measured'].notna().any()
                has_uw = group['unit weight(γ) measured'].notna().any()
                if has_w:
                    w_count += 1
                    if has_su:
                        w_and_su_count += 1
                if has_F:
                    F_count += 1
                    if has_su:
                        F_and_su_count += 1
                if has_uw:
                    uw_count += 1
                    if has_su:
                        uw_and_su_count += 1
            if w_and_su_count == 0:
                messagebox.showerror("No data found!",
                                    "No parameters found, please use a tek file with Su and sampling parameters")
            else:
                data_summary_df = pd.DataFrame(
                    {"Undrained shear": ["strength (Su)", "--------------------",
                                        extracted_data_df['Su_measured'].notna().sum(), su_count,
                                        su_filter_df['Su_measured'].notna().sum(),
                                        su_with_atleast_1param_count,
                                        cleaned_data_df['Su + interpolation'].notna().sum(),
                                        complete_data_df['Su + interpolation'].notna().sum()],
                    "Water": ["content (w)", "--------------------",
                            extracted_data_df['water content(w) measured'].notna().sum(), w_count,
                            su_filter_df['water content(w) measured'].notna().sum(), w_and_su_count,
                            cleaned_data_df['water content(w) + interpolation'].notna().sum(),
                            complete_data_df['water content(w) + interpolation'].notna().sum()],
                    "Fineness": ["number (F)", "--------------------------",
                                extracted_data_df['fineness number(F) measured'].notna().sum(), F_count,
                                su_filter_df['fineness number(F) measured'].notna().sum(), F_and_su_count,
                                cleaned_data_df['fineness number(F) + interpolation'].notna().sum(),
                                complete_data_df['fineness number(F) complete'].notna().sum()],
                    "Unit": ["weight (γ)", "------------------------",
                            extracted_data_df['unit weight(γ) measured'].notna().sum(), uw_count,
                            su_filter_df['unit weight(γ) measured'].notna().sum(), uw_and_su_count,
                            cleaned_data_df['unit weight(γ) + interpolation'].notna().sum(),
                            complete_data_df['unit weight(γ) complete'].notna().sum()],
                    "Friction": ["angle (φ)", "------------------------", 0, 0, 0, 0, 0,
                                complete_data_df['friction angle (φ)'].notna().sum()],
                    "Plasticity": ["Index (PI)", "-----------------------", 0, 0, 0, 0, 0,
                                    complete_data_df['Plasticity index (PI)'].notna().sum()]
                    }, index=[" ", " ", "Total measured test points", "No. of measurement boreholes",
                            "Su test points with other parameters along the depth",
                            "Su Boreholes containing other parameters along the depth",
                            "Total points after interpolation",
                            "Total points after manual entry or system default"]
                )
                data_summary_df.loc[""] = ["-----------------------"] * data_summary_df.shape[1]
                data_summary_df.loc["Total Filled values with interpolation"] = data_summary_df.loc[
                                                                                    "Total points after interpolation"] - \
                                                                                data_summary_df.loc[
                                                                                    "Su test points with other parameters along the depth"]
                data_summary_df.loc["Total Filled values (manual entry/system default)"] = data_summary_df.loc[
                                                                                            "Total points after manual entry or system default"] - \
                                                                                        data_summary_df.loc[
                                                                                            "Total points after interpolation"]
                data_summary_df.loc["Current Missing parameters"] = data_summary_df.loc[
                                                                        "Total points after interpolation", "Undrained shear"] - \
                                                                    data_summary_df.loc[
                                                                        "Total points after manual entry or system default"]
                # Presenting manually updated data summary table on the GUI
                # Column widths (custom)
                column_widths = {
                    "Parameter": 315,
                    "Undrained shear": 100,
                    "Water": 70,
                    "Fineness": 70,
                    "Unit": 65,
                    "friction": 65,
                    "Plasticity": 65
                }
                tree = ttk.Treeview(root, height=12)
                tree.place(relx=0.54, rely=0.57)
                # Define columns
                tree["columns"] = ["Parameter"] + list(data_summary_df.columns)
                tree["show"] = "headings"
                # Create columns
                for col in tree["columns"]:
                    tree.heading(col, text=col)
                    if col == "Parameter":
                        tree.column(col, anchor='w', width=column_widths.get(col, 350))
                    else:
                        tree.column(col, anchor='center',
                                    width=column_widths.get(col, 90))  # use custom width or default 90
                # Insert data rows
                for idx, row in data_summary_df.iterrows():
                    values = [idx] + list(row)
                    tree.insert("", "end", values=values)
    except Exception as e:
        messagebox.showerror("Status", "No data found.\nPlease browse and extract tek file first.")
        logging.error(f"Error in filling missing parameters - {e}")
        return


# Add an option for the user to review and alter the system default parameter estimations
def check_system_filled():
    """returns a DateFrame of system default with manually revised data"""
    global complete_data_df
    try:
        os.path.isfile("preprocessing.pkl") # check if the pkl file from the initial data extraction/updated version is present
        df = pd.read_pickle("preprocessing.pkl")   # create a df from the pickled file
        complete_data_df = open_manual_editor(df)   # update the system default estimations with the manually revised one
        complete_data_df.to_pickle("preprocessing.pkl")    # save the updated version
    except FileNotFoundError:
        messagebox.showerror(f"Status","No data found!\nPlease first click fill missing parameters with system default option")
    except Exception as e:
        messagebox.showerror("Status", "An error occurred while loading the data.\nPlease check the tek file and try again.")
        logging.error(f"Error in filling missing parameters - {e}")

reduction_method_choice = None
final_rounded_df = None
final_df_to_excel = None
searched_df = None


# Su reduction calculation
def su_reduction_calculation():
    """returns a DataFrame of calculation outputs : reduced Su in different stress states and theoretical min Su"""
    global final_rounded_df, df_tt_si, reduction_method_choice, squeezed_columns, complete_data_df
    if complete_data_df["Su + interpolation"].empty:
        messagebox.showerror("Status", "There are no Su measurements with enough w data")
        return

    try:
        # rename parameter names for compatibility to save it to the SQL database
        complete_data_df = complete_data_df.rename(columns={'depth': 'Depth_m', "Su_measured": "Su_FVT_Measured_kPa", 'Su + interpolation': 'Su_FVT_complete', 'Suv + interpolation': 'Suv_FVT_gaps_Interpolated_kPa',
                                                        'sensitivity_St': 'Sensitivity_St', 'unit weight(γ) complete': 'Unit_wt_kN_per_m3',
                                                        "friction angle (φ)": 'friction_angle_phi',"water content(w) + interpolation": 'water_content_w_%',
                                                        'fineness number(F) complete': 'Fineness_number_F_%', "Plasticity index (PI)": 'Plasticity_Index_PI_%' })
    
        # Reduction Method
        def calculate_mu(row, reduction_method_choice, F_option):
            """returns reduction factor, mu for the selected method of choice"""
            F = float(row[F_option])

            if reduction_method_choice == "SGY":
                if F <= 43:
                    return 1
                else:
                    return (0.43 / (F / 100)) ** 0.45  # Assuming wL = F
            elif reduction_method_choice == "Liikenneviraston/Helenelund ":
                if F <= 50:
                    return 1
                else:
                    return 1.5 / (1 + F / 100)
            elif reduction_method_choice == "SGI/Eurocode":
                if F <= 29:
                    return 1.2
                else:
                    return (0.43 / (F / 100)) ** 0.45  # Assuming wL = F
            return None

        # Implement calculation logics
        def su_calculation_and_theoretical_min(df):
            """returns a Dataframe of calculation results"""
            global reduction_method_choice
            df = df.copy()
            reduction_method_choice = reduction_method.get()
            # Calculate reduction factor (mu) based on the selected method
            df['mu'] = df.apply(lambda row: calculate_mu(row, reduction_method_choice,F_option="Fineness_number_F_%"), axis=1)
            df['mu2'] = df.apply(lambda row: calculate_mu(row, reduction_method_choice, F_option="fineness number(F) all correlation"), axis=1)
            # Calculate Reduced Su by multiplying mu with Su_FVT
            df['Reduced_Su_FVT_kPa'] = df['Su_FVT_complete'] * df['mu']
            df['Reduced Su F_all_correlated'] = df['Su_FVT_complete'] * df['mu2']
            # Calculate Su for different stress states
            df['Triaxial_Compression_CKUC'] = df['Reduced_Su_FVT_kPa'] / 0.64
            df['Direct_Simple_Shear_DSS'] = df['Reduced_Su_FVT_kPa'] / 0.64 * 0.61
            df['Triaxial_Extension_CKUE'] = df['Reduced_Su_FVT_kPa'] / 0.64 * 0.49
            df['Plane_Strain_Compression_PSC'] = df['Reduced_Su_FVT_kPa'] / 0.64 * 1.03
            df['Unconsolidated_Undrained_UU'] = df['Reduced_Su_FVT_kPa'] / 0.64 * 0.83
            df['Plane_Strain_Extension_PSE'] = df['Reduced_Su_FVT_kPa'] / 0.64 * 0.58
            # Theoretical minimum Su; MCC and Liikennevirasto, 2018, _ with effective unit weight (assuming unit weight of water as 10kg/m3)
            df['ko'] = df.apply(lambda row: 1 - np.sin(np.radians(row['friction_angle_phi'])), axis=1)  # Ko, at rest earth pressure coefficient
            df['M'] = df.apply(lambda row: (6 * np.sin(np.radians(row['friction_angle_phi']))) / (      # M, slope of the CSL
                    3 - np.sin(np.radians(row['friction_angle_phi']))), axis=1)
            df['p_ini'] = (df['Depth_m'] * (df['Unit_wt_kN_per_m3'] - 10) + 2 * df['Depth_m'] * (       # P_ini, initial mean effective stress
                        df['Unit_wt_kN_per_m3'] - 10) * df['ko']) / 3
            df['q_ini'] = df['Depth_m'] * (df['Unit_wt_kN_per_m3'] - 10) - df['Depth_m'] * (df['Unit_wt_kN_per_m3'] - 10) * df['ko'] # q_ini, initial deviatoric stress
            df['po_iso'] = np.where(df['Depth_m'] == 0, 0, ((df['q_ini'] ** 2) / (df['M'] ** 2 * df['p_ini'])) + df['p_ini'])   # Po, initial isotropic stress
            df['p_csl'] = df['po_iso'] / 2                                                                                      # P_csl, mean effective stress at CSL
            df['q_csl'] = df.apply(lambda row: (row['M'] ** 2 * row['p_csl'] * (row['po_iso'] - row['p_csl'])) ** 0.5, axis=1)  # q_csl deviatoric stress at CSL
            df['min_Su_MCC'] = df['q_csl'] / 2 / 2  # the second division is because the comparison is made with FVT, which is about half of the triaxial compression.
            df['min_Su_Liikennevirasto'] = 0.15 * df['Depth_m'] * (df['Unit_wt_kN_per_m3'] - 10)    # Theoretical min Su from Liikennevirasto ohjeita
            # rearranging the calculation output columns
            selected_columns = ['Global_ID', 'Point_ID', 'X', 'Y', 'Z', 'Depth_m', "soil_type", "Su_FVT_Measured_kPa", 'Su_FVT_complete',
                                'Suv_FVT_gaps_Interpolated_kPa', 'Sensitivity_St', 'unit weight(γ) measured', 'unit weight(γ) + interpolation',
                                'Unit_wt_kN_per_m3', 'friction_angle_phi', 'water content(w) measured', 'water_content_w_%',"fineness number(F) measured",
                                "fineness number(F) + interpolation", 'fineness number(F) all correlation','Fineness_number_F_%', 'Plasticity_Index_PI_%',
                                'mu', 'Reduced Su F_all_correlated', 'Reduced_Su_FVT_kPa', 'Triaxial_Compression_CKUC', 'Direct_Simple_Shear_DSS',
                                'Triaxial_Extension_CKUE','Plane_Strain_Compression_PSC','Unconsolidated_Undrained_UU', 'Plane_Strain_Extension_PSE',
                                'p_ini', 'q_ini', 'po_iso', 'p_csl', 'q_csl', 'ko', 'M', 'min_Su_MCC', 'min_Su_Liikennevirasto']
            return df[selected_columns]
        # store the calculation data to a new data frame
        final_df = su_calculation_and_theoretical_min(complete_data_df)
        # round the calculation output columns to 6 significant digits
        cols_to_round = ["Su_FVT_Measured_kPa", 'Su_FVT_complete', 'Suv_FVT_gaps_Interpolated_kPa', 'Sensitivity_St', 'Unit_wt_kN_per_m3', 'friction_angle_phi',
                        'water_content_w_%',"fineness number(F) measured", "fineness number(F) + interpolation",'fineness number(F) all correlation',
                        'Fineness_number_F_%', 'Plasticity_Index_PI_%','mu', 'ko', 'M', 'Reduced Su F_all_correlated', 'Reduced_Su_FVT_kPa','Triaxial_Compression_CKUC',
                        'Direct_Simple_Shear_DSS', 'Triaxial_Extension_CKUE','Plane_Strain_Compression_PSC','Unconsolidated_Undrained_UU', 'Plane_Strain_Extension_PSE',
                        'p_ini', 'q_ini', 'po_iso', 'p_csl', 'q_csl', 'min_Su_MCC', 'min_Su_Liikennevirasto']
        final_rounded_df = final_df.copy()
        final_rounded_df[cols_to_round] = final_df[cols_to_round].round(6)
        # rename some parameters for better representation
        final_rounded_df = final_rounded_df.rename(columns={"fineness number(F) measured": "fineness_number_F_measured", "Su_FVT_complete": "Su_FVT_gaps_Interpolated_kPa", "mu": "red_factor_mu"})
        final_rounded_df["friction_angle_phi"] = pd.to_numeric(final_rounded_df["friction_angle_phi"], errors="coerce")
        final_rounded_df["Su_FVT_Measured_kPa"] = np.where(final_rounded_df["Su_FVT_Measured_kPa"] == 0, np.nan,
                                                        final_rounded_df["Su_FVT_Measured_kPa"])
        final_rounded_df["Point_ID"] = np.where(final_rounded_df["Point_ID"].notna(), final_rounded_df["Point_ID"], 0)
    except Exception as e:
        messagebox.showerror("Status", f"No data found.\nPlease browse and extract tek file and fill missing parameters first.")
        logging.error(f"No data found - {e}")
        return

    # up on pressing Calculate Store the data with these Server and database details
    try:
        server = 'FI-PF44D7ZX'
        database = 'Su_from_FVT_data'
        table_name = 'Su_from_FVT_Database'

        # store the final_df with the data
        output_store(final_rounded_df, server, database, table_name)
        messagebox.showinfo("Status", "Su,red calculation complete!\n\n"
                                        f"Reduction Method:  {reduction_method_choice}")
    except Exception as e:
        messagebox.showwarning("Status", f"Su,red calculation complete!\nBut the output data could not be stored to the database\n{e}")


# From final dataframe to the local database
def create_database_engine(server, database):
    """returns a connection string to connect to the SQL database"""
    connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;"
    return connection_string


def output_store(df, server, database, table_name):
    """creates a database if not already created and stores data to it"""
    try:
        num_cols = df.select_dtypes(include=['number']).columns
        df[num_cols] = df[num_cols].fillna(0)
        df["soil_type"] = df["soil_type"].fillna(0)
        df = df.rename(columns= {'water_content_w_%':'water_content_w', 'Fineness_number_F_%': 'Fineness_number_F', 'Plasticity_Index_PI_%': 'Plasticity_Index_PI'})
        connection_string = create_database_engine(server, database)
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()  # Define the cursor here

        # Create a table in the SQL server
        cursor.execute(f"""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{table_name}' AND xtype='U')
        CREATE TABLE [{table_name}] (
            Global_ID NVARCHAR(30),
            Point_ID NVARCHAR(30),
            X FLOAT,
            Y FLOAT,
            Z FLOAT,
            Depth_m FLOAT,
            soil_type NVARCHAR(30),
            Unit_wt_kN_per_m3 FLOAT,
            friction_angle_phi FLOAT,
            water_content_w FLOAT,
            fineness_number_F_measured FLOAT,
            Fineness_number_F FLOAT, 
            Plasticity_Index_PI FLOAT,
            Su_FVT_Measured_kPa FLOAT,
            Su_FVT_gaps_Interpolated_kPa FLOAT,
            Suv_FVT_gaps_Interpolated_kPa FLOAT, 
            Sensitivity_St FLOAT,
            red_factor_mu FLOAT,
            Reduced_Su_FVT_kPa FLOAT,
            Triaxial_Compression_CKUC FLOAT,
            Direct_Simple_Shear_DSS FLOAT,
            Triaxial_Extension_CKUE FLOAT,
            Plane_Strain_Compression_PSC FLOAT,
            Unconsolidated_Undrained_UU FLOAT,
            Plane_Strain_Extension_PSE FLOAT,
            p_ini FLOAT,
            q_ini FLOAT,
            po_iso FLOAT,
            p_csl FLOAT,
            q_csl FLOAT,
            M FLOAT,
            ko FLOAT,
            min_Su_MCC FLOAT,
            min_Su_Liikennevirasto FLOAT

        )
        """)

        # Insert or update data if a data in that location and depth is already found
        for index, row in df.iterrows():
            cursor.execute(f"""
            MERGE [{table_name}] AS target
            USING (
                SELECT ? AS Global_ID, ? AS Point_ID, ? AS X, ? AS Y, ? AS Z, ? AS Depth_m, ? AS soil_type, 
                    ? AS Unit_wt_kN_per_m3, ? AS friction_angle_phi, ? AS water_content_w, ? AS fineness_number_F_measured, 
                    ? AS Fineness_number_F, ? AS Plasticity_Index_PI, ? AS Su_FVT_Measured_kPa,
                    ? AS Su_FVT_gaps_Interpolated_kPa, ? AS Suv_FVT_gaps_Interpolated_kPa, ? AS Sensitivity_St, ? AS red_factor_mu, 
                    ? AS Reduced_Su_FVT_kPa, ? AS Triaxial_Compression_CKUC, ? AS Direct_Simple_Shear_DSS, ? AS Triaxial_Extension_CKUE,
                    ? AS Plane_Strain_Compression_PSC, ? AS Unconsolidated_Undrained_UU, ? AS Plane_Strain_Extension_PSE,
                    ? AS p_ini, ? AS q_ini, ? AS po_iso, ? AS p_csl, ? AS q_csl, ? AS ko, ? AS M, ? AS min_Su_MCC, ? AS min_Su_Liikennevirasto
            ) AS source
            ON (
                target.Global_ID = source.Global_ID AND
                target.X = source.X AND
                target.Y = source.Y AND
                target.Z = source.Z AND
                target.Depth_m = source.Depth_m
            )
            WHEN MATCHED THEN
                UPDATE SET Point_ID = source.Point_ID, Depth_m = source.Depth_m, soil_type = source.soil_type, Unit_wt_kN_per_m3 = source.Unit_wt_kN_per_m3,
                        friction_angle_phi = source.friction_angle_phi, water_content_w = source.water_content_w, 
                        fineness_number_F_measured = source.fineness_number_F_measured, Fineness_number_F = source.Fineness_number_F, 
                        Plasticity_Index_PI = source.Plasticity_Index_PI, 
                        Su_FVT_Measured_kPa = source.Su_FVT_Measured_kPa,
                        Su_FVT_gaps_Interpolated_kPa = source.Su_FVT_gaps_Interpolated_kPa, 
                        Suv_FVT_gaps_Interpolated_kPa = source.Suv_FVT_gaps_Interpolated_kPa, Sensitivity_St = source.Sensitivity_St,
                        red_factor_mu = source.red_factor_mu, Reduced_Su_FVT_kPa = source.Reduced_Su_FVT_kPa,
                        Triaxial_Compression_CKUC = source.Triaxial_Compression_CKUC,
                        Direct_Simple_Shear_DSS = source.Direct_Simple_Shear_DSS,
                        Triaxial_Extension_CKUE = source.Triaxial_Extension_CKUE,
                        Plane_Strain_Compression_PSC = source.Plane_Strain_Compression_PSC,
                        Unconsolidated_Undrained_UU = source.Unconsolidated_Undrained_UU,
                        Plane_Strain_Extension_PSE = source.Plane_Strain_Extension_PSE,
                        p_ini = source.p_ini, q_ini = source.q_ini, po_iso = source.po_iso,
                        p_csl = source.p_csl, q_csl = source.q_csl, ko = source.ko, M = source.M,    
                        min_Su_MCC = source.min_Su_MCC, min_Su_Liikennevirasto = source.min_Su_Liikennevirasto

            WHEN NOT MATCHED THEN
                INSERT (Global_ID, Point_ID, X, Y, Z, Depth_m, soil_type, Unit_wt_kN_per_m3, friction_angle_phi, water_content_w, fineness_number_F_measured, Fineness_number_F, Plasticity_Index_PI, Su_FVT_Measured_kPa, 
                        Su_FVT_gaps_Interpolated_kPa, Suv_FVT_gaps_Interpolated_kPa, Sensitivity_St, red_factor_mu,
                        Reduced_Su_FVT_kPa, Triaxial_Compression_CKUC, Direct_Simple_Shear_DSS, Triaxial_Extension_CKUE,
                        Plane_Strain_Compression_PSC, Unconsolidated_Undrained_UU, Plane_Strain_Extension_PSE,
                        p_ini, q_ini, po_iso, p_csl, q_csl, ko, M, min_Su_MCC, min_Su_Liikennevirasto)
                VALUES (source.Global_ID, source.Point_ID, source.X, source.Y, source.Z, source.Depth_m, source.soil_type, source.Unit_wt_kN_per_m3, source.friction_angle_phi, 
                        source.water_content_w, source.fineness_number_F_measured, source.Fineness_number_F, source.Plasticity_Index_PI, source.Su_FVT_Measured_kPa, source.Su_FVT_gaps_Interpolated_kPa,
                        source.Suv_FVT_gaps_Interpolated_kPa, source.Sensitivity_St,  
                        source.red_factor_mu, source.Reduced_Su_FVT_kPa, source.Triaxial_Compression_CKUC, source.Direct_Simple_Shear_DSS, 
                        source.Triaxial_Extension_CKUE, source.Plane_Strain_Compression_PSC, source.Unconsolidated_Undrained_UU, 
                        source.Plane_Strain_Extension_PSE, source.p_ini, source.q_ini, source.po_iso, source.p_csl, source.q_csl, 
                        source.ko, source.M, source.min_Su_MCC, source.min_Su_Liikennevirasto);
        """, (
                row['Global_ID'], row['Point_ID'], row['X'], row['Y'], row['Z'], row['Depth_m'], row["soil_type"],
                row['Unit_wt_kN_per_m3'], row['friction_angle_phi'], row['water_content_w'], row['fineness_number_F_measured'], row['Fineness_number_F'], row['Plasticity_Index_PI'],
                row['Su_FVT_Measured_kPa'],
                row['Su_FVT_gaps_Interpolated_kPa'], row['Suv_FVT_gaps_Interpolated_kPa'], row['Sensitivity_St'], row['red_factor_mu'],
                row['Reduced_Su_FVT_kPa'], row['Triaxial_Compression_CKUC'], row['Direct_Simple_Shear_DSS'],
                row['Triaxial_Extension_CKUE'],
                row['Plane_Strain_Compression_PSC'], row['Unconsolidated_Undrained_UU'], row['Plane_Strain_Extension_PSE'],
                row['p_ini'], row['q_ini'], row['po_iso'], row['p_csl'], row['q_csl'], row['ko'], row['M'], row['min_Su_MCC'],
                row['min_Su_Liikennevirasto']
            ))

        # Commit the data to the table
        conn.commit()
        # Close the connection
        conn.close()
    except:
        pass # avoid database access error when the tool is working on other local machines

# Query the database table by ID_SI
def get_rows_by_global_id(search_id):
    """returns a data frame containing calculation data of a test point from the database using the global id"""
    # Local machine server and database specifications
    server = 'FI-PF44D7ZX'
    database = 'Su_from_FVT_data'
    table_name = 'Su_from_FVT_Database'
    # CONNECT TO THE SERVER
    connection_url = (
        f"mssql+pyodbc://{server}/{database}"
        "?driver=ODBC+Driver+17+for+SQL+Server"
        "&trusted_connection=yes"
    )
    engine = create_engine(connection_url)
    # SQL query using the global id
    query = f"SELECT * FROM {table_name} WHERE Global_ID = '{search_id}'"
    df = pd.read_sql(query, engine)
    return df

# establish an error management and call the query function
def database_query():
    """updates the global DataFrame which goes to plot and download"""
    global searched_df, final_rounded_df
    search_id = Global_ID.get() # store the search id to a variable
    try:
        searched_df = get_rows_by_global_id(f'{search_id}')
        if searched_df.empty:
            messagebox.showerror("Status", "Couldn't find the searched test location")
            final_rounded_df = pd.DataFrame()   ##update the global dataframe to empty
        else:
            messagebox.showinfo("Status", "Test location found\nYou can proceed to plot or download options")
            final_rounded_df = searched_df  # update the global dataframe
    except Exception as e:
        messagebox.showerror("Status", "Couldn't access the database")
        logging.error(f"Error, couldn't access the database - {e}")
        final_rounded_df = pd.DataFrame()   #update the global dataframe to empty


# Plotting all test locations with valid data
def plot_all_locations(df, selected_columns):
    """Plots Su vs depth on a new window for all the stress path options"""
    global reduction_method_choice
    df["Su_FVT_Measured_kPa"] = np.where(df["Su_FVT_Measured_kPa"] == 0, np.nan, df["Su_FVT_Measured_kPa"]) # replace 0 values with nan to avoid plot irregularities
    df["Reduced_Su_from_Measurement"] = np.where(df["fineness_number_F_measured"]!=0, df["Reduced_Su_FVT_kPa"], np.nan) # create a column containing the reduced Su from measured F only
    df["Reduced Su FVT_interp+correl"] = df["Reduced_Su_FVT_kPa"]   # create a column containing the reduced Su from both measured and estimated F values

    # Find unique X and Y values (plot per location)
    unique_coords = df[['X', 'Y']].drop_duplicates()
    num_plots = len(unique_coords)
    NUM_COLUMNS = 2 # set number of plot columns to 2
    num_rows = 1 if (num_plots + NUM_COLUMNS - 1) // NUM_COLUMNS < 1 else (num_plots + NUM_COLUMNS - 1) // NUM_COLUMNS  # Calculate number of rows needed
    fig, axes = plt.subplots(num_rows, NUM_COLUMNS, figsize=(12, num_rows * 8))
    axes = axes.flatten()
    # assigning plot markers and colours for each line
    markers = ['^', 'o', 'o', 'o', 's', 'D', 'v', '<', '>', 'x', '*']
    colors = ['#FF7171', '#1f77b4', 'orange', '#28A745', 'magenta', '#9982FF', 'black', 'cyan', 'yellow', 'navy', 'blue']
    index = 0
    for i in range(len(unique_coords)):
        x_value = unique_coords.iloc[i]['X']
        y_value = unique_coords.iloc[i]['Y']
        ax = axes[index]
        index += 1
        # Filter the original data to get only rows for this (X, Y) location
        filtered_df = df[(df['X'] == x_value) & (df['Y'] == y_value)]
        if filtered_df.empty:
            continue
        i = 0
        for col_index, col in enumerate(selected_columns):
            if col in filtered_df.columns:
                su_values = filtered_df[col]
                depth_values = filtered_df['Depth_m']
                marker = markers[col_index]
                color = colors[col_index]
                label_name = col.replace('_', ' ')
                ax.plot(su_values, depth_values, marker=marker, color=color, label=label_name)
                i += 1
        ax.set_xlabel('Undrained Shear Strength, Su (kN/m²)')
        ax.set_ylabel('Depth (m)')
        ax.invert_yaxis()
        # give plot title as the point id if available or global id if not
        try:
            point_id = filtered_df["Point_ID"][(filtered_df["Point_ID"].notna()) & (filtered_df["Point_ID"] != 0) &
                                               (filtered_df["Point_ID"] != "0") & (filtered_df["Point_ID"] != '-')].unique()[0]
            ax.set_title(f"{point_id}")
        except:
            ax.set_title(f'X={x_value}, Y={y_value}')
        ax.legend()
        ax.grid(True)
        ax.set_xlim(left=0)
        ax.set_ylim(top=0)
    for i in range(num_plots, len(axes)):  # to remove any empty subplots
        fig.delaxes(axes[i])
    plt.tight_layout()
    # plots on a new window
    plot_window = Toplevel()
    plot_window.geometry("1240x950+100+20")
    plot_window.title("Su plots")
    canvas = tk.Canvas(plot_window)
    scrollbar = tk.Scrollbar(plot_window, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    # Add the matplotlib figure to scrollable frame
    figure_canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
    figure_canvas.draw()
    figure_canvas.get_tk_widget().pack(fill="both", expand=True)
    plt.close(fig)

#Receive the user preference plots and map them to the plot names
def plotting():
    """returns selected Su types for plotting with names mapped to the dataframe column names"""
    global final_rounded_df, searched_df
    try:
        if not final_rounded_df.empty: # check if the df is not empty for plotting
            #map the names from the GUI to the output DataFrame
            column_map = {
                "Measured Su, FVT": "Su_FVT_Measured_kPa",
                "Reduced Su, FVT": "Reduced Su FVT_interp+correl",
                "Reduced_Su_from_Measurement": "Reduced_Su_from_Measurement",
                "Su, Triaxial Compression, CKUC": "Triaxial_Compression_CKUC",
                "Su, Direct Simple shear, DSS": "Direct_Simple_Shear_DSS",
                "Su, Triaxial Extension, CKUE": "Triaxial_Extension_CKUE",
                "Su, Plane Strain Compression, PSC": "Plane_Strain_Compression_PSC",
                "Su, Unconsolidated Undrained, UU": "Unconsolidated_Undrained_UU",
                "Su, Plane Strain Extension, PSE": "Plane_Strain_Extension_PSE",
                "Su, Minimum, MCC": "min_Su_MCC",
                "Su, Minimum, Liikennevirasto": "min_Su_Liikennevirasto"
            }
            # check the selected plot types and store them in a list
            selected = []
            if var1.get():
                selected.append("Measured Su, FVT")
            if var2.get():
                selected.append("Reduced Su, FVT")
                selected.append("Reduced_Su_from_Measurement")
            if var3.get():
                selected.append("Su, Triaxial Compression, CKUC")
            if var4.get():
                selected.append("Su, Direct Simple shear, DSS")
            if var5.get():
                selected.append("Su, Triaxial Extension, CKUE")
            if var6.get():
                selected.append("Su, Plane Strain Compression, PSC")
            if var7.get():
                selected.append("Su, Unconsolidated Undrained, UU")
            if var8.get():
                selected.append("Su, Plane Strain Extension, PSE")
            if var9.get():
                selected.append("Su, Minimum, MCC")
            if var10.get():
                selected.append("Su, Minimum, Liikennevirasto")
            selected_columns = [column_map[label] for label in selected]
            if selected_columns:
                selected_columns.insert(0, "Reduced Su F_all_correlated")   # add the fully correlated F Su reduced plots as default
                plot_all_locations(final_rounded_df, selected_columns)
            else:
                messagebox.showinfo("Check Status", "No options selected.\nPlease select which plot to include.")
        else:
            messagebox.showerror("Status", "No data found.\nPlease complete previous steps.")
    except Exception as e:
        messagebox.showerror("Status", "No data found.\nPlease complete previous steps.")
        logging.error(f"No data found error - {e}")


# Open the user guide pdf which contains system default background calculation and correlations used
def open_pdf():
    """checks the platform and opens the user documentation pdf"""
    if getattr(sys, 'frozen', False):  # running in bundle to convert to .exe
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")  # running in normal python environment (not for .exe)
    pdf_path = os.path.join(base_path, "files/SufromFVT_SystemDefaults_and_user_guide.pdf")

    if pdf_path:
        if platform.system() == "Windows": # for windows
            os.startfile(pdf_path)
        elif platform.system() == "Darwin":  # for macOS
            os.system(f"open '{pdf_path}'")
        else:  # for Linux
            os.system(f"xdg-open '{pdf_path}'")


# Create dxf file containing each plot at its respective location
def create_dxf_from_dataframe(df, output_file, scale):
    """Create dxf file containing each plot at its respective location, aligning with the original soundings in civil 3D"""
    doc = ezdxf.new()   # create a new dxf file
    model_space = doc.modelspace()  # create a new model space (model_space) in the created dxf
    # write to the model space
    grouped = df.groupby(['X', 'Y'])    #group each depth and reduced Su at its respective location
    for (x_ref, y_ref), group in grouped:
        depths = group["Depth_m"].tolist()
        su_values = group["Reduced_Su_FVT_kPa"].tolist()
        # Draw a point of test location and ID at the reference x,y
        model_space.add_circle(center=(x_ref, y_ref), radius=0.1, dxfattribs={"color": 7})
        # put the test point id at each test point next to the reference point
        try:
            point_id = group["Point_ID"][(group["Point_ID"] != 0) &
                                         (group["Point_ID"] != "0") & (group["Point_ID"] != '-')].unique()[0]
            model_space.add_text((f"ID: {point_id}"), dxfattribs={"height": 0.5, "color": 7, "insert": (x_ref, y_ref + 0.5)})
        except:
            model_space.add_text(f"ID: {x_ref}_{y_ref}", dxfattribs={"height": 0.5, "color": 7, "insert": (x_ref, y_ref + 0.5)})

        # From now on the y-axis is assumed to act as the Z-axis as soundings in Civil 3D are in x-y plane
        # Draw reference vertical line,
        y_top = y_ref
        y_bottom = y_ref - max(depths)
        model_space.add_line((x_ref, y_top), (x_ref, y_bottom), dxfattribs={"color": 1})
        offset_x = x_ref + (1.2 * scale*10)

        su_points = []
        for depth, su in zip(depths, su_values):
            y_coord = y_ref - depth
            x_coord = offset_x + (su * scale)
            su_points.append((x_coord, y_coord))
            # Text label of the Su values
            model_space.add_text(
                f"{round(su, 1)}",
                dxfattribs={"height": 0.3, "color": 1, "insert": (x_coord + 0.2, y_coord)}
            )
            # Point marker
            model_space.add_circle(center=(x_coord, y_coord), radius=0.1, dxfattribs={"color": 1})

        # Draw polyline after the loop
        if len(su_points) > 1:
            # Add bottom (last depth) closing line to the ref line
            bottom_depth = depths[-1]
            y_bottom_point = y_ref - bottom_depth
            su_points.append((x_ref, y_bottom_point))

            # Add top (first depth) closure to ref line
            top_depth = depths[0]
            y_top_point = y_ref - top_depth
            su_points.insert(0, (x_ref, y_top_point))

            # Now draw the closed polyline
            model_space.add_lwpolyline(su_points, close=True, dxfattribs={"color": 1})
            model_space.add_line((offset_x, y_ref), (offset_x, y_ref - max(depths)), dxfattribs={"color": 1})

        max_su = max(su_values) # take the max Su value to define the scale max limit
        scale_max = int((round(max_su / 10 + 0.5)) * 10)  # round up to nearest 10

        # Add horizontal scale bar beneath the plot
        scale_bar_y = y_bottom - 2  # slightly below the lowest Su depth
        for i in range(0, scale_max + 1, 10):  # put tick marks at every 25
            tick_x = offset_x + i * scale
            model_space.add_line((tick_x, scale_bar_y - 0.1), (tick_x, scale_bar_y + 0.1), dxfattribs={"color": 1})
            model_space.add_text(
                str(i),
                dxfattribs={"height": 0.5, "color": 1, "insert": (tick_x, scale_bar_y - 0.66),"halign": 1,
                            "align_point": (tick_x, scale_bar_y - 0.66)})
        # Draw the scale bar line
        model_space.add_line((offset_x, scale_bar_y), (offset_x + scale_max * scale, scale_bar_y), dxfattribs={"color": 1})
        # Add "Su,reduction" label
        model_space.add_text("Su,reduced_kN/m2",dxfattribs={"height": 0.5, "color": 1, "insert": (offset_x + scale_max * scale + 1, scale_bar_y - 0.5)})

    doc.saveas(Path.home() / "Downloads" / output_file) # save the dxf file in the downloads folder


# Call the dxf creation and download function
def download_output_cad():
    global final_rounded_df
    # Exporting the final data to cad
    try:
        scale = int(cad_scale.get()) / 1000
        if scale > 0:
            output_file = f"Output_Reduced_Su_{file_path.get().split('/')[-1].split('.')[0]}.dxf"   # use the tek file name for the dxf file as well
            create_dxf_from_dataframe(final_rounded_df, output_file, scale)
            messagebox.showinfo("Download",
                                f"Output downloaded as: {output_file}\nPlease check downloads folder")
        else:
            messagebox.showerror("Status", f"Enter a valid cad scale: e.g., 100, 200, ...")
    except Exception as e:
        messagebox.showerror("Status", f"Download failed!\nPlease complete the previous steps: {e}")
        logging.error(f"Error, couldn't download the dxf - {e}")


#Download the created Excel output file to the downloads folder
def download_output_excel():
    """creates and downloads an excel file containing the output data"""
    global data_summary_df, final_rounded_df, final_df_to_excel, squeezed_columns, searched_df
    try:
        final_df_to_excel = final_rounded_df.copy()
        # create a column named Elevation for easier level visualization
        final_df_to_excel["Elevation"] = final_df_to_excel["Z"] - final_df_to_excel["Depth_m"]
        final_df_to_excel = final_df_to_excel.rename(columns={"Z":"Z at ground level"}) #rename Z to best describe its position
        #Rearrange the columns for the Excel output
        final_df_to_excel = final_df_to_excel[
            ['Global_ID', 'Point_ID', 'X', 'Y', "Z at ground level", 'Depth_m', 'Elevation', 'soil_type', 'Su_FVT_Measured_kPa',
            'Su_FVT_gaps_Interpolated_kPa', 'Suv_FVT_gaps_Interpolated_kPa', 'Sensitivity_St', "unit weight(γ) measured",
            "unit weight(γ) + interpolation", 'Unit_wt_kN_per_m3', 'water content(w) measured', 'water_content_w_%',
            "fineness_number_F_measured",  "fineness number(F) + interpolation", 'fineness number(F) all correlation','Fineness_number_F_%',
            'friction_angle_phi', 'Plasticity_Index_PI_%', 'red_factor_mu', 'Reduced_Su_FVT_kPa',
            'Triaxial_Compression_CKUC',
            'Direct_Simple_Shear_DSS', 'Triaxial_Extension_CKUE', 'Plane_Strain_Compression_PSC',
            'Unconsolidated_Undrained_UU',
            'Plane_Strain_Extension_PSE', 'p_ini', 'q_ini', 'po_iso', 'p_csl', 'q_csl', 'ko', 'M', 'min_Su_MCC',
            'min_Su_Liikennevirasto']]
        #store columns which the 0 values need to changed to nan, to avoid confusions in the results
        columns_to_set_0_to_nan = final_df_to_excel.columns.difference(['X', 'Y', "Z at ground level", 'Depth_m', 'Elevation', 'soil_type',
            'Suv_FVT_gaps_Interpolated_kPa', 'Sensitivity_St', 'red_factor_mu', 'Reduced_Su_FVT_kPa', 'Triaxial_Compression_CKUC',
            'Direct_Simple_Shear_DSS', 'Triaxial_Extension_CKUE', 'Plane_Strain_Compression_PSC', 'Unconsolidated_Undrained_UU',
            'Plane_Strain_Extension_PSE', 'p_ini', 'q_ini', 'po_iso', 'p_csl', 'q_csl', 'ko', 'M', 'min_Su_MCC',
            'min_Su_Liikennevirasto'])

        final_df_to_excel[columns_to_set_0_to_nan] = final_df_to_excel[columns_to_set_0_to_nan].replace(0, np.nan) #set columns with 0 values to nan

        # adding a third sheet to customize the table for the maaparametrit excel template
        number_of_rows = len(final_df_to_excel)
        for_maaparametrit_df = pd.DataFrame({'PAALU': [''] * number_of_rows, 'Piste': final_df_to_excel["Point_ID"],
                                            'Syvyys maanpinnasta m': final_df_to_excel["Depth_m"],
                                            'Korkeustaso m': final_df_to_excel["Elevation"],
                                            'Maalaji': final_df_to_excel['soil_type'],
                                            'Humus-pitoisuus %': [''] * number_of_rows,
                                            'W %': final_df_to_excel["water_content_w_%"],
                                            'F %': final_df_to_excel["Fineness_number_F_%"],
                                            'Su kPa': final_df_to_excel["Su_FVT_Measured_kPa"],
                                            'red': final_df_to_excel["red_factor_mu"],
                                            'Su, red kPa': final_df_to_excel["Reduced_Su_FVT_kPa"],
                                            'Su, gaps interpolated kPa': final_df_to_excel["Su_FVT_gaps_Interpolated_kPa"],
                                            'Unit weight kN/m3': final_df_to_excel["Unit_wt_kN_per_m3"],
                                            'Friction angle deg': final_df_to_excel["friction_angle_phi"]
                                            })

        # Color code input parameters based on their way of estimation
        def style_su(row):
            """returns a font color of blue if the Su is interpolated"""
            if pd.notna(row['Su_FVT_gaps_Interpolated_kPa']):
                if pd.isna(row['Su_FVT_Measured_kPa']):
                    return 'color: blue'
            return ''  # default style

        def style_suv(row):
            """returns a font color of blue if the Suv is interpolated"""
            if pd.notna(row['Suv_FVT_gaps_Interpolated_kPa']):
                if pd.isna(row['Su_FVT_Measured_kPa']):
                    return 'color: blue'
            return ''  # default style

        def style_f(row):
            """returns a font color of red with yellow highlight if the F is from cutoff values, red if correlated, and blue if interpolated"""
            if pd.notna(row['Fineness_number_F_%']):
                if pd.isna(row['fineness number(F) + interpolation']):
                    if row['Fineness_number_F_%'] >= 200 or row['Fineness_number_F_%'] <= 22:  # cutoff for F: 22-200
                        return 'background-color: yellow; color: red'
                    else:
                        return 'color: red'
                elif pd.isna(row['fineness_number_F_measured']):
                    return 'color: blue'
            return ''  # default style

        def style_w(row):
            """returns a font color of blue if the w is interpolated"""
            if pd.notna(row['water_content_w_%']):
                if pd.isna(row['water content(w) measured']):
                    return 'color: blue'
            return ''  # default style

        def style_uw(row):
            """returns a font color of red with yellow highlight if the uw is from cutoff values, red if correlated, and blue if interpolated"""
            if pd.notna(row['Unit_wt_kN_per_m3']):
                if pd.isna(row['unit weight(γ) + interpolation']):
                    if row['Unit_wt_kN_per_m3'] >= 22 or row['Unit_wt_kN_per_m3'] <= 10:  # cutoff for uw: 10-22
                        return 'background-color: yellow; color: red'
                    else:
                        return 'color: red'
                elif pd.isna(row['unit weight(γ) measured']):
                    return 'color: blue'
            return ''  # default style

        def style_phi(row):
            """returns a font color of red with yellow highlight if the phi is from cutoff values, red if correlated, and blue if interpolated"""
            if pd.notna(row['friction_angle_phi']):
                if row['friction_angle_phi'] >= 25 or row['friction_angle_phi'] <= 18:  # cutoff for phi: 18-25
                    return 'background-color: yellow; color: red'
                else:
                    return 'color: red'
            return ''  # default style

        # apply color code to the downloaded Excel file depending on interpolated, correlated and threshold values
        preprocessed_df = (final_df_to_excel.style
            .apply(lambda row: [style_su(row) if col == 'Su_FVT_gaps_Interpolated_kPa' else '' for col in final_df_to_excel.columns],axis=1)
            .apply(lambda row: [style_suv(row) if col == 'Suv_FVT_gaps_Interpolated_kPa' else '' for col in
                                final_df_to_excel.columns], axis=1)
            .apply(lambda row: [style_f(row) if col == 'Fineness_number_F_%' else '' for col in
                                final_df_to_excel.columns], axis=1)
            .apply(lambda row: [style_w(row) if col == 'water_content_w_%' else '' for col in
                                final_df_to_excel.columns], axis=1)
            .apply(lambda row: [style_uw(row) if col == 'Unit_wt_kN_per_m3' else '' for col in
                                final_df_to_excel.columns], axis=1)
            .apply(lambda row: [style_f(row) if col == 'fineness number(F) + interpolation' else '' for col in
                                final_df_to_excel.columns], axis=1)
            .apply(lambda row: [style_phi(row) if col == 'friction_angle_phi' else '' for col in final_df_to_excel.columns],axis=1)
            .apply(lambda row: ['color: red' if col == 'fineness number(F) all correlation' else '' for col in
                                final_df_to_excel.columns], axis=1)
            .apply(lambda row: ['color: red' if col == 'Plasticity_Index_PI_%' else '' for col in
                                final_df_to_excel.columns], axis=1)
        )

        # Exporting the final data to excel
        output_file_name = f"Output_data_{file_path.get().split('/')[-1].split('.')[0]}.xlsx"   #use the tek file name for the Excel file as well
        downloads_path = Path.home() / "Downloads" / output_file_name
        preprocessed_df.to_excel(downloads_path,
                                sheet_name='Calculation Output', engine="openpyxl", index=False)

        with pd.ExcelWriter(downloads_path, engine='openpyxl',
                            mode='a') as writer:
            data_summary_df.to_excel(writer, sheet_name='Data Summary', index=True)
            for_maaparametrit_df.to_excel(writer, sheet_name='for maaparametrit excel', index=False)

        messagebox.showinfo("Download\n",f"Output downloaded as: {output_file_name}\nPlease check downloads folder")
    except Exception as e:
        messagebox.showerror("Status", f"Download failed!\nPlease complete the previous steps: {e}")
        logging.error(f"Error, couldn't download the excel - {e}")

# Tooltip definition for the stress path options
class ToolTip:
    """Adds a tool tip to each test types"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text    # text to display
        self.tooltip_window = None  # top level window which opens when hover
        # bind mouse enter and leave events to the widget
        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip_window or not self.text:    #if a tooltip exists or no text provided.
            return
        x = self.widget.winfo_pointerx() + 5    # x offset the screen coordinates from cursor
        y = self.widget.winfo_pointery() + 5    # y offset the screen coordinates from cursor
        self.tooltip_window = tw = tk.Toplevel(self.widget) # make the window a child of the main window (root)
        tw.wm_overrideredirect(True)    # remove the default window styles
        tw.wm_geometry(f"+{x}+{y}") # place the top left corner of the window at (x, y) pixels from top left of the screen
        # put style to the label
        label = tk.Label(tw, text=self.text, background="#E0E0E0", font=("Arial", 9), justify="left")
        label.pack(ipadx=2, ipady=2)

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()   # destroy tool tip window when the cursor leaves
            self.tooltip_window = None


## GUI
root = tk.Tk()  # initiate the main window
# Customize the window with a title, size, and color
root.geometry("1700x980+100+20")
root.config(bg="white")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
root.title("Su from FVT")
# Add an icon for the window
if getattr(sys, 'frozen', False): # running in bundle to convert to .exe
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".") #running in normal python environment (not for .exe)
image_path = os.path.join(base_path, "images\\S.ico")
root.iconbitmap(image_path)

# Heading for the GUI
tk.Label(root, text="Su from FVT", width=10, height=2, bg="#1FABF0", font=("Helvetica", 22, "bold"), anchor="center",
         justify="center", fg="white").pack(expand=True, fill="x")

# Browse tek file and store path to a variable
file_path = tk.StringVar()
frame0 = tk.LabelFrame(root, text= "File selection", bg="white")
frame0.pack(fill="x", padx=5, pady=10, expand=True)
ttk.Label(frame0, text="Browse .tek file").grid(row=0, column=0, sticky="w", padx=5, pady=10)
ttk.Entry(frame0, textvariable=file_path, width=60).grid(row=0, column=1, sticky="w", padx=5)
ttk.Button(frame0, text="Browse", command=browse).grid(row=0, column=2, sticky="w")

# Call the extract and preprocessing function up on clicking Extract
ttk.Button(frame0, text="Extract", command=extract_data_and_preprocessing).grid(row=0, column=3, sticky="w", padx=5)

# Add a search entry box and call the search function
ttk.Label(frame0, text='Enter the test location as "X_Y" to search for an already calculated reduced Su  ').grid(row=1, column=0, columnspan=2,sticky="w", padx=5, pady=10)
Global_ID = tk.StringVar()
ttk.Entry(frame0, textvariable=Global_ID, width=40).grid(row=1, column=2, sticky="w")
ttk.Button(frame0, text="Search", command=database_query).grid(row=1, column=3, sticky="w", padx=5)

# Add a button which calls the user documentation pdf
open_button = tk.Button(frame0, text="Open System Default and User Guide PDF", fg="red", command=open_pdf)
open_button.grid(row=0, column=6, sticky="e", padx=300)

# Display a parameters link to Su diagram on the start page
if getattr(sys, 'frozen', False): # running in bundle to convert to .exe
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".") #running in normal python environment (not for .exe)
image_path2 = os.path.join(base_path, "images/parameters link to Su.png")
Su_parameter_link_image = Image.open(image_path2)
tk_image = ImageTk.PhotoImage(Su_parameter_link_image.resize((550,250)))
image_label = tk.Label(root, image=tk_image, bg="white")
image_label.image = tk_image
image_label.place(relx=0.5867,rely=0.2128)

# Missing Parameter Handling
tk.Label(root, text="\nMissing Parameter Handling Choice").pack(anchor="w", padx=10)
missing_parameter = tk.StringVar(value="manual")
tk.Radiobutton(root, text="Manual Entry", variable=missing_parameter, value="manual", bg="white").pack(anchor="w", padx=30)
tk.Label(root, text="after selecting Manual entry, press fill missing parameters, which opens an excel file to fill the missing parameters manually",
         font=("Arial", 8, "italic"), bg="white").pack(anchor="w", padx=40)
frame2 = tk.Frame(root, bg="white")
frame2.pack(anchor="w", padx=10, pady=5)
tk.Radiobutton(frame2, text="Use System Default", variable=missing_parameter, value="default", bg="white").pack(
    side="left", padx=20)
ttk.Button(root, text="Fill missing parameters", command=missing_parameters_estimation).pack(pady=10)
tk.Button(root, text="Review/Update system filled parameters", fg="#087AFB", command=check_system_filled).pack(padx=30,pady=(1,20), anchor="w")

# Su reduction and correlation with other stress paths
ttk.Label(root, text="\nReduction Method Choice").pack(anchor="w", padx=10)
reduction_method = tk.StringVar(value="Liikenneviraston/Helenelund")    # store the choice of reduction method
methods = [("The Liikenneviraston/Helenelund Method", "Liikenneviraston/Helenelund"),
           ("The SGY Method", "SGY"),
           ("The SGI/Eurocode_1997-2:2007/ Method", "SGI/Eurocode")]
for text, value in methods:
    tk.Radiobutton(root, text=text, variable=reduction_method, value=value, bg="white").pack(anchor="w", padx=30)
ttk.Button(root, text="Calculate", command=su_reduction_calculation).pack()

# Output Plot Preferences
ttk.Label(root, text="\nOutput Plot Preferences").pack(anchor="w", padx=10)
var1 = tk.IntVar()
var2 = tk.IntVar()
var3 = tk.IntVar()
var4 = tk.IntVar()
var5 = tk.IntVar()
var6 = tk.IntVar()
var7 = tk.IntVar()
var8 = tk.IntVar()
var9 = tk.IntVar()
var10 = tk.IntVar()

# store the chosen stress types and add tool tips for the test types
check1 = tk.Checkbutton(root, text="Measured Su, FVT", variable=var1, bg="white")
check1.pack(anchor="w", padx=30)
ToolTip(check1, "Measured Su, FVT - Shear strength measured from the field vane test (FVT).\n"
                "This is the extracted Su data from the tek file.")
check2 = tk.Checkbutton(root, text="Reduced Su, FVT", variable=var2, bg="white")
check2.pack(anchor="w", padx=30)
ToolTip(check2, "Reduced Su, FVT - Su from FVT reduced as per the selected reduction factor method.\n"
                "Reduced Su from interpolated/correlated Fineness number (F) values are coloured different.\n"
                "Note: reduced Su based only on correlated F is shown on the plot by default for a reference.")
check3 = tk.Checkbutton(root, text="Su, Triaxial Compression, CKUC", variable=var3, bg="white")
check3.pack(anchor="w", padx=30)
ToolTip(check3, "Su,CKUC - Anisotropically (Ko) consolidated undrained triaxial compression test.\n"
                "Sample is first consolidated under K₀ stress (σ’1 = K₀*σ’3), then axially compressed without allowing drainage.\n"
                "It is often used for estimating bearing capacity of deep foundations, and for stability calculations by averaging\n"
                "Su,CKUC & Su,CKUE when DSS test is not available.\n"
                "Su,CKUC is calculated in the system by multiplying reduced Su,FVT with a correlation factor of 1.57.")
check4 = tk.Checkbutton(root, text="Su, Direct Simple shear, DSS", variable=var4, bg="white")
check4.pack(anchor="w", padx=30)
ToolTip(check4, "Su,DSS - Direct simple shear test.\n"
                "Cylindrical sample confined in rigs is sheared by sliding its top plate horizontally at a constant vertical stress\n"
                "creating uniform simple-shear deformation throughout. \n"
                "It is a fair representation of soil behaviour under shearing and is generally less sensitive to sample disturbance.\n"
                "It is widely used for embankment and slope stability analysis.\n"
                "Su,DSS  is calculated in the system by multiplying reduced Su,FVT with a correlation factor of 0.95.")
check5 = tk.Checkbutton(root, text="Su, Triaxial Extension, CKUE", variable=var5, bg="white")
check5.pack(anchor="w", padx=30)
ToolTip(check5, "Su,CKUE - Anisotropically (Ko) consolidated undrained triaxial extension test.\n"
                "Sample is first consolidated under K₀ stress (σ’1 = K₀*σ’3), then axial stress is lowered without allowing drainage.\n"
                "It is often used for estimating unloading conditions and low lateral confinement areas, or for stability calculations\n"
                "by averaging Su,CKUE & Su,CKUC when DSS test is not available.\n"
                "Su,CKUE is calculated in the system by multiplying reduced Su,FVT is with a correlation factor of 0.76.")
check6 = tk.Checkbutton(root, text="Su, Plane Strain Compression, PSC", variable=var6, bg="white")
check6.pack(anchor="w", padx=30)
ToolTip(check6, "Su,PSC - Su from Plane Strain Compression.\n"
                "A laterally confined (lateral strain in one horizontal direction is prevented) sample is compressed vertically.\n"
                "Because one direction is restrained, the soil mobilizes higher shear strength than CKUC.\n"
                "It is mainly applicable for long structures where compression is dominant.\n"
                "Su,PSC is calculated in the system by multiplying reduced Su,FVT with a correlation factor of 1.62.")
check7 = tk.Checkbutton(root, text="Su, Unconsolidated Undrained, UU", variable=var7, bg="white")
check7.pack(anchor="w", padx=30)
ToolTip(check7, "Su,UU - Su from Unconsolidated undrained triaxial test.\n"
                "The specimen is axially loaded to failure without prior consolidation step, giving a quick total-stress.\n"
                "It is mainly used for short term (construction stage) calculations.\n"
                "Su,UU is calculated in the system by multiplying reduced Su,FVT with a correlation factor of 1.31.")
check8 = tk.Checkbutton(root, text="Su, Plane Strain Extension, PSE", variable=var8, bg="white")
check8.pack(anchor="w", padx=30)
ToolTip(check8, "Su,PSE - Su from Plane Strain Extension.\n"
                "The specimen is extended axially (horizontal pressure increased or vertical pressure reduced) under plane-strain constraint.\n"
                "It is mainly used for long structures where extension/unloading is dominant.\n"
                "Su,PSE is calculated in the system by multiplying reduced Su,FVT with a correlation factor of 0.9.")
check9 = tk.Checkbutton(root, text="Su, Theoretical Minimum, Modified Cam Clay (MCC) - Su red,FVT comparable", variable=var9, bg="white")
check9.pack(anchor="w", padx=30)
ToolTip(check9, "Su,min_MCC - Theoretical minimum Su calculated using Modified Cam Clay (MCC) model, is formulated assuming no volume\n"
                "change between the initial and critical states; slope of the loading curve, K = 0.\n"
                "The Su from the MCC model is normally equivalent to the triaxial compression,\n"
                "But, in this Su,min_MCC plot, it is divided by 2 for comparison with Su,FVT.")
check10 = tk.Checkbutton(root, text="Su,Theoretical Minimum, from Liikennevirasto guideline", variable=var10, bg="white")
check10.pack(anchor="w", padx=30),
ToolTip(check10, "Su,min_Liikennevirasto - Theoretical minimum Su calculated based on the Finnish Transport Agency's guideline 14/2018,\n"
                 "calculation instructions for the stability of embankments, used for qualitative assessments.\n"
                 "Su,min_Liikennevirasto = 0.15 * σ'v0, this is 15% of the effective in-situ vertical stress.")


# Plotting button
ttk.Button(root, text="Plot", command=plotting).pack(pady=(5,15))

# Cad scale and dxf download
frame3 = tk.Frame(root, bg="white")
frame3.pack(anchor="w", padx=20, pady= (5,2))
tk.Label(frame3, text="Enter cad scale (e.g., 1:200)  1:", bg="white").pack(side="left")
cad_scale = tk.IntVar()
cad_scale.set(200)
tk.Entry(frame3, textvariable=cad_scale).pack(side="left")
tk.Button(frame3, text="download output dxf", command=download_output_cad).pack(side="left", padx=(20, 500))

# Download output as excel
tk.Button(frame3, text="download output excel", command=download_output_excel).pack(side="left")
frame4= tk.Frame(root, bg="white")
frame4.pack(anchor="w", padx=320)

# Info to remind the dxf contains the reduced FVT, not other stress states
tk.Label(frame4, text="The dxf contains reduced Su from FVT (SI)", bg="white").pack(side="left")
line=tk.Frame(root, height=1, bd=0, relief='sunken', bg='black')
line.pack(fill='x', pady=15)

# add label for the tool Version
tk.Label(root, text="Su from FVT automation tool v1.0", width=50, height= 10, bg="#FFFFFF", fg="#969292", font=("calibri", 9, "italic")).pack(pady=(10,5))

# Closing the gui with close, X
def gui_close():
    root.quit()
    root.destroy()
    try:
        os.remove("preprocessing.pkl")
    except:
        pass
# Close the main window with x button
root.protocol("WM_DELETE_WINDOW", gui_close)

root.mainloop() # Start the tkinter event loop
