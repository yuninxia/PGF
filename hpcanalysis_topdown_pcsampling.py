import hpcanalysis
import pandas as pd

# Configure pandas for better display of DataFrames
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1500)
pd.set_option('display.precision', 3)
pd.set_option('display.max_colwidth', None)

DB_PATH = "/home/yx87/pg/ft-codellama-sample-hpctoolkit/hpctoolkit-matmul-database"

# 1. Define Target Metrics
# Base names for querying hpcanalysis
QUERY_BASE_NAMES = [
    "gins",
    "gins:stl_any",
    "gins:stl_ifet",
    "gins:stl_idep",
    "gins:stl_gmem",
    "gins:stl_cmem",
    "gins:stl_pipe",
    "gins:stl_mthr",
]

# Display names for the final table, matching hpcviewer screenshot
TARGET_METRICS_DISPLAY = [
    "GINS: Sum (I)",
    "GINS:STL_ANY: Sum (I)",
    "GINS:STL_IFET: Sum (I)",
    "GINS:STL_IDEP: Sum (I)",
    "GINS:STL_GMEM: Sum (I)",
    "GINS:STL_CMEM: Sum (I)",
    "GINS:STL_PIPE: Sum (I)",
    "GINS:STL_MTHR: Sum (I)",
]

# --- Helper function to reconstruct CCT paths ---
def get_cct_paths(cct_df, file_map, func_map):
    """
    Reconstructs full paths for CCT nodes using provided file and function name maps.
    cct_df should be the DataFrame from query_api.query_cct(), with node ID as index.
    It needs 'parent', 'name', 'file_path', 'line', 'type' columns.
    file_map: Dictionary mapping file IDs to path strings.
    func_map: Dictionary mapping function IDs to name strings.
    """
    nodes = cct_df.to_dict('index')
    paths = {}

    def get_node_str(node_id):
        if node_id == 1 and nodes[node_id].get('type') == 'entry':
            return "<program root>"
        if node_id not in nodes:
            return f"Unknown_Node_ID:{node_id}"
        
        node = nodes[node_id]
        node_name_id = node.get('name')
        file_id = node.get('file_path')
        line = node.get('line', 0)
        node_type = node.get('type', '')

        # Resolve names and paths using maps
        resolved_name = str(node_name_id) # Default to ID if not resolved
        if node_type == 'function' and node_name_id in func_map:
            resolved_name = func_map[node_name_id]
        elif node_type == 'function': # Function ID not in map
            resolved_name = f"<unknown function: {node_name_id}>"

        resolved_file_path = str(file_id) # Default to ID
        if file_id in file_map:
            resolved_file_path = file_map[file_id]
        elif file_id is not None: # File ID not in map but exists
            resolved_file_path = f"<unknown file: {file_id}>"

        if node_type == 'function':
            return resolved_name
        if node_type == 'loop' and file_id is not None and line > 0:
            return f"loop at {resolved_file_path}:{line}"
        if node_type == 'line' and file_id is not None and line > 0:
            return f"{resolved_file_path}:{line}"
        if node_type == 'entry' and node_id == 1:
             return "<program root>"
        if node_name_id:
             if node_name_id in func_map: return func_map[node_name_id]
             return str(node_name_id)
        return f"Node_{node_id} ({node_type})"


    for node_id, node_data in nodes.items():
        path_list = []
        curr_id = node_id
        while curr_id is not None and curr_id in nodes and curr_id != 0:
            path_list.append(get_node_str(curr_id))
            parent_id = nodes[curr_id].get('parent')
            if parent_id == curr_id:
                break
            curr_id = parent_id
        paths[node_id] = " » ".join(reversed(path_list))
    return pd.Series(paths, name="cct_path")

# --- Main script ---
try:
    print(f"Opening database: {DB_PATH}\n")
    data_analyzer = hpcanalysis.open_db(DB_PATH)
    query_api = data_analyzer._query_api

    # --- Inspect internal mappings from hpcanalysis ---
    try:
        source_files_df = query_api._source_files
        print("\n--- query_api._source_files (Head) ---")
        print(source_files_df.head())
        print("\n--- query_api._source_files (Info) ---")
        source_files_df.info()

        functions_df = query_api._functions
        print("\n--- query_api._functions (Head) ---")
        print(functions_df.head())
        print("\n--- query_api._functions (Info) ---")
        functions_df.info()
    except AttributeError as e:
        print(f"Could not access _source_files or _functions from query_api: {e}")
        source_files_df = pd.DataFrame()
        functions_df = pd.DataFrame()
    print("-" * 50)

    # 1. List and verify metric descriptions
    print("--- Available Metric Descriptions (searching for GINS) ---")
    all_metric_descs = query_api.query_metric_descriptions(metrics_exp="*")
    print("Columns in all_metric_descs:", all_metric_descs.columns)
    print("Head of all_metric_descs:")
    print(all_metric_descs.head())
    
    # Filter for GINS metrics to verify names and get their IDs
    gins_metrics = all_metric_descs[
        all_metric_descs['name'].str.contains("GINS", case=False, na=False)
    ]
    print(gins_metrics)
    print("-" * 50)

    # Extract the actual metric names and their IDs that hpcanalysis uses.
    target_metric_ids = {}
    for m_name in TARGET_METRICS_DISPLAY:
        parts = m_name.split(': ')
        base_name_display = parts[0]

    # 2. Query Calling Context Tree (CCT)
    print("\n--- Querying CCT Nodes ---")
    cct_exp = "*"
    try:
        cct_df = query_api.query_cct(cct_exp=cct_exp)
        print("Columns in cct_df:", cct_df.columns)
        print("Head of cct_df:")
        print(cct_df.head())
        print("\n--- cct_df after initial query ---")
        print(cct_df.info())
        print(cct_df.head())
        
        # --- Populate and inspect internal mappings from hpcanalysis ---
        try:
            source_files_df = query_api._source_files
            print("\n--- query_api._source_files (Head) ---")
            print(source_files_df.head())
            print("\n--- query_api._source_files (Info) ---")
            source_files_df.info()

            functions_df = query_api._functions
            print("\n--- query_api._functions (Head) ---")
            print(functions_df.head())
            print("\n--- query_api._functions (Info) ---")
            functions_df.info()
        except AttributeError as e:
            print(f"Could not access _source_files or _functions from query_api after CCT query: {e}")
        print("-" * 50)

        # Create ID to Name/Path maps
        file_id_to_path_map = {}
        if not source_files_df.empty and 'file_path' in source_files_df.columns:
            file_id_to_path_map = source_files_df['file_path'].to_dict()
            print("\n--- Sample file_id_to_path_map ---")
            print(list(file_id_to_path_map.items())[:5])
        else:
            print("\n--- source_files_df is empty or missing 'file_path' column for path mapping ---")

        func_id_to_name_map = {}
        if not functions_df.empty and 'name' in functions_df.columns:
            func_id_to_name_map = functions_df['name'].to_dict()
            print("\n--- Sample func_id_to_name_map ---")
            print(list(func_id_to_name_map.items())[:5])
        else:
            print("\n--- functions_df is empty or missing 'name' column for function name mapping ---")
        print("-" * 50)
        
        # Add CCT paths
        if not cct_df.empty:
             cct_paths_series = get_cct_paths(cct_df, file_id_to_path_map, func_id_to_name_map)
             cct_df = cct_df.join(cct_paths_series)
             print("\n--- cct_df after adding cct_paths ---")
             print(cct_df.info())
             print(cct_df.reset_index()[['id', 'cct_path', 'name', 'file_path', 'line']].head())

    except Exception as e:
        print(f"Error querying CCT with '{cct_exp}': {e}")
        print("You might need a more specific CCT expression (e.g., 'function(main)', or related to 'matrix_mul.cu').")
        cct_df = pd.DataFrame()


    # 3. Query Profile Slices for base metrics
    print(f"\n--- Querying Profile Slices for all metrics (will filter for {QUERY_BASE_NAMES} later) ---")
    # First, list profile descriptions to find a suitable one.
    profile_descs = query_api.query_profile_descriptions(profiles_exp="*")
    print("Available Profile Descriptions:")
    print(profile_descs)
    
    profile_exp_to_use = "*"
    if not profile_descs.empty:
        pass

    if not cct_df.empty:
        profile_slices_df = query_api.query_profile_slices(
            profiles_exp=profile_exp_to_use, 
            cct_exp=cct_exp,
            metrics_exp="*" # Query all metrics
        )
        print(f"Query for profile_slices returned {len(profile_slices_df)} entries.")
        if len(profile_slices_df) > 0:
            print(profile_slices_df.head())
        print("-" * 50)
        print("\n--- profile_slices_df after query_profile_slices (index includes profile_id, cct_id, metric_id) ---")
        print(profile_slices_df.info())
        print(profile_slices_df.head())

        # Reset index to make profile_id, cct_id, metric_id regular columns
        if not profile_slices_df.empty:
            profile_slices_df = profile_slices_df.reset_index()
            print("\n--- profile_slices_df after reset_index() ---")
            print(profile_slices_df.info())
            print(profile_slices_df.head())

        # 3. Process Queried Data
        metric_id_to_base_name_map = all_metric_descs.set_index('id')['name'].to_dict()
        metric_id_to_scope_map = all_metric_descs.set_index('id')['scope'].to_dict()
        metric_id_to_agg_map = all_metric_descs.set_index('id')['aggregation'].to_dict()

        profile_slices_df['metric_base_name'] = profile_slices_df['metric_id'].map(metric_id_to_base_name_map)
        profile_slices_df['metric_scope'] = profile_slices_df['metric_id'].map(metric_id_to_scope_map)
        profile_slices_df['metric_aggregation'] = profile_slices_df['metric_id'].map(metric_id_to_agg_map)

        # Filter for Sum (I) variants of our query base names
        filtered_slices_mask = (
            (profile_slices_df['metric_base_name'].isin(QUERY_BASE_NAMES)) &
            (profile_slices_df['metric_aggregation'] == 'sum') &
            (profile_slices_df['metric_scope'] == 'i') # Focusing on Inclusive Sum
        )
        filtered_slices = profile_slices_df[filtered_slices_mask].copy()

        print(f"After filtering for Sum (I) variants, {len(filtered_slices)} entries remaining.")
        if len(filtered_slices) > 0:
            print(filtered_slices.head())
        print("-" * 50)
        print("\n--- filtered_slices after initial filter (Sum I) ---")
        print(filtered_slices.info())
        print(filtered_slices.head())

        # Create a display name column that matches TARGET_METRICS_DISPLAY
        def create_display_name_from_parts(row):
            base_name = row['metric_base_name']
            aggregation = row['metric_aggregation']
            scope = row['metric_scope']

            # Formatting to match hpcviewer: GINS:STL_ANY: Sum (I)
            name_parts = base_name.split(':')
            formatted_base_name = name_parts[0].upper() # GINS
            if len(name_parts) > 1:
                suffix = ":".join(name_parts[1:]) # STL_ANY, STL_IFET etc.
                formatted_base_name += ":" + suffix.upper()
            
            agg_display = aggregation.capitalize() # Sum
            scope_display = f"({scope.upper()})" # (I)
            return f"{formatted_base_name}: {agg_display} {scope_display}"

        if not filtered_slices.empty:
            filtered_slices['metric_display_name'] = filtered_slices.apply(create_display_name_from_parts, axis=1)
        else:
            filtered_slices['metric_display_name'] = pd.Series(dtype='str') 
        
        print("\n--- filtered_slices after adding metric_display_name ---")
        print(filtered_slices.info())
        print(filtered_slices[['cct_id', 'metric_id', 'metric_base_name', 'metric_display_name', 'value']].head())

        # Ensure we only proceed with metrics we want to display
        filtered_slices = filtered_slices[filtered_slices['metric_display_name'].isin(TARGET_METRICS_DISPLAY)]
        print(f"After filtering for specific TARGET_METRICS_DISPLAY, {len(filtered_slices)} entries remaining.")
        if len(filtered_slices) > 0:
            print(filtered_slices[['cct_id', 'metric_display_name', 'value']].head())
        print("-" * 50)
        print("\n--- filtered_slices after TARGET_METRICS_DISPLAY filter ---")
        print(filtered_slices.info())
        print(filtered_slices.head())


        # Aggregate values if multiple profiles were fetched by profile_exp="*"
        if profile_exp_to_use == "*":
            aggregated_values = filtered_slices.groupby(['cct_id', 'metric_display_name'])['value'].sum().reset_index()
        else:
            aggregated_values = filtered_slices # Use as is if a specific profile was queried
        
        print("\n--- aggregated_values ---")
        print(aggregated_values.info())
        print(aggregated_values.head())

        # Pivot the table: CCT nodes as rows, metrics as columns
        final_df = aggregated_values.pivot_table(
            index='cct_id', 
            columns='metric_display_name', # Use the new display name for columns
            values='value'
        ).reset_index()

        # Join with CCT information (path, name, etc.)
        # Ensure cct_df has 'id' as the CCT node identifier
        if 'cct_path' in cct_df.columns: # Check if join was successful by presence of cct_path
            # For merging, reset index if cct_df's ID is in the index
            cct_df_for_merge = cct_df.reset_index()
            # The index name is 'id' as seen in the output, so reset_index() makes it a column.
            final_df = pd.merge(cct_df_for_merge[['id', 'cct_path', 'name', 'file_path', 'line', 'type']], final_df, left_on='id', right_on='cct_id', how='inner')
            # Select and order columns for display
            final_df.rename(columns={'cct_path': 'Scope'}, inplace=True)
            
            # Add a 'File' column with resolved file paths
            if 'file_path' in final_df.columns: # file_path here is the ID
                final_df['File'] = final_df['file_path'].map(file_id_to_path_map).fillna('')
            else:
                final_df['File'] = ''

            print("\n--- final_df after merge, Scope rename, and File column addition ---")
            print(final_df.info())
            print(final_df.head())

            # Ensure all TARGET_METRICS_DISPLAY are present as columns, fill missing with NaN if necessary
            for col_name in TARGET_METRICS_DISPLAY:
                if col_name not in final_df.columns:
                    final_df[col_name] = pd.NA # Or use 0 or np.nan if preferred

            ordered_value_cols = [col for col in TARGET_METRICS_DISPLAY if col in final_df.columns]
            base_cct_cols = ['Scope', 'File', 'line'] # UPDATED to include File and line
            display_columns_abs = base_cct_cols + ordered_value_cols

            print("\n--- Combined PC Sampling Data View (Absolute Values) ---")
            # Sort by the Scope column for top-down view
            sort_column = 'Scope' if 'Scope' in final_df.columns else None # New sort
            if sort_column:
                final_df_sorted = final_df.sort_values(by=sort_column, ascending=True) # Ascending for hierarchical paths
            else:
                final_df_sorted = final_df
            
            print("\n--- final_df_sorted before dropna and percentage calc ---")
            print(final_df_sorted.info())
            print(final_df_sorted.head())
            
            # Drop rows where all target metric values are NA (if any were added and not filled)
            final_df_sorted = final_df_sorted.dropna(subset=ordered_value_cols, how='all')
            print(final_df_sorted[display_columns_abs])
            print("-" * 50)
            print("\n--- final_df_sorted after dropna, before percentage calc ---")
            print(final_df_sorted.info())
            print(final_df_sorted.head())

            # 5. Calculate Percentages
            # Find the aggregate row (e.g., cct_id == 1, or where cct_path is <program root> or similar)
            # Using cct_id = 1 as an assumption for the aggregate row based on prior output observation
            aggregate_row_cct_id = 1 
            if not cct_df_for_merge[cct_df_for_merge['id'] == aggregate_row_cct_id].empty:
                aggregate_values = final_df_sorted[final_df_sorted['id'] == aggregate_row_cct_id][ordered_value_cols].iloc[0]
                percentage_cols = []
                for col in ordered_value_cols:
                    percent_col_name = f"{col} %"
                    percentage_cols.append(percent_col_name)
                    # Ensure baseline is not zero to avoid division by zero
                    baseline = aggregate_values[col]
                    if pd.isna(baseline) or baseline == 0:
                        final_df_sorted[percent_col_name] = 0.0 # Or pd.NA
                    else:
                        final_df_sorted[percent_col_name] = (final_df_sorted[col] / baseline) * 100
                
                print("\n--- Combined PC Sampling Data View (With Percentages) ---")
                display_columns_all = base_cct_cols[:] # Create a copy to extend
                for val_col in ordered_value_cols:
                    display_columns_all.append(val_col)
                    if f"{val_col} %" in final_df_sorted.columns:
                         display_columns_all.append(f"{val_col} %")
                print("\n--- final_df_sorted after percentage calculation (Sorted by Scope) ---")
                print(final_df_sorted.info())
                print(final_df_sorted.head())
                print(final_df_sorted[display_columns_all])            
            else:
                print(f"Could not find aggregate row with cct_id {aggregate_row_cct_id} for percentage calculation.")

        else:
            print("CCT DataFrame does not have 'cct_path' column, join likely failed or cct_df was empty.")

    else:
        if cct_df.empty:
            print("CCT DataFrame is empty, cannot proceed to query profile slices effectively.")
        if not TARGET_METRICS_DISPLAY:
            print("No target metrics specified or found, cannot query profile slices.")


except FileNotFoundError:
    print(f"ERROR: Database not found at {DB_PATH}")
except ImportError as e:
    print(f"ERROR: hpcanalysis or pandas might not be installed correctly. {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
