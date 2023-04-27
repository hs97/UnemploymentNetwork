import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Create linkage between names in Sahin and BEA
    short_names = ['accom', 'arts', 'cons', 'dur', 'educ', 
                 'finance', 'govt', 'health', 'info', 'mining', 
                 'nondur', 'other', 'profbusserv', 'realestate', 
                 'retail', 'trans', 'wholesale']
    sector_names = ['Accommodation and food services', 'Arts, entertainment, and recreation', 
                'Construction', 'Durable goods', 'Educational services',
                'Finance and insurance', 'Government', 'Health care and social assistance',
                'Information', 'Mining', 'Nondurable goods', 'Other services, except government', 
                'Professional and business services',  
                'Real estate and rental and leasing', 'Retail trade', 
                'Transportation and warehousing', 'Wholesale trade']
    merge_map = pd.DataFrame({'BEA_sector': sector_names, 'Sahin_sector': short_names})
    # Read in NAICS codes
    codes = pd.read_excel("data/update_raw/IO_table_07_12_405industry.xlsx", sheet_name='NAICS Codes', skiprows=4)
    codes.drop(codes.tail(6).index, inplace=True) # Dropping documentation rows
    codes.rename(columns={'Summary':'BEA_sector', 'Sector':'code', 'Detail':'6-digit_code'}, inplace=True)
    # Create code map
    sector_code_key = codes[['code', 'BEA_sector']].dropna()
    sector_code_key['BEA_sector'] = sector_code_key['BEA_sector'].str.lower().str.capitalize()
    merge_map = pd.merge(left=merge_map, right=sector_code_key, on='BEA_sector', how='left').set_index('Sahin_sector')
    # Profbusserv has multiple two digit codes
    merge_map.loc[['trans', 'accom', 'profbusserv'], 'code'] = ['48TW', '72', ['54', '55', '56']]
    merge_map = merge_map.explode('code').astype(str).reset_index()
    # Create map between 2-digit and 3-digit codes
    codes['code'] = codes['code'].ffill()
    codes = codes.astype(str)
    codes = codes[['code', 'BEA_sector']].dropna()
    # pick out 6-digit codes and 5-digit codes(for local governments only)
    codes = codes.loc[(codes['BEA_sector'].str.len() == 6) | (codes['BEA_sector'].str.len() == 5)
                      | (codes['BEA_sector'].str.len() == 4)| (codes['BEA_sector'].str.len() == 3)
                      | (codes['BEA_sector'].str.len() == 2)]
    codes = codes.loc[(codes['BEA_sector']!='nan') & (codes['BEA_sector']!='MINING') 
                      & (codes['BEA_sector']!='Used') &  (codes['BEA_sector']!='Other')]
    codes.rename(columns={'BEA_sector':'3-digit_code'}, inplace=True)

    # Merge together 2-digit code, 3-digit code, and names
    merge_map = pd.merge(merge_map, codes, on='code', how='left')
    # Read in input-output table
    Use_tab = pd.read_excel("data/update_raw/IO_Use_table_2021_71industry.xlsx", sheet_name='Table', skiprows=5)
    Make_tab = pd.read_excel("data/update_raw/IO_Make_table_2021_71industry.xlsx", sheet_name='Table', skiprows=5)
    
    Use_tab = pd.melt(Use_tab, id_vars='Input Commodity(3-digit)', 
                    var_name='3-digit_code', value_name='Input Usage(3-digit)')
    Make_tab = pd.melt(Make_tab, id_vars='Sector(3-digit)', 
                    var_name='3-digit-code', value_name='Commodity Production(3-digit)')
    Use_tab = pd.merge(merge_map[['code', '3-digit_code', 'BEA_sector']], Use_tab, on='3-digit_code', how='left')
    

    Use_tab.loc[:, ['Input Sector(3-digit)', '3-digit_code']] = Use_tab[['Input Sector(3-digit)', '3-digit_code']].astype(str)
