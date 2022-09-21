import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Create linkage between names in Sahin and BEA
    sahin_names = ['accom', 'arts', 'cons', 'dur', 'educ', 
                 'finance', 'govt', 'health', 'info', 'mining', 
                 'nondur', 'other', 'profbusserv', 'realestate', 
                 'retail', 'trans', 'wholesale']
    bea_names = ['Accommodation and food services', 'Arts, entertainment, and recreation', 
                'Construction', 'Durable goods', 'Educational services',
                'Finance and insurance', 'Government', 'Health care and social assistance',
                'Information', 'Mining', 'Nondurable goods', 'Other services, except government', 
                'Professional and business services',  
                'Real estate and rental and leasing', 'Retail trade', 
                'Transportation and warehousing', 'Wholesale trade']
    merge_map = pd.DataFrame({'BEA_sector': bea_names, 'Sahin_sector': sahin_names})
    # Read in NAICS codes
    codes = pd.read_excel("data/raw/IO_table_07_12.xlsx", sheet_name='NAICS Codes', skiprows=4)
    codes.drop(codes.tail(6).index, inplace=True) # Dropping documentation rows
    codes.rename(columns={'Summary':'BEA_sector', 'Sector':'code', 'Detail':'6-digit_code'}, inplace=True)
    # Create code map
    sector_code_key = codes[['code', 'BEA_sector']].dropna()
    sector_code_key['BEA_sector'] = sector_code_key['BEA_sector'].str.lower().str.capitalize()
    merge_map = pd.merge(left=merge_map, right=sector_code_key, on='BEA_sector', how='left').set_index('Sahin_sector')
    # Profbusserv has multiple two digit codes
    merge_map.loc[['trans', 'accom', 'profbusserv'], 'code'] = ['48TW', '72', ['54', '55', '56']]
    merge_map = merge_map.explode('code').astype(str).reset_index()
    # Create map between 2-digit and 6-digit codes
    codes['code'] = codes['code'].ffill()
    codes = codes.astype(str)
    codes = codes[['code', '6-digit_code']].dropna()
    # pick out 6-digit codes and 5-digit codes(for local governments only)
    codes = codes.loc[(codes['6-digit_code'].str.len() == 6) | (codes['6-digit_code'].str.len() == 5)]

    # Merge together 2-digit code, 6-digit code, and names
    merge_map = pd.merge(merge_map, codes, on='code', how='left')
    # Read in input-output table
    IO_tab = pd.read_excel("data/raw/IO_table_07_12.xlsx", sheet_name='2007', skiprows=5).drop(columns='Commodity Description').rename(columns={'Code':'Input Sector(6-digit)'})
    IO_tab = pd.melt(IO_tab, id_vars='Input Sector(6-digit)', 
                     var_name='6-digit_code', value_name='Input Usage(6-digit)')
    IO_tab.loc[:, ['Input Sector(6-digit)', '6-digit_code']] = IO_tab[['Input Sector(6-digit)', '6-digit_code']].astype(str)
    PC = IO_tab[(IO_tab['6-digit_code'] == 'T001') | (IO_tab['6-digit_code'] == 'T019')] # This corresponds to personal consumption expenditures
    IO_tab = pd.merge(merge_map[['code', '6-digit_code', 'BEA_sector']], IO_tab, on='6-digit_code', how='left')
    PC = PC.pivot(index='Input Sector(6-digit)', columns='6-digit_code', values='Input Usage(6-digit)').reset_index()
    PC = pd.merge(merge_map[['code', '6-digit_code', 'BEA_sector']].rename(columns={'6-digit_code':'Input Sector(6-digit)', 
                                                                                     'code':'Input Sector(2-digit)'}), 
                  PC, on='Input Sector(6-digit)', how='left')
    # Compute total output of commodities that is not a part of intermediate use
    PC.loc[:, 'Final User'] = PC['T019'] - PC['T001']
    PC = PC.groupby(['Input Sector(2-digit)'])['Final User'].sum()/1000
    PC = PC.reset_index()
    PC = pd.merge(PC, merge_map[['code', 'BEA_sector']].rename(columns={'code':'Input Sector(2-digit)', 
                                                                          'BEA_sector': 'Input Sector'}).drop_duplicates(),
                   on='Input Sector(2-digit)', how='left')
    PC = PC.groupby(['Input Sector'])['Final User'].sum().reset_index()

    # Add in labor usage
    merge_map = merge_map.append({'Sahin_sector':'labor', 'BEA_sector':'Labor',
                                  'code':'V0', '6-digit_code':'V00100'}, ignore_index=True)

    IO_tab = pd.merge(merge_map[['code', '6-digit_code']].rename(columns={'6-digit_code':'Input Sector(6-digit)', 
                                                                          'code':'Input Sector(2-digit)'}), 
                      IO_tab, on='Input Sector(6-digit)', how='left')
    # Summing from 6-digit sectors to 2-digit sectors
    IO_tab = IO_tab.groupby(['code', 'Input Sector(2-digit)'])['Input Usage(6-digit)'].sum()/1000
    IO_tab = IO_tab.reset_index().rename(columns={'Input Usage(6-digit)': 'Input Usage'})
    # Read in sectoral output
    output_by_sector = pd.read_csv("data/raw/Gross_Output_Sector.csv", skiprows=4).drop(columns='Line', index=0).rename(columns={'Unnamed: 1':'BEA_sector'})
    output_by_sector.dropna(inplace=True)
    output_by_sector['BEA_sector'] = output_by_sector['BEA_sector'].str.strip()
    output_by_sector = output_by_sector[output_by_sector['BEA_sector'].isin(bea_names)]
    output_by_sector = pd.melt(output_by_sector, id_vars='BEA_sector', var_name='year', value_name='output')
    output_by_sector = pd.merge(output_by_sector, merge_map[['code', 'BEA_sector']].drop_duplicates(), on='BEA_sector', how='inner')
    # Pick 2007 to be consistent with input-output data
    output_by_sector = output_by_sector.loc[output_by_sector.year == '2007'].drop(columns='year')
    # Compute input shares
    A = pd.merge(IO_tab, merge_map[['code', 'BEA_sector']].drop_duplicates(), on='code')
    A = pd.merge(A, merge_map[['code', 'BEA_sector']].rename(columns={'code':'Input Sector(2-digit)', 
                                                                      'BEA_sector': 'Input Sector'}).drop_duplicates(),
                on='Input Sector(2-digit)', how='left')
    A = A.groupby(["BEA_sector", "Input Sector"]).agg({'Input Usage':'sum'}).reset_index()
    # Calculate input shares
    A.loc[:, 'share'] = A['Input Usage'] / A.groupby("BEA_sector")["Input Usage"].transform("sum")
    A.sort_values('BEA_sector', inplace=True)
    # Separate out labor shares
    Labor = A.loc[A['Input Sector'] == 'Labor']
    Labor[['BEA_sector', 'share']].to_csv('data/clean/labor_share.csv', index=False)
    A = A.loc[A['Input Sector'] != 'Labor']
    # Sorting column names to make sure in correct order
    A.sort_values('Input Sector', inplace=True)
    A = A.pivot(index='BEA_sector', columns='Input Sector', values='share')
    A.to_csv('data/clean/A.csv')    
    # Read in GDP data
    GDP = pd.read_csv("data/raw/GDP.csv", skiprows=4).drop(columns='Line', index=0).set_index('Unnamed: 1')
    GDP = pd.melt(GDP.loc[['        Gross domestic product']], var_name='year', value_name='GDP').set_index('year')
    # Compute output share for each sector in 2007
    output_by_sector['γ'] = output_by_sector['output']/GDP.loc['2007', 'GDP']
    γ = output_by_sector[['BEA_sector', 'γ']].drop_duplicates().set_index('BEA_sector')
    γ.sort_values('BEA_sector', inplace=True)
    # Compute preference parameters
    θ = np.matmul(np.identity(len(A.index)) - A.T, np.asarray(γ))
    PC = PC.set_index('Input Sector')
    θ_alt = PC['Final User']/PC['Final User'].sum()
    # Compute λ
    λ = np.matmul(θ.T, np.linalg.inv(np.identity(len(A.index)) - A))
    λ_alt = np.matmul(θ_alt.T, np.linalg.inv(np.identity(len(A.index)) - A))

    params = γ
    params.loc[:, 'θ'] = θ
    params.loc[:, 'θ_alt'] = θ_alt
    params.loc[:, 'λ'] = list(λ.iloc[0, :])
    params.loc[:, 'λ_alt'] = λ_alt
    params.loc[:, 'α'] = list(Labor['share'])
    # Read in φ from Sahin et al estimates
    phi = pd.read_stata("data/raw/phi_breakpoint_v12_hires.dta").rename(columns={'ind':'Sahin_sector', 'phi':'φ'})
    phi = pd.merge(merge_map[['Sahin_sector', 'BEA_sector']].drop_duplicates(), phi, on='Sahin_sector').drop(columns='Sahin_sector')
    params = params.join(phi.set_index('BEA_sector'))
    params.to_csv('data/clean/params.csv')
    employment = pd.read_csv('data/raw/EstabSurv_Employ_ind.csv')
    employment = pd.melt(employment, id_vars='year', var_name='Sahin_sector', value_name='e')
    employment['Sahin_sector'] = employment['Sahin_sector'].str[1:]
    employment = pd.merge(merge_map[['Sahin_sector', 'BEA_sector']].drop_duplicates(), employment, on='Sahin_sector').drop(columns='Sahin_sector')
    employment['date'] = pd.to_datetime(employment['year'], format='%b-%Y')
    employment.drop(columns='year', inplace=True)
    # Read in JOLTs data including sector-specific unemployment and vacancies. 
    JOLTs = pd.read_stata("data/raw/jolts_industry_fulldata.dta")[['date', 'ind', 'year', 'v', 'u']].rename(columns={'ind':'Sahin_sector'})
    JOLTs = pd.merge(merge_map[['Sahin_sector', 'BEA_sector']].drop_duplicates(), JOLTs, on='Sahin_sector').drop(columns='Sahin_sector')
    JOLTs['date'] = pd.to_datetime(JOLTs['date'], format='%Y%m')
    JOLTs = pd.merge(JOLTs, employment, on=['BEA_sector', 'date'], how='left')
    # Compute aggregate unemployment monthly and annually. 
    # Also aggregate monthly observations into annual observations. 
    U_monthly = JOLTs.groupby('date')['u'].sum().reset_index().rename(columns={'u': 'U_monthly'})
    JOLTs_monthly = pd.merge(JOLTs[['BEA_sector', 'date', 'v', 'u', 'e']], U_monthly, on='date')
    JOLTs_monthly.to_csv('data/clean/labor_market_monthly.csv', index=False)
    JOLTs_yearly = JOLTs[JOLTs['year'] != 2000]
    JOLTs_yearly = JOLTs_yearly.groupby(['year', 'BEA_sector']).agg({'v':'sum', 'u':'sum', 'e':'sum'}).reset_index()
    U_yearly = JOLTs_yearly.groupby('year')['u'].sum().reset_index().rename(columns={'u': 'U_yearly'})
    JOLTs_yearly = pd.merge(JOLTs_yearly, U_yearly, on='year')
    JOLTs_yearly.to_csv('data/clean/labor_market_yearly.csv', index=False)