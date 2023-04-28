import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Create linkage between names in Sahin and BEA
    short_names = ['mining', 'trans', 'const', 'dur', 'nondur',
                    'trade', 'info', 'fin','profserv', 'edhealth',
                    'accom', 'other', 'gov']
    sector_names = ['Mining', 'Transportation and Utilities', 'Construction',
                    'Durable goods', 'Nondurable good', 'Wholesale and Retail trade',
                    'Information', 'Financial Activities', 'Professional and business services',
                    'Education and Health Services', 'Leisure  and Hospitality',
                    'Other services, except government','Government']
    sector_codes = ['21', ['22', '48TW'] ,'23', '33DG', ['31ND','11'], ['42', '44RT'],
                     '51', ['52', '53'], ['54','55','56'],['61','62'], ['71','72'],
                     '81','G']
    merge_map = pd.DataFrame({'BEA_sector': sector_names, 'short_names': short_names,'code':sector_codes})
    merge_map = merge_map.explode('code').astype(str).reset_index()

    # Read in NAICS codes
    codes = pd.read_excel("data/update_raw/IO_table_07_12_405industry.xlsx", sheet_name='NAICS Codes', skiprows=4)
    codes.drop(codes.tail(6).index, inplace=True) # Dropping documentation rows
    codes.rename(columns={'Summary':'sector_names', 'Sector':'code', 'Detail':'6-digit_code'}, inplace=True)
    # Create code map

    # Create map between 2-digit and 3-digit codes
    codes['code'] = codes['code'].ffill()
    codes = codes.astype(str)
    codes = codes[['code', 'sector_names']].dropna()
    # pick out 3-digit codes
    codes = codes.loc[(codes['sector_names'].str.len() == 6) | (codes['sector_names'].str.len() == 5)
                      | (codes['sector_names'].str.len() == 4)| (codes['sector_names'].str.len() == 3)
                      | (codes['sector_names'].str.len() == 2)]
    codes = codes.loc[(codes['sector_names']!='nan') & (codes['sector_names']!='MINING') 
                      & (codes['sector_names']!='Used') &  (codes['sector_names']!='Other')]
    codes.rename(columns={'sector_names':'3-digit_code'}, inplace=True)

    # Merge together 2-digit code, 3-digit code, and names
    merge_map = pd.merge(merge_map, codes, on='code', how='left')
    # Read in input-output table
    Use_tab = pd.read_excel("data/update_raw/IO_Use_table_2021_71industry.xlsx", sheet_name='Table', skiprows=5)

    # Demand elasticities
    demand_tab = pd.DataFrame({'3-digit_code':Use_tab['3-digit_code'][:71],'intermediate_uses':Use_tab['T001'][:71],
                               'final_uses':Use_tab['T019'][:71]})
    demand_tab = demand_tab.merge(merge_map[['3-digit_code','short_names']], how='left', on='3-digit_code')
    demand_tab = demand_tab.groupby('short_names').sum()
    demand_tab['consumption'] = demand_tab['final_uses'] - demand_tab['intermediate_uses']
    demand_tab['demand_elasticity'] = demand_tab['consumption']/np.sum(demand_tab['consumption'])

    # Labor elasticities 
    labor_tab = pd.DataFrame({'3-digit_code':Use_tab['3-digit_code'][:71], 'total_output':np.array(Use_tab.iloc[79,1:72]),
                               'total_intermediate':np.array(Use_tab.iloc[73,1:72])})
    labor_tab = labor_tab.merge(merge_map[['3-digit_code','short_names']],how='left',on='3-digit_code')
    labor_tab = labor_tab[['total_output','total_intermediate','short_names']].groupby('short_names').sum()
    labor_tab['labor_income'] = labor_tab['total_output']-labor_tab['total_intermediate']
    labor_tab['labor_elasticity'] = labor_tab['labor_income']/labor_tab['total_output']


    # Intermediate input elasticities
    U = pd.DataFrame(data = np.array(Use_tab.iloc[:71,1:72]),
                     index=Use_tab['3-digit_code'][:71],
                     columns=Use_tab['3-digit_code'][:71]) 
    U = U.transpose()
    U = U.merge(merge_map[['3-digit_code','short_names']],on='3-digit_code')
    U = U.groupby('short_names').sum()
    U = U.transpose()
    U.index.name = '3-digit_code' 
    U = U.merge(merge_map[['3-digit_code','short_names']],on='3-digit_code')
    U = U.groupby('short_names').sum()
    U = U.divide(labor_tab.total_output,axis=1)

    Make_tab = pd.read_excel("data/update_raw/IO_Make_table_2021_71industry.xlsx", sheet_name='Table', skiprows=5)
    output_tab = pd.DataFrame({'3-digit_code':Make_tab['3-digit_code'][:71],
                               'total_industry_output':np.array(Make_tab['Total Industry Output'][:71])})
    output_tab = output_tab.merge(merge_map[['3-digit_code', 'short_names']],on='3-digit_code')
    output_tab = output_tab.groupby('short_names').sum()

    M = pd.DataFrame(data=np.array(Make_tab.iloc[:71,1:72]),
                     index=Make_tab['3-digit_code'][:71],
                     columns=Make_tab['3-digit_code'][:71])
    M = M.transpose()
    M = M.merge(merge_map[['3-digit_code', 'short_names']],on='3-digit_code')
    M = M.groupby('short_names').sum()
    M = M.transpose()
    M.index.name = '3-digit_code'
    M = M.merge(merge_map[['3-digit_code', 'short_names']],on='3-digit_code')
    M = M.groupby('short_names').sum()
    M = M.divide(output_tab.total_industry_output,axis=0)
    
    A = pd.DataFrame(data = np.array(np.array(M) @ np.array(U)).T, index=M.index, columns=M.index)
    adjustment = np.sum(np.array(A),axis=1) + np.array(labor_tab.labor_elasticity) #minor adjusments to ensure sum to 1, these should be small
    A = A.divide(adjustment, axis=0)
    labor_tab.labor_elasticity = labor_tab.labor_elasticity.divide(adjustment)
    
    #writing to csv
    A.to_csv('data/clean/A.csv')
    labor_tab.to_csv('data/clean/labor_tab.csv')
    demand_tab.to_csv('data/clean/demand_tab.csv')


    

    

