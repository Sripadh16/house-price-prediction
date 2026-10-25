import numpy as np
import pandas as pd

def generate_house_dataset(num_samples=2048, random_state=42):
    """
    Generate a realistic, correlated synthetic housing dataset
    compatible with the Ames Housing benchmark and course project requirements.
    """
    np.random.seed(random_state)
    
    # Neighborhoods with associated base price premiums
    neighborhoods = [
        ('NAmes', 0), ('CollgCr', 15000), ('OldTown', -12000), 
        ('Edwards', -10000), ('Sawyer', -5000), ('NridgHt', 65000), 
        ('Timber', 25000), ('Gilbert', 12000), ('Somerst', 30000), 
        ('StoneBr', 60000)
    ]
    neigh_names = [n[0] for n in neighborhoods]
    neigh_weights = [0.18, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.08, 0.07, 0.06]
    neigh_weights = np.array(neigh_weights) / sum(neigh_weights)
    
    selected_neighs = np.random.choice(neigh_names, size=num_samples, p=neigh_weights)
    neigh_dict = dict(neighborhoods)
    
    # House styles
    house_styles = ['1Story', '2Story', '1.5Fin', 'SFoyer', 'SLvl']
    style_weights = [0.45, 0.35, 0.08, 0.06, 0.06]
    selected_styles = np.random.choice(house_styles, size=num_samples, p=style_weights)
    
    # Exterior qualities
    exter_quals = ['Ex', 'Gd', 'TA', 'Fa']
    exter_weights = [0.08, 0.35, 0.52, 0.05]
    exter_premiums = {'Ex': 45000, 'Gd': 18000, 'TA': 0, 'Fa': -15000}
    selected_exters = np.random.choice(exter_quals, size=num_samples, p=exter_weights)
    
    # Year built & remod
    year_built = np.random.randint(1920, 2023, size=num_samples)
    year_remod = np.array([
        np.random.randint(yb, 2024) if np.random.rand() > 0.4 else yb 
        for yb in year_built
    ])
    
    # Overall Quality (1-10) and Condition (1-9) correlated with YearBuilt
    qual_noise = np.random.normal(0, 1.2, size=num_samples)
    overall_qual = np.clip(np.round(3 + (year_built - 1920) / 102 * 5 + qual_noise), 1, 10).astype(int)
    
    cond_noise = np.random.normal(0, 1.0, size=num_samples)
    overall_cond = np.clip(np.round(5 + cond_noise), 1, 9).astype(int)
    
    # Lot Area (skewed log-normal)
    lot_area = np.round(np.random.lognormal(mean=9.1, sigma=0.5, size=num_samples)).astype(int)
    lot_area = np.clip(lot_area, 1800, 45000)
    
    # Total Basement SF
    has_bsmt = np.random.rand(num_samples) > 0.04
    bsmt_sf = np.where(
        has_bsmt,
        np.round(np.random.normal(1050, 350, size=num_samples)),
        0
    )
    bsmt_sf = np.clip(bsmt_sf, 0, 3200).astype(int)
    
    # 1st Floor SF
    first_flr_sf = np.round(np.random.normal(1150, 380, size=num_samples))
    first_flr_sf = np.clip(first_flr_sf, 450, 3000).astype(int)
    
    # 2nd Floor SF based on House Style
    second_flr_sf = np.zeros(num_samples, dtype=int)
    for i, style in enumerate(selected_styles):
        if style in ['2Story', '1.5Fin']:
            second_flr_sf[i] = int(np.clip(np.random.normal(first_flr_sf[i] * 0.85, 200), 250, 2200))
        elif style == 'SLvl':
            second_flr_sf[i] = int(np.clip(np.random.normal(350, 100), 100, 700))
    
    gr_liv_area = first_flr_sf + second_flr_sf
    
    # Bedrooms above grade
    bedrooms = np.clip(np.round(gr_liv_area / 450 + np.random.normal(0, 0.6, size=num_samples)), 1, 6).astype(int)
    
    # Bathrooms
    full_bath = np.clip(np.round(1 + (gr_liv_area > 1300).astype(int) + (gr_liv_area > 2200).astype(int) + np.random.choice([0, 1], p=[0.75, 0.25], size=num_samples)), 1, 4).astype(int)
    half_bath = np.where(selected_styles == '2Story', np.random.choice([0, 1], p=[0.3, 0.7], size=num_samples), np.random.choice([0, 1], p=[0.8, 0.2], size=num_samples)).astype(int)
    
    # Garage
    garage_cars = np.clip(np.round(overall_qual / 3.5 + np.random.normal(0, 0.7, size=num_samples)), 0, 4).astype(int)
    garage_area = np.where(garage_cars > 0, np.round(garage_cars * 240 + np.random.normal(0, 60, size=num_samples)), 0).astype(int)
    garage_area = np.clip(garage_area, 0, 1400)
    
    # Fireplaces
    fireplaces = np.clip(np.round((overall_qual >= 7).astype(int) + np.random.choice([0, 1, 2], p=[0.55, 0.35, 0.10], size=num_samples)), 0, 3).astype(int)
    
    # Sale Price generation with realistic real-estate valuation formula
    base_price = 35000
    price_liv = gr_liv_area * 78
    price_bsmt = bsmt_sf * 42
    price_qual = (overall_qual ** 1.65) * 4200
    price_cond = (overall_cond - 5) * 3200
    price_age = (year_built - 1920) * 320 + (year_remod - year_built) * 200
    price_garage = garage_cars * 9500 + garage_area * 18
    price_bath = full_bath * 11000 + half_bath * 5500
    price_fire = fireplaces * 6500
    price_lot = np.sqrt(lot_area) * 110
    price_neigh = np.array([neigh_dict[n] for n in selected_neighs])
    price_exter = np.array([exter_premiums[e] for e in selected_exters])
    
    # Realistic market noise (~8%)
    price_noise = np.random.normal(1.0, 0.08, size=num_samples)
    
    raw_sale_price = (
        base_price + price_liv + price_bsmt + price_qual + price_cond +
        price_age + price_garage + price_bath + price_fire + price_lot +
        price_neigh + price_exter
    ) * price_noise
    
    sale_price = np.round(np.clip(raw_sale_price, 55000, 750000), -2).astype(int)
    
    # Add small fraction of realistic outliers (luxury mansions / extreme lots)
    outlier_idx = np.random.choice(num_samples, size=int(num_samples * 0.015), replace=False)
    for idx in outlier_idx:
        sale_price[idx] = int(sale_price[idx] * np.random.uniform(1.35, 1.65))
        
    df = pd.DataFrame({
        'LotArea': lot_area,
        'OverallQual': overall_qual,
        'OverallCond': overall_cond,
        'YearBuilt': year_built,
        'YearRemodAdd': year_remod,
        'GrLivArea': gr_liv_area,
        '1stFlrSF': first_flr_sf,
        '2ndFlrSF': second_flr_sf,
        'BedroomAbvGr': bedrooms,
        'FullBath': full_bath,
        'HalfBath': half_bath,
        'GarageCars': garage_cars,
        'GarageArea': garage_area,
        'TotalBsmtSF': bsmt_sf,
        'Fireplaces': fireplaces,
        'Neighborhood': selected_neighs,
        'HouseStyle': selected_styles,
        'ExterQual': selected_exters,
        'SalePrice': sale_price
    })
    
    return df

if __name__ == '__main__':
    df = generate_house_dataset(num_samples=2048)
    filename = 'house_dataset_big.csv'
    df.to_csv(filename, index=False)
    print(f"Generated {len(df)} samples into '{filename}'.")
    print(df.head())
    print("\nDataset Summary:")
    print(df.describe())
