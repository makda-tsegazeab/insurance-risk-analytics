# hypothesis_testing.py
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class InsuranceHypothesisTester:
    def __init__(self, data_path, delimiter='|'):
        """Initialize with data path."""
        print(f"Loading data from {data_path}...")
        
        # Load with pipe delimiter
        self.df = pd.read_csv(data_path, delimiter=delimiter, low_memory=False)
        print(f"Data loaded. Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Clean column names (remove trailing/leading spaces)
        self.df.columns = self.df.columns.str.strip()
        
        # Check if TotalClaims exists
        if 'TotalClaims' not in self.df.columns:
            # Try to find similar column
            claims_cols = [col for col in self.df.columns if 'claim' in col.lower() or 'Claim' in col]
            if claims_cols:
                print(f"Using '{claims_cols[0]}' as TotalClaims column")
                self.df['TotalClaims'] = self.df[claims_cols[0]]
            else:
                print("ERROR: No claims column found!")
                return
        
        # Calculate key metrics
        self.df['Claim_Frequency'] = (self.df['TotalClaims'] > 0).astype(int)
        self.df['Claim_Severity'] = self.df['TotalClaims'].where(self.df['TotalClaims'] > 0, np.nan)
        
        if 'TotalPremium' in self.df.columns:
            self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']
        else:
            print("WARNING: TotalPremium column not found")
            self.df['Margin'] = np.nan
    
    def test_province_risk(self):
        """H₀: There are no risk differences across provinces."""
        print("\n" + "="*60)
        print("TEST 1: Risk Differences Across Provinces")
        print("="*60)
        
        # Check for province column (might be 'Province' or similar)
        province_col = None
        for col in ['Province', 'PROVINCE', 'province', 'State', 'STATE']:
            if col in self.df.columns:
                province_col = col
                break
        
        if not province_col:
            print("Error: No province/state column found in data")
            print(f"Available columns: {[c for c in self.df.columns if 'prov' in c.lower() or 'state' in c.lower()]}")
            return
        
        print(f"Using column '{province_col}' for province analysis")
        
        # Clean province data
        self.df[province_col] = self.df[province_col].astype(str).str.strip()
        
        provinces = self.df[province_col].dropna().unique()
        print(f"Found {len(provinces)} provinces: {list(provinces)[:10]}...")  # Show first 10
        
        if len(provinces) < 2:
            print("Need at least 2 provinces for comparison")
            return
        
        # Claim Frequency by Province (Chi-square test)
        contingency = pd.crosstab(self.df[province_col], self.df['Claim_Frequency'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        print(f"\nClaim Frequency Test (Chi-square):")
        print(f"  Chi2 Statistic: {chi2:.4f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Result: {'REJECT H₀' if p_value < 0.05 else 'FAIL TO REJECT H₀'}")
        
        # Claim Severity by Province (ANOVA) - only for provinces with enough data
        severity_by_province = []
        province_names = []
        for p in provinces:
            severity_data = self.df[self.df[province_col] == p]['Claim_Severity'].dropna()
            if len(severity_data) > 10:  # Minimum sample size
                severity_by_province.append(severity_data)
                province_names.append(p)
        
        if len(severity_by_province) >= 2:
            f_stat, p_value_anova = stats.f_oneway(*severity_by_province)
            print(f"\nClaim Severity Test (ANOVA for {len(province_names)} provinces with enough data):")
            print(f"  F Statistic: {f_stat:.4f}")
            print(f"  P-value: {p_value_anova:.6f}")
            print(f"  Result: {'REJECT H₀' if p_value_anova < 0.05 else 'FAIL TO REJECT H₀'}")
        
        # Calculate actual differences
        print("\nRisk Metrics by Province (top 10 by count):")
        province_stats = self.df.groupby(province_col).agg({
            'Claim_Frequency': ['mean', 'count'],
            'Claim_Severity': 'mean',
            'TotalPremium': 'mean',
            'TotalClaims': 'mean'
        }).round(2)
        
        # Sort by count and show top 10
        province_stats = province_stats.sort_values(('Claim_Frequency', 'count'), ascending=False)
        print(province_stats.head(10))
        
        return p_value
    
    def test_gender_risk(self):
        """H₀: There is no significant risk difference between Women and Men."""
        print("\n" + "="*60)
        print("TEST 4: Risk Differences Between Women and Men")
        print("="*60)
        
        # Check for gender column
        gender_col = None
        for col in ['Gender', 'GENDER', 'gender', 'Sex', 'SEX', 'Title']:
            if col in self.df.columns:
                gender_col = col
                break
        
        if not gender_col:
            print("Error: No gender column found in data")
            return
        
        print(f"Using column '{gender_col}' for gender analysis")
        
        # Clean and map gender values
        self.df[gender_col] = self.df[gender_col].astype(str).str.strip()
        
        # Map common gender values
        gender_mapping = {
            'Male': ['Male', 'M', 'male', 'Mr', 'MR', 'Mr.'],
            'Female': ['Female', 'F', 'female', 'Mrs', 'MRS', 'Mrs.', 'Ms', 'MS', 'Ms.']
        }
        
        # Create standardized gender column
        self.df['Gender_Std'] = None
        for std_gender, variants in gender_mapping.items():
            for variant in variants:
                mask = self.df[gender_col].str.lower() == variant.lower()
                self.df.loc[mask, 'Gender_Std'] = std_gender
        
        # Filter for Male/Female only
        gender_df = self.df[self.df['Gender_Std'].isin(['Male', 'Female'])]
        
        if len(gender_df) < 100:
            print(f"Warning: Only {len(gender_df)} records with identifiable gender")
        
        # Claim Frequency (Chi-square)
        contingency = pd.crosstab(gender_df['Gender_Std'], gender_df['Claim_Frequency'])
        print(f"\nContingency table:")
        print(contingency)
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        print(f"\nClaim Frequency Test (Chi-square):")
        print(f"  Chi2 Statistic: {chi2:.4f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Result: {'REJECT H₀' if p_value < 0.05 else 'FAIL TO REJECT H₀'}")
        
        # Claim Severity (t-test)
        male_severity = gender_df[gender_df['Gender_Std'] == 'Male']['Claim_Severity'].dropna()
        female_severity = gender_df[gender_df['Gender_Std'] == 'Female']['Claim_Severity'].dropna()
        
        print(f"\nSample sizes - Male: {len(male_severity)}, Female: {len(female_severity)}")
        
        if len(male_severity) > 5 and len(female_severity) > 5:
            t_stat, p_value_ttest = stats.ttest_ind(male_severity, female_severity, equal_var=False)
            print(f"\nClaim Severity Test (Welch\'s t-test):")
            print(f"  T Statistic: {t_stat:.4f}")
            print(f"  P-value: {p_value_ttest:.6f}")
            print(f"  Result: {'REJECT H₀' if p_value_ttest < 0.05 else 'FAIL TO REJECT H₀'}")
        else:
            print("Insufficient data for severity t-test")
            p_value_ttest = 1.0
        
        # Show differences
        print("\nRisk Metrics by Gender:")
        gender_stats = gender_df.groupby('Gender_Std').agg({
            'Claim_Frequency': ['mean', 'count'],
            'Claim_Severity': 'mean',
            'TotalPremium': 'mean',
            'TotalClaims': 'mean'
        }).round(2)
        print(gender_stats)
        
        return min(p_value, p_value_ttest)
    
    def test_zipcode_risk(self):
        """H₀: There are no risk differences between zip codes."""
        print("\n" + "="*60)
        print("TEST 2: Risk Differences Between Zip Codes")
        print("="*60)
        
        # Check for zipcode column
        zip_col = None
        for col in ['PostalCode', 'POSTALCODE', 'postalcode', 'ZipCode', 'ZIPCODE', 'zipcode', 'PostCode']:
            if col in self.df.columns:
                zip_col = col
                break
        
        if not zip_col:
            print("Error: No zip/postal code column found")
            return
        
        print(f"Using column '{zip_col}' for zipcode analysis")
        
        # Clean zip codes
        self.df[zip_col] = self.df[zip_col].astype(str).str.strip()
        
        # Get top zip codes by count
        top_zips = self.df[zip_col].value_counts().head(10).index
        zip_df = self.df[self.df[zip_col].isin(top_zips)]
        
        print(f"\nAnalyzing top 10 zip codes by policy count:")
        for zip_code in top_zips:
            count = len(self.df[self.df[zip_col] == zip_code])
            print(f"  {zip_code}: {count} policies")
        
        # For many categories, use ANOVA on claim severity
        severity_by_zip = []
        zip_names = []
        for z in top_zips:
            severity_data = zip_df[zip_df[zip_col] == z]['Claim_Severity'].dropna()
            if len(severity_data) > 5:  # Minimum sample
                severity_by_zip.append(severity_data)
                zip_names.append(z)
        
        if len(severity_by_zip) >= 2:
            f_stat, p_value = stats.f_oneway(*severity_by_zip)
            print(f"\nClaim Severity Test (ANOVA across {len(zip_names)} zip codes):")
            print(f"  F Statistic: {f_stat:.4f}")
            print(f"  P-value: {p_value:.6f}")
            print(f"  Result: {'REJECT H₀' if p_value < 0.05 else 'FAIL TO REJECT H₀'}")
            
            # Show severity by zip
            print("\nClaim Severity by Zip Code:")
            for z, data in zip(zip_names, severity_by_zip):
                print(f"  {z}: Mean = R{data.mean():.2f}, N = {len(data)}")
        else:
            print("Insufficient data for ANOVA test")
            p_value = 1.0
        
        return p_value
    
    def test_zipcode_margin(self):
        """H₀: There is no significant margin difference between zip codes."""
        print("\n" + "="*60)
        print("TEST 3: Margin Differences Between Zip Codes")
        print("="*60)
        
        if 'Margin' not in self.df.columns or self.df['Margin'].isna().all():
            print("Margin data not available")
            return 1.0
        
        # Check for zipcode column
        zip_col = None
        for col in ['PostalCode', 'POSTALCODE', 'postalcode', 'ZipCode']:
            if col in self.df.columns:
                zip_col = col
                break
        
        if not zip_col:
            print("No zip code column found")
            return 1.0
        
        # Get top zip codes
        top_zips = self.df[zip_col].value_counts().head(5).index
        margin_by_zip = []
        zip_names = []
        
        for z in top_zips:
            margin_data = self.df[self.df[zip_col] == z]['Margin'].dropna()
            if len(margin_data) > 5:
                margin_by_zip.append(margin_data)
                zip_names.append(z)
                print(f"  {z}: Mean margin = R{margin_data.mean():.2f}, N = {len(margin_data)}")
        
        if len(margin_by_zip) >= 2:
            f_stat, p_value = stats.f_oneway(*margin_by_zip)
            print(f"\nMargin Test (ANOVA across {len(zip_names)} zip codes):")
            print(f"  F Statistic: {f_stat:.4f}")
            print(f"  P-value: {p_value:.6f}")
            print(f"  Result: {'REJECT H₀' if p_value < 0.05 else 'FAIL TO REJECT H₀'}")
        else:
            print("Insufficient data for margin ANOVA")
            p_value = 1.0
        
        return p_value
    
    def run_all_tests(self):
        """Run all hypothesis tests."""
        results = {}
        
        print("INSURANCE RISK HYPOTHESIS TESTING")
        print("="*60)
        print(f"Total records: {len(self.df):,}")
        print(f"Claims filed: {(self.df['Claim_Frequency'] == 1).sum():,}")
        print(f"Claim rate: {(self.df['Claim_Frequency'] == 1).mean():.2%}")
        
        # Test 1: Province Risk
        results['province_risk'] = self.test_province_risk()
        
        # Test 2: Zip Code Risk
        results['zipcode_risk'] = self.test_zipcode_risk()
        
        # Test 3: Margin differences
        results['zipcode_margin'] = self.test_zipcode_margin()
        
        # Test 4: Gender Risk
        results['gender_risk'] = self.test_gender_risk()
        
        # Summary
        print("\n" + "="*60)
        print("HYPOTHESIS TESTING SUMMARY")
        print("="*60)
        for test_name, p_value in results.items():
            test_label = {
                'province_risk': 'Province Risk Differences',
                'zipcode_risk': 'Zip Code Risk Differences',
                'zipcode_margin': 'Zip Code Margin Differences',
                'gender_risk': 'Gender Risk Differences'
            }[test_name]
            
            decision = "REJECT" if p_value < 0.05 else "FAIL TO REJECT"
            print(f"{test_label}: p = {p_value:.6f} -> {decision} H₀")
        
        return results

if __name__ == "__main__":
    # Run with pipe delimiter
    print("Starting hypothesis testing with pipe-delimited data...")
    tester = InsuranceHypothesisTester("data/raw/machineLearningRating_v3.txt", delimiter='|')
    results = tester.run_all_tests()
    
    # Save results
    pd.Series(results).to_csv("hypothesis_test_results.csv")
    print("\nResults saved to hypothesis_test_results.csv")
    
    # Generate quick report
    print("\n" + "="*60)
    print("BUSINESS RECOMMENDATIONS")
    print("="*60)
    
    if results.get('province_risk', 1) < 0.05:
        print("• ADJUST premiums by province (significant risk differences found)")
    else:
        print("• Province-based pricing may not be necessary")
    
    if results.get('zipcode_risk', 1) < 0.05:
        print("• Consider zip code granularity in risk assessment")
    
    if results.get('gender_risk', 1) < 0.05:
        print("• Gender is a significant risk factor - adjust pricing accordingly")
    else:
        print("• Gender-neutral pricing appears appropriate")