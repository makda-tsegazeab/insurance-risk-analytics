# generate_report.py - FIXED VERSION
import pandas as pd
import numpy as np

def generate_report():
    """Generate a summary report of hypothesis testing results."""
    
    report = """# Insurance Risk Hypothesis Testing Report
## AlphaCare Insurance Solutions (ACIS)
### Date: December 2025

## Executive Summary
This report presents the results of A/B hypothesis testing to identify risk drivers in the insurance portfolio. The findings will inform segmentation strategy and premium optimization.

## Methodology
- **Risk Metrics**: Claim Frequency (proportion with claims) and Claim Severity (average claim amount)
- **Margin Metric**: TotalPremium - TotalClaims
- **Statistical Tests**: Chi-square for frequencies, ANOVA/T-tests for continuous variables
- **Significance Level**: α = 0.05

## Results Summary
"""
    
    try:
        results = pd.read_csv("hypothesis_test_results.csv", index_col=0)
        
        # Convert Series to dict if needed
        if isinstance(results, pd.Series):
            results_dict = results.to_dict()
        else:
            # If it's a DataFrame with one column
            results_dict = results.iloc[:, 0].to_dict()
        
        test_descriptions = {
            'province_risk': 'Risk Differences Across Provinces',
            'zipcode_risk': 'Risk Differences Between Zip Codes',
            'zipcode_margin': 'Margin Differences Between Zip Codes',
            'gender_risk': 'Risk Differences Between Women and Men'
        }
        
        for test_name, p_value in results_dict.items():
            test_desc = test_descriptions.get(test_name, test_name.replace('_', ' ').title())
            
            # Ensure p_value is a number
            try:
                p_val = float(p_value)
                decision = "REJECT" if p_val < 0.05 else "FAIL TO REJECT"
            except:
                p_val = 1.0
                decision = "INCONCLUSIVE"
            
            report += f"\n### {test_desc}\n"
            report += f"- **P-value**: {p_val:.6f}\n"
            report += f"- **Decision**: {decision} the null hypothesis\n"
            report += f"- **Business Implication**: "
            
            if p_val < 0.05:
                report += "Statistically significant differences exist. Consider adjusting premiums or marketing strategy for this segment.\n"
            else:
                report += "No statistically significant differences found. Current pricing strategy may be appropriate.\n"
    
    except FileNotFoundError:
        report += "\n*Results file not found. Please run hypothesis testing first.*"
    except Exception as e:
        report += f"\n*Error generating report: {str(e)}*"
    
    report += """
## Recommendations

### For Marketing Strategy:
1. **Target Low-Risk Segments**: Identify provinces/zip codes with lower claim frequencies for targeted marketing campaigns
2. **Risk-Based Pricing**: Adjust premiums based on statistically significant risk factors
3. **Geographic Focus**: Concentrate resources on high-margin, low-risk regions

### For Future Analysis:
1. **Additional Segmentation**: Test other variables like VehicleType, Age, MaritalStatus
2. **Time-Series Analysis**: Monitor risk trends over time
3. **Predictive Modeling**: Build ML models to predict claim likelihood and severity

## Limitations
- Analysis based on historical data from Feb 2014-Aug 2015
- External factors (economic conditions, weather) not accounted for
- Sample size limitations for some geographic segments
"""
    
    # Save report
    with open("hypothesis_testing_report.md", "w") as f:
        f.write(report)
    
    print("Report generated: hypothesis_testing_report.md")
    print(report)

if __name__ == "__main__":
    generate_report()