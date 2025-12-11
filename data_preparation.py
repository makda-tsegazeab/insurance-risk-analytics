# data_preparation.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class InsuranceDataPreparer:
    def __init__(self, data_path, delimiter='|'):
        """Load and prepare insurance data for modeling."""
        print("Loading data for predictive modeling...")
        
        # Load data
        self.df = pd.read_csv(data_path, delimiter=delimiter, low_memory=False)
        print(f"Original data shape: {self.df.shape}")
        
        # Clean column names
        self.df.columns = self.df.columns.str.strip()
        
        # Display column information
        print(f"\nColumns available: {len(self.df.columns)}")
        print("First 20 columns:", list(self.df.columns[:20]))
        
    def explore_data_structure(self):
        """Explore the data structure and key variables."""
        print("\n" + "="*60)
        print("DATA EXPLORATION FOR MODELING")
        print("="*60)
        
        # Check target variables
        target_vars = ['TotalClaims', 'TotalPremium', 'CalculatedPremiumPerTerm']
        available_targets = [col for col in target_vars if col in self.df.columns]
        print(f"Available target variables: {available_targets}")
        
        # Check key feature columns
        feature_categories = {
            'Client Info': ['Gender', 'MaritalStatus', 'Language', 'Age'],
            'Location': ['Province', 'PostalCode', 'Country', 'MainCrestaZone'],
            'Vehicle': ['Make', 'Model', 'VehicleType', 'RegistrationYear', 'Bodytype'],
            'Policy': ['CoverType', 'Product', 'SumInsured', 'ExcessSelected']
        }
        
        print("\nAvailable features by category:")
        for category, cols in feature_categories.items():
            available = [col for col in cols if col in self.df.columns]
            if available:
                print(f"  {category}: {available}")
        
        # Basic statistics
        if 'TotalClaims' in self.df.columns:
            print(f"\nClaims statistics:")
            print(f"  Total policies: {len(self.df)}")
            print(f"  Policies with claims: {(self.df['TotalClaims'] > 0).sum()}")
            print(f"  Claim rate: {(self.df['TotalClaims'] > 0).mean():.2%}")
            print(f"  Average claim amount: R{self.df[self.df['TotalClaims'] > 0]['TotalClaims'].mean():.2f}")
    
    def prepare_claim_severity_data(self):
        """Prepare data for claim severity prediction (regression)."""
        print("\n" + "="*60)
        print("PREPARING CLAIM SEVERITY DATA")
        print("="*60)
        
        # Only policies with claims
        severity_df = self.df[self.df['TotalClaims'] > 0].copy()
        print(f"Policies with claims: {len(severity_df)}")
        
        if len(severity_df) == 0:
            print("No claims data available!")
            return None, None, None, None
        
        # Target variable
        y = severity_df['TotalClaims'].values
        
        # Select features
        features = self.select_features(severity_df)
        X = severity_df[features]
        
        print(f"Selected {len(features)} features for severity prediction")
        print(f"Feature names: {features}")
        
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Encode categorical variables
        X_encoded, encoders = self.encode_categorical_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.3, random_state=42
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, encoders
    
    def prepare_premium_prediction_data(self):
        """Prepare data for premium prediction (regression)."""
        print("\n" + "="*60)
        print("PREPARING PREMIUM PREDICTION DATA")
        print("="*60)
        
        # Use all policies
        premium_df = self.df.copy()
        
        # Target variable - try different premium columns
        if 'CalculatedPremiumPerTerm' in premium_df.columns and not premium_df['CalculatedPremiumPerTerm'].isna().all():
            y = premium_df['CalculatedPremiumPerTerm'].values
            print("Using CalculatedPremiumPerTerm as target")
        elif 'TotalPremium' in premium_df.columns:
            y = premium_df['TotalPremium'].values
            print("Using TotalPremium as target")
        else:
            print("No premium column found!")
            return None, None, None, None
        
        # Remove rows where target is missing or zero
        mask = ~pd.isna(y) & (y > 0)
        premium_df = premium_df[mask]
        y = y[mask]
        
        print(f"Valid policies for premium prediction: {len(premium_df)}")
        
        # Select features
        features = self.select_features(premium_df, include_risk=True)
        X = premium_df[features]
        
        print(f"Selected {len(features)} features for premium prediction")
        
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Encode categorical variables
        X_encoded, encoders = self.encode_categorical_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.3, random_state=42
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, encoders
    
    def prepare_claim_probability_data(self):
        """Prepare data for claim probability prediction (classification)."""
        print("\n" + "="*60)
        print("PREPARING CLAIM PROBABILITY DATA")
        print("="*60)
        
        # Use all policies
        claim_df = self.df.copy()
        
        # Create binary target: 1 if claim > 0, else 0
        if 'TotalClaims' in claim_df.columns:
            y = (claim_df['TotalClaims'] > 0).astype(int).values
            print(f"Claim rate: {y.mean():.2%}")
        else:
            print("No claims data available!")
            return None, None, None, None
        
        # Select features
        features = self.select_features(claim_df, include_risk=True)
        X = claim_df[features]
        
        print(f"Selected {len(features)} features for claim probability")
        
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Encode categorical variables
        X_encoded, encoders = self.encode_categorical_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Claim rate in training: {y_train.mean():.2%}")
        print(f"Claim rate in test: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test, encoders
    
    def select_features(self, df, include_risk=False):
        """Select relevant features for modeling."""
        feature_candidates = []
        
        # Client features
        client_features = ['Gender', 'MaritalStatus', 'Language', 'Title']
        feature_candidates.extend([f for f in client_features if f in df.columns])
        
        # Location features
        location_features = ['Province', 'MainCrestaZone', 'SubCrestaZone']
        feature_candidates.extend([f for f in location_features if f in df.columns])
        
        # Vehicle features
        vehicle_features = [
            'Make', 'VehicleType', 'RegistrationYear', 'Bodytype',
            'Cylinders', 'cubiccapacity', 'kilowatts', 'NumberOfDoors'
        ]
        feature_candidates.extend([f for f in vehicle_features if f in df.columns])
        
        # Policy features
        policy_features = [
            'CoverType', 'Product', 'SumInsured', 'ExcessSelected',
            'TermFrequency', 'CoverCategory'
        ]
        feature_candidates.extend([f for f in policy_features if f in df.columns])
        
        # Remove features with too many missing values
        selected_features = []
        for feature in feature_candidates:
            missing_ratio = df[feature].isna().mean()
            unique_ratio = df[feature].nunique() / len(df)
            
            # Keep if not too many missing and not too many unique values (for categorical)
            if missing_ratio < 0.5 and unique_ratio < 0.9:
                selected_features.append(feature)
        
        return selected_features
    
    def handle_missing_values(self, X):
        """Handle missing values in features."""
        # For numerical columns, fill with median
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        
        # For categorical columns, fill with mode
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
        
        return X
    
    def encode_categorical_features(self, X):
        """Encode categorical features using Label Encoding."""
        encoders = {}
        X_encoded = X.copy()
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            # Handle unseen values by adding 'Unknown' category
            unique_vals = X[col].unique()
            le.fit(list(unique_vals) + ['Unknown'])
            
            # Transform and handle any new values
            try:
                X_encoded[col] = le.transform(X[col])
            except ValueError:
                # For any unseen values, use 'Unknown'
                X_encoded[col] = X[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else le.transform(['Unknown'])[0]
                )
            
            encoders[col] = le
        
        return X_encoded, encoders

if __name__ == "__main__":
    # Test data preparation
    preparer = InsuranceDataPreparer("data/raw/machineLearningRating_v3.txt", delimiter='|')
    preparer.explore_data_structure()
    
    # Prepare different datasets
    print("\n" + "="*60)
    print("PREPARING ALL MODELING DATASETS")
    print("="*60)
    
    # Claim severity
    severity_data = preparer.prepare_claim_severity_data()
    
    # Premium prediction
    premium_data = preparer.prepare_premium_prediction_data()
    
    # Claim probability
    probability_data = preparer.prepare_claim_probability_data()
    
    print("\nData preparation complete!")