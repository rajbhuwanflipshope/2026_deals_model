import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
import time
import sys
import json

# Load once at start
model = joblib.load("labeled2.pkl")
# scaler = joblib.load("scaler.pkl") # No scaler used for labeled2.pkl

# Hard-coded category dictionaries
category_avg_discount = {
    1: 45, 2: 35, 3: 40, 4: 30, 5: 30,
    6: 20, 7: 25, 8: 45, 9: 45, 10: 20,
    11: 40, 12: 25, 13: 25, 14: 45,
    15: 28, 16: 35, 17: 25
}

category_tolerance = {
    1: 3, 2: 2, 3: 4, 4: 2, 5: 3,
    6: 2, 7: 2, 8: 3, 9: 3, 10: 2,
    11: 4, 12: 1, 13: 1, 14: 3,
    15: 3, 16: 3, 17: 2
}

cat_cols = ['category']
num_cols = ['price', 'median_180', 'min_180', 'flash_factor']

print("_READY", flush=True)

while True:
    try:
        raw = sys.stdin.readline()
        if not raw:
            break
 
        input_dict = json.loads(raw.strip())
        job_id = str(input_dict.pop("job_id", ""))
        start = time.time()
 
        df = pd.DataFrame([input_dict])
        
        # Hard coded part for category (Replaces encoder.transform)
        df['category'] = pd.to_numeric(df['category'], errors='coerce')
        avg_category_discount = df['category'].map(category_avg_discount).fillna(25)
        category_tolerance_val = df['category'].map(category_tolerance).fillna(2)
        cat_target_pct = (avg_category_discount - category_tolerance_val) / 100.0
        
        encoded_df = pd.DataFrame({
            'avg_category_discount': avg_category_discount,
            'category_tolerance_val': category_tolerance_val,
            'cat_target_pct': cat_target_pct
        }, index=df.index)

        # df[num_cols] = scaler.transform(df[num_cols]) # Scaler was not used for this model
        
        X_input = pd.concat([df.drop(columns=cat_cols, errors='ignore'), encoded_df], axis=1)
        
        # Ensure we only pass the columns the model was trained on to prevent feature mismatch errors
        model_features = getattr(model, 'feature_names_in_', num_cols)
        prob = model.predict_proba(X_input[model_features])[:, 1][0]
 
        end = time.time()
        result = {
            "job_id": job_id,
            "probability": int(round(float(prob) * 1000))
        }
 
        print(json.dumps(result))
        sys.stdout.flush()
 
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.stdout.flush()
