from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
import pymongo
import datetime
import os
import sys
import subprocess
import json

app = Flask(__name__)



# Load Labeled2 Model
labeled2_model = None
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
paths = [
    os.path.join(root_path, "labeled2.pkl"),
    "labeled2.pkl",
    "dashboard/labeled2.pkl"
]
for path in paths:
    if os.path.exists(path):
        try:
            labeled2_model = joblib.load(path)
            print(f"Successfully loaded labeled2 model from {path}")
            break
        except Exception as e:
            print(f"Error loading labeled2 model from {path}: {str(e)}")

# Load Old Models
old_model = None
old_scaler = None
old_encoder = None

old_dir = os.path.join(root_path, "old")
if os.path.exists(old_dir):
    try:
        old_model = joblib.load(os.path.join(old_dir, "deals_model.pkl"))
        old_scaler = joblib.load(os.path.join(old_dir, "scaler.pkl"))
        old_encoder = joblib.load(os.path.join(old_dir, "encoder.pkl"))
        print("Successfully loaded old models from old/ directory")
    except Exception as e:
        print(f"Error loading old models from old/ directory: {str(e)}")


def extract_all_features(doc):
    def to_float(val, default=0.0):
        if val is None or str(val).strip() == "" or str(val).strip().lower() in ["none", "null"]:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    # Safe category extraction
    cat_val = doc.get("category")
    try:
        cat_val = int(float(cat_val)) if cat_val is not None else 2
    except (ValueError, TypeError):
        cat_val = 2

    return {
        "price": to_float(doc.get("price")),
        "rating": to_float(doc.get("rating")),
        "rating_count": to_float(doc.get("rating_count")),
        "last_lowest_price": to_float(doc.get("last_lowest_price")),
        "ma_15": to_float(doc.get("ma_15")),
        "ma_3": to_float(doc.get("ma_3")),
        "ma_30": to_float(doc.get("ma_30")),
        "ma_7": to_float(doc.get("ma_7")),
        "median": to_float(doc.get("median")),
        "day_percent_30": to_float(doc.get("day_percent_30")),
        "day_percent_90": to_float(doc.get("day_percent_90")),
        "drop_median": to_float(doc.get("drop_median")),
        "drop_p20": to_float(doc.get("drop_p20")),
        "category": cat_val
    }

def predict_old_models_batch(feats):
    if not feats or old_model is None or old_scaler is None or old_encoder is None:
        return [0] * len(feats)
    try:
        df = pd.DataFrame(feats)
        encoded = old_encoder.transform(df[['category']])
        encoded_df = pd.DataFrame(encoded, columns=old_encoder.get_feature_names_out(['category']), index=df.index)
        num_cols = [
            'price', 'rating', 'rating_count', 'last_lowest_price', 'ma_15', 'ma_3', 'ma_30',
            'ma_7', 'median', 'day_percent_30', 'day_percent_90', 'drop_median', 'drop_p20'
        ]
        df[num_cols] = old_scaler.transform(df[num_cols])
        X_input = pd.concat([df.drop(columns=['category']), encoded_df], axis=1)
        cols = list(old_model.feature_names_in_)
        X_input = X_input[cols]
        probs = old_model.predict_proba(X_input)[:, 1]
        return [int(round(float(prob) * 1000)) for prob in probs]
    except Exception as e:
        print(f"Error in old model prediction batch: {str(e)}")
        return [0] * len(feats)


category_avg_discount = {
    1: 45, 2: 35, 3: 55, 4: 30, 5: 30,
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

def predict_labeled2_batch(feats):
    if not feats or labeled2_model is None:
        return [0] * len(feats)
    try:
        df = pd.DataFrame(feats)
        
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

        model_features = list(getattr(labeled2_model, 'feature_names_in_', ['price', 'median_180', 'min_180', 'flash_factor']))
        cols_to_drop = [c for c in ['category'] if c not in model_features]
        X_input = pd.concat([df.drop(columns=cols_to_drop, errors='ignore'), encoded_df], axis=1)
        
        probs = labeled2_model.predict_proba(X_input[model_features])[:, 1]
        

        
        return [int(round(float(prob) * 1000)) for prob in probs]
    except Exception as e:
        print(f"Error in labeled2 prediction batch: {str(e)}")
        return [0] * len(feats)



# MongoDB Configuration
mongo_uri = "mongodb://read_only:v%3F8lT%21sw%26pu4ec2zaPra@143.110.184.59:27017/?authMechanism=DEFAULT"
client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)

# One-time Index Initialization at startup
try:
    _db = client["fs_graph"]
    _db["deals_queue"].create_index([("Time", -1)], background=True)
    _db["price_graph"].create_index([("sid", 1), ("pid", 1)], background=True)
    print("Database indexes initialized successfully.")
except Exception as _e:
    print(f"Warning: Could not initialize database indexes at startup: {str(_e)}")

def calculate_180d_anchors(graph_data):
    """Calculate median_180, min_180, and flash_factor from graph_data."""
    if not graph_data:
        return {"median_180": 0.0, "min_180": 0.0, "flash_factor": 0.0}

    graph_dict = {}
    if isinstance(graph_data, list):
        for entry in graph_data:
            if isinstance(entry, dict) and "time" in entry:
                date_str = entry["time"]
                min_p = entry.get("min_price")
                if min_p is None:
                    min_p = entry.get("price", 0)
                max_p = entry.get("max_price", min_p)
                graph_dict[date_str] = {"0": min_p, "1": max_p}
    elif isinstance(graph_data, dict):
        for k, v in graph_data.items():
            if isinstance(v, dict):
                graph_dict[k] = v
            else:
                graph_dict[k] = {"0": v, "1": v}

    if not graph_dict:
        return {"median_180": 0.0, "min_180": 0.0, "flash_factor": 0.0}

    try:
        all_dates = []
        for d in graph_dict.keys():
            try:
                all_dates.append(datetime.datetime.strptime(d, "%Y-%m-%d"))
            except ValueError:
                continue

        if not all_dates:
            return {"median_180": 0.0, "min_180": 0.0, "flash_factor": 0.0}

        latest_date = max(all_dates)
    except Exception:
        return {"median_180": 0.0, "min_180": 0.0, "flash_factor": 0.0}

    cutoff_date = latest_date - datetime.timedelta(days=180)

    prices_low = []
    prices_high = []

    for date_str, values in graph_dict.items():
        try:
            price_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            # strict < latest_date
            if cutoff_date <= price_date < latest_date and isinstance(values, dict):
                low_p = float(values.get("0", 0))
                high_p = float(values.get("1", low_p))
                if high_p <= 0:
                    high_p = low_p
                if low_p > 0:
                    prices_low.append(low_p)
                    prices_high.append(high_p)
        except Exception:
            continue

    if len(prices_low) < 3:
        return {"median_180": 0.0, "min_180": 0.0, "flash_factor": 0.0}

    prices_low_sorted = sorted(prices_low)
    n = len(prices_low_sorted)
    if n % 2 != 0:
        hist_median = prices_low_sorted[n // 2]
    else:
        hist_median = (prices_low_sorted[n // 2 - 1] + prices_low_sorted[n // 2]) / 2.0

    clean_floor = min(prices_low)

    intraday_spreads = [(h - l) / h for l, h in zip(prices_low, prices_high) if h > 0]
    if intraday_spreads:
        mean_spread = sum(intraday_spreads) / len(intraday_spreads)
    else:
        mean_spread = 0.0

    flash_factor = min(mean_spread / 0.30, 1.0)

    return {
        "median_180": float(hist_median),
        "min_180": float(clean_floor),
        "flash_factor": float(flash_factor)
    }

def extract_features(item, graph_data=None):
    """Extract features for model prediction."""
    features = {
        "category": item.get("category"),
        "price": item.get("price"),
        "median_180": item.get("median_180"),
        "min_180": item.get("min_180"),
        "flash_factor": item.get("flash_factor")
    }

    if (features["median_180"] is None or 
        features["min_180"] is None or 
        features["flash_factor"] is None):
        
        # Use pre-fetched graph_data if available
        g_data = graph_data if graph_data is not None else item.get("graph_data")
        if g_data:
            computed = calculate_180d_anchors(g_data)
            for k in ["median_180", "min_180", "flash_factor"]:
                if features[k] is None:
                    features[k] = computed[k]
        else:
            for k in ["median_180", "min_180", "flash_factor"]:
                if features[k] is None:
                    features[k] = 0.0

    try:
        features["price"] = float(features["price"]) if features["price"] is not None else 0.0
    except ValueError:
        features["price"] = 0.0

    return features

def get_time_ago(doc_time):
    """Formats timestamp into relative 'ago' format."""
    if not doc_time:
        return "recently"
    
    val = None
    if isinstance(doc_time, dict) and "$date" in doc_time:
        val = doc_time["$date"]
    elif isinstance(doc_time, str):
        val = doc_time
    else:
        val = str(doc_time)
        
    try:
        if "T" in val:
            val = val.replace("Z", "")
            if "." in val:
                val = val.split(".")[0]
            dt = datetime.datetime.strptime(val, "%Y-%m-%dT%H:%M:%S")
        else:
            dt = datetime.datetime.strptime(val.split(".")[0], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return "recently"
        
    now = datetime.datetime.utcnow()
    delta = now - dt
    diff_secs = int(delta.total_seconds())
    
    if diff_secs < 0:
        now = datetime.datetime.now()
        delta = now - dt
        diff_secs = int(delta.total_seconds())
        
    if diff_secs < 60:
        return f"{max(1, diff_secs)} sec ago"
    elif diff_secs < 3600:
        return f"{diff_secs // 60} min ago"
    elif diff_secs < 86400:
        return f"{diff_secs // 3600} hour ago"
    else:
        return f"{diff_secs // 86400} day ago"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/deals')
def get_deals():
    try:
        db = client["fs_graph"]
        coll = db["deals_queue"]
        

            
        # 1. Fetch latest 300 documents using primary key sorting (extremely fast using indexed Time)
        raw_docs = list(coll.find(sort=[("Time", -1)]).limit(1000))
        if not raw_docs:
            return jsonify([])

        # 2. Determine 1-hour time window
        times = []
        for d in raw_docs:
            t = d.get("Time")
            if isinstance(t, datetime.datetime):
                times.append(t)
            elif t is not None:
                try:
                    val = str(t)
                    if "T" in val:
                        val = val.replace("Z", "").split(".")[0]
                        times.append(datetime.datetime.strptime(val, "%Y-%m-%dT%H:%M:%S"))
                    else:
                        times.append(datetime.datetime.strptime(val.split(".")[0], "%Y-%m-%d %H:%M:%S"))
                except Exception:
                    continue
        
        docs = []
        if times:
            max_time = max(times)
            cutoff = max_time - datetime.timedelta(hours=1)
            # Filter locally in memory
            for d in raw_docs:
                t = d.get("Time")
                if isinstance(t, datetime.datetime):
                    if t >= cutoff:
                        docs.append(d)
                elif t is not None:
                    try:
                        val = str(t)
                        if "T" in val:
                            val = val.replace("Z", "").split(".")[0]
                            parsed_t = datetime.datetime.strptime(val, "%Y-%m-%dT%H:%M:%S")
                        else:
                            parsed_t = datetime.datetime.strptime(val.split(".")[0], "%Y-%m-%d %H:%M:%S")
                        if parsed_t >= cutoff:
                            docs.append(d)
                    except Exception:
                        docs.append(d)
        else:
            docs = raw_docs

        if not docs:
            return jsonify([])

        # 3. Batch query price_graph collection for all matching sid/pid pairs
        graph_lookup = {}
        pairs = []
        for doc in docs:
            pid_val = doc.get("pid")
            sid_val = doc.get("sid")
            if pid_val is not None and sid_val is not None:
                pairs.append({"sid": sid_val, "pid": pid_val})
        
        if pairs:
            pg_coll = db["price_graph"]
            pg_docs = list(pg_coll.find({"$or": pairs}, {"sid": 1, "pid": 1, "data": 1, "imgurl": 1}))
            for pg in pg_docs:
                pg_sid = pg.get("sid")
                pg_pid = pg.get("pid")
                if pg_sid is not None and pg_pid is not None:
                    graph_lookup[(pg_sid, pg_pid)] = {
                        "data": pg.get("data"),
                        "imgurl": pg.get("imgurl")
                    }

        # 4. Extract features for all docs
        features_list = []
        for doc in docs:
            pid = doc.get("pid", "Unknown")
            sid = doc.get("sid")
            pg_info = graph_lookup.get((sid, pid), {})
            g_data = pg_info.get("data")
            feat = extract_features(doc, graph_data=g_data)
            features_list.append(feat)

        # 5. Perform predictions using both models and calculate average score
        # Retrieve old score directly from database (using doc.get("score") which matches predicted old model score)
        old_scores = []
        for doc in docs:
            raw_old = doc.get("score")
            try:
                val = int(float(raw_old)) if raw_old is not None else 0
            except (ValueError, TypeError):
                val = 0
            old_scores.append(val)

        all_feats_labeled2 = features_list
        labeled2_scores = predict_labeled2_batch(all_feats_labeled2)

        scores = [int(round((o_s + l_s) / 2.0)) for o_s, l_s in zip(old_scores, labeled2_scores)]

        # 6. Format JSON response
        processed = []
        for idx, doc in enumerate(docs):
            pid = doc.get("pid", "Unknown")
            sid = doc.get("sid")
            pg_info = graph_lookup.get((sid, pid), {})
            g_data = pg_info.get("data")
            pg_imgurl = pg_info.get("imgurl")

            feat = features_list[idx]
            average_score = scores[idx]
            labeled2_score = labeled2_scores[idx]
            db_score = old_scores[idx]

            # Show only products predicted as a deal (average score > 600)
            if average_score <= 400:
                continue

            # Exclude products which don't have a valid title
            title_val = doc.get("title")
            if not title_val or str(title_val).strip().lower() in ["", "none", "null"]:
                continue

            # Safe numeric fields for UI rendering
            raw_price = feat.get("price")
            raw_mrp = doc.get("mrp")
            raw_discount = doc.get("discount")

            try:
                price_val = int(float(raw_price)) if raw_price is not None else 0
            except (ValueError, TypeError):
                price_val = 0

            try:
                mrp_val = int(float(raw_mrp)) if raw_mrp is not None else price_val
            except (ValueError, TypeError):
                mrp_val = price_val

            try:
                discount_val = int(float(raw_discount)) if raw_discount is not None else 0
            except (ValueError, TypeError):
                discount_val = 0

            raw_db_score = doc.get("score")
            try:
                db_score_val = int(float(raw_db_score)) if raw_db_score is not None else "N/A"
            except (ValueError, TypeError):
                db_score_val = "N/A"

            img = doc.get("imgurl") or doc.get("image") or pg_imgurl or ""

            # Extract price history for Chart.js (6 months / 180 days only)
            history_list = []
            if g_data and isinstance(g_data, dict):
                parsed_history = []
                for date_str, price_info in g_data.items():
                    try:
                        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                        if isinstance(price_info, dict):
                            low_p = price_info.get("0")
                        else:
                            low_p = price_info
                        parsed_history.append((dt, date_str, float(low_p) if low_p is not None else 0.0))
                    except ValueError:
                        continue
                
                if parsed_history:
                    # Sort chronological (ascending date)
                    parsed_history.sort(key=lambda x: x[0])
                    # Keep records within the last 180 days from the maximum available date
                    max_date = parsed_history[-1][0]
                    six_months_ago = max_date - datetime.timedelta(days=180)
                    for dt, date_str, price in parsed_history:
                        if dt >= six_months_ago:
                            history_list.append({
                                "date": date_str,
                                "price": price
                            })

            # Construct product URL dynamically based on sid and pid if aff_url is missing
            title_str = doc.get("title", "No Title")
            url = doc.get("aff_url")
            if not url or url == "#":
                if sid == 1:
                    url = f"https://www.flipkart.com/product/p/itme?pid={pid}"
                elif sid == 2:
                    url = f"https://www.amazon.in/dp/{pid}"
                elif sid == 7:
                    url = f"https://www.myntra.com/{pid}"
                elif sid == 9:
                    url = f"https://www.ajio.com/search/?text={pid}"
                elif sid == 10:
                    if str(pid).endswith(".html"):
                        url = f"https://www.pepperfry.com/product/{pid}"
                    else:
                        import re
                        slug = re.sub(r'[^a-zA-Z0-9\s-]', '', title_str).strip().lower()
                        slug = re.sub(r'[\s-]+', '-', slug)
                        url = f"https://www.pepperfry.com/product/{slug}-{pid}.html"
                elif sid == 13:
                    url = f"https://www.croma.com/search/p/{pid}"
                elif sid == 14:
                    url = f"https://www.reliancedigital.in/search?q={pid}:relevance"
                else:
                    import urllib.parse
                    url = f"https://www.google.com/search?q={urllib.parse.quote(title_str or pid)}"

            processed.append({
                "pid": pid,
                "sid": sid,
                "title": title_str,
                "price": price_val,
                "mrp": mrp_val,
                "discount": discount_val,
                "score": labeled2_score,
                "db_score": db_score,
                "imgurl": img,
                "aff_url": url,
                "time_ago": get_time_ago(doc.get("Time")),
                "history": history_list,
                "median_180": int(round(feat.get("median_180", 0.0))) if feat.get("median_180") is not None else price_val,
                "category": str(doc.get("category", "N/A"))
            })

        return jsonify(processed)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Start Flask server on port from environment or default to 5005
    port = int(os.environ.get("PORT", 5005))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
