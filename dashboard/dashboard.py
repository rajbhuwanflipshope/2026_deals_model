from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
import pymongo
import datetime
import os
import sys

app = Flask(__name__)

# Load XGBoost Model
model = None
paths = ["labeled2.pkl", "../labeled2.pkl", "dashboard/labeled2.pkl"]
for path in paths:
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            print(f"Successfully loaded XGBoost model from {path}")
            break
        except Exception as e:
            print(f"Error loading model from {path}: {str(e)}")

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
        

            
        # 1. Fetch latest 300 documents using primary key sorting (extremely fast)
        raw_docs = list(coll.find(sort=[("Time", -1)]).limit(300))
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

        # 4. Predict score and format JSON response
        processed = []
        for doc in docs:
            pid = doc.get("pid", "Unknown")
            sid = doc.get("sid")
            pg_info = graph_lookup.get((sid, pid), {})
            g_data = pg_info.get("data")
            pg_imgurl = pg_info.get("imgurl")

            feat = extract_features(doc, graph_data=g_data)
            score = 0
            if model is not None:
                try:
                    df_in = pd.DataFrame([{
                        "price": feat["price"],
                        "median_180": feat["median_180"],
                        "min_180": feat["min_180"],
                        "flash_factor": feat["flash_factor"]
                    }])
                    prob = model.predict_proba(df_in)[:, 1][0]
                    score = int(round(float(prob) * 1000))
                except Exception:
                    score = 0

            # Show only products predicted as a deal (score >= 500)
            if score < 500:
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
                elif sid == 13:
                    url = f"https://www.myntra.com/{pid}"
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
                "score": score,
                "db_score": db_score_val,
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
    # Start Flask server on port 5000
    app.run(host='0.0.0.0', port=5005, debug=True, use_reloader=False)
