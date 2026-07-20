import sys
import json
import datetime

def calculate_180d_anchors(graph_data):
    """
    Computes median_180, min_180, and flash_factor from graph_data.
    Accepts either a dict: {date_str: {"0": min_price, "1": max_price}}
    or a list of dicts: [{"time": date_str, "min_price": min_p, "max_price": max_p}]
    """
    if not graph_data:
        return {"median_180": 0.0, "min_180": 0.0, "flash_factor": 0.0}

    # Normalize graph_data to a dictionary {date_str: {"0": min, "1": max}}
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
            # strict < latest_date as per calculate_180d_anchors_cleaned
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

    # Pure Python median
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

def extract_record(item):
    """
    Extracts category, price, and calculates median_180, min_180, flash_factor.
    """
    extracted = {
        "category": item.get("category"),
        "price": item.get("price"),
        "median_180": item.get("median_180"),
        "min_180": item.get("min_180"),
        "flash_factor": item.get("flash_factor")
    }

    # If any of the numerical history fields are missing, compute them from graph_data
    if (extracted["median_180"] is None or 
        extracted["min_180"] is None or 
        extracted["flash_factor"] is None):
        
        graph_data = item.get("graph_data") or item.get("data")
        if graph_data:
            computed = calculate_180d_anchors(graph_data)
            for k in ["median_180", "min_180", "flash_factor"]:
                if extracted[k] is None:
                    extracted[k] = computed[k]
        else:
            # Fallback defaults if graph_data is missing too
            for k in ["median_180", "min_180", "flash_factor"]:
                if extracted[k] is None:
                    extracted[k] = 0.0

    # Ensure clean types
    if extracted["category"] is not None:
        try:
            extracted["category"] = int(extracted["category"])
        except ValueError:
            pass

    if extracted["price"] is not None:
        try:
            extracted["price"] = float(extracted["price"])
        except ValueError:
            pass

    return extracted

def process_json_data(raw_data):
    try:
        parsed = json.loads(raw_data)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON input: {str(e)}"}
        
    if isinstance(parsed, list):
        return [extract_record(item) if isinstance(item, dict) else item for item in parsed]
    elif isinstance(parsed, dict):
        return extract_record(parsed)
    else:
        return {"error": "JSON root must be a list or an object."}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            with open(sys.argv[1], 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(json.dumps({"error": f"Failed to read file: {str(e)}"}))
            sys.exit(1)
    else:
        content = sys.stdin.read()
        
    if not content.strip():
        print(json.dumps({"error": "No input data provided."}))
        sys.exit(1)
        
    output = process_json_data(content)
    print(json.dumps(output))
