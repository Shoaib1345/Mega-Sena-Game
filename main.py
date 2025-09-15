import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, jaccard_score, f1_score

# ===================== CONFIG =====================
CSV_FILENAME = "Mega-Sena (2).csv"
MIN_RELATED_PER_GAME = 2        # each game must include at least this many relation-based numbers
N_GAMES = 50
NUMBERS_PER_GAME = 6
# ==================================================

# ------------- Load & preprocess data -------------
df = pd.read_csv(CSV_FILENAME, encoding='latin1')

# Normalize common Mega-Sena headers
rename_map_options = [
    {
        'Concurso': 'contest',
        'Data do Sorteio': 'date',
        'Bola1': 'n1', 'Bola2': 'n2', 'Bola3': 'n3',
        'Bola4': 'n4', 'Bola5': 'n5', 'Bola6': 'n6'
    },
    {
        'concurso': 'contest',
        'data': 'date',
        'dezena_1': 'n1', 'dezena_2': 'n2', 'dezena_3': 'n3',
        'dezena_4': 'n4', 'dezena_5': 'n5', 'dezena_6': 'n6'
    }
]
for rm in rename_map_options:
    if all(col in df.columns for col in rm.keys()):
        df = df.rename(columns=rm)
        break

expected_cols = ['contest', 'date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6']
df = df[[c for c in expected_cols if c in df.columns]].copy()

# robust date parsing for dataset (usually DD/MM/YYYY)
df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)

for c in ['contest', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

df = df.dropna(subset=['contest', 'date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6']).copy()
df['contest'] = df['contest'].astype(int)

# make draw set for quick ops
df['draw_set'] = df[['n1', 'n2', 'n3', 'n4', 'n5', 'n6']].apply(lambda r: set(int(x) for x in r), axis=1)

# ---------------- Utility functions ----------------
def wrap1to60(n: int) -> int:
    """Map any integer to 1..60 (0 -> 60)."""
    return ((int(n) - 1) % 60) + 1

def digits_of(n: int):
    return [int(d) for d in str(abs(int(n)))]

def reverse_number(n: int) -> int:
    return int(str(abs(int(n)))[::-1])

def two_digit_chunks_from_date(dt: datetime):
    dd = dt.day
    mm = dt.month
    yyyy = dt.year
    yy = yyyy % 100
    cc = yyyy // 100
    return dd, mm, yy, cc

def related_numbers_and_reasons(contest: int, date: datetime):
    """
    Build numbers in 1..60 from (contest, date).
    Returns: (set_of_numbers, {number: reason})
    """
    reasons = {}
    out = set()

    dd, mm, yy, cc = two_digit_chunks_from_date(date)
    yyyy = date.year
    weekday = date.weekday()  # 0=Mon..6=Sun

    cdigs = digits_of(contest)
    first = cdigs[0]
    last = cdigs[-1]
    sdigits = sum(cdigs)
    pdigits = np.prod(cdigs) if len(cdigs) > 1 else cdigs[0]

    def add(n, why):
        v = wrap1to60(n)
        out.add(v)
        reasons.setdefault(v, why)  # keep first reason for cleanliness

    # date parts
    add(dd, f"Day in date ({dd:02d})")
    add(mm, f"Month in date ({mm:02d})")
    add(yy, f"Year last two digits ({yy:02d})")
    add(cc, f"Century digits ({cc})")

    # basic ops on date
    add(dd + mm, "Sum of day+month")
    add(abs(dd - mm), "Abs diff day-month")
    add(dd * mm, "Product day*month")
    if mm != 0:
        add(dd // mm, "Day//Month")
    if dd != 0:
        add(mm // dd, "Month//Day")

    # year mixes
    add((yyyy // 100) + (yyyy % 100), "cc+yy")
    add((yyyy // 100) - (yyyy % 100), "cc-yy")
    add((yyyy % 100) + dd, "yy+day")
    add((yyyy % 100) + mm, "yy+month")

    # weekday mixes
    add(weekday + dd, "weekday+day")
    add(weekday + mm, "weekday+month")

    # contest digit features
    add(sdigits, "Sum contest digits")
    add(pdigits, "Product contest digits")
    add(first + last, "Sum first+last contest digits")
    add(first * last, "Product first+last digits")
    add(contest, "Contest number mapped")

    # reversals
    add(reverse_number(contest), f"Reverse of contest {contest}")
    add(int(str(dd)[::-1]), "Reverse of day")
    add(int(str(mm)[::-1]), "Reverse of month")
    add(int(str(yy).zfill(2)[::-1]), "Reverse of yy")
    add(int(str(cc)[::-1]), "Reverse of cc")

    # more mixes
    add(dd + yy, "Day+yy")
    add(mm + yy, "Month+yy")

    return out, reasons

def row_has_relation(row):
    rels, _ = related_numbers_and_reasons(int(row['contest']), pd.to_datetime(row['date']))
    return len(rels.intersection(row['draw_set'])) > 0

# “new bank”: only rows where at least one relation matched historically
filtered_df = df[df.apply(row_has_relation, axis=1)].copy().reset_index(drop=True)

def extract_features(contest: int, date: datetime):
    cdigs = digits_of(contest)
    first, last = cdigs[0], cdigs[-1]
    dd, mm, yy, cc = two_digit_chunks_from_date(date)
    return [
        dd, mm, date.year, date.weekday(),
        sum(cdigs),
        first * last if len(cdigs) > 1 else first,
        cc, yy, (cc + yy),
        abs(dd - mm), dd + mm, dd * (mm if mm != 0 else 1)
    ]

def encode_labels(draw_set: set):
    y = np.zeros(60, dtype=int)
    for n in draw_set:
        if 1 <= int(n) <= 60:
            y[int(n) - 1] = 1
    return y

# ---------------- Build training data ----------------
X, Y = [], []
for _, r in filtered_df.iterrows():
    X.append(extract_features(int(r['contest']), pd.to_datetime(r['date'])))
    Y.append(encode_labels(r['draw_set']))
X, Y = np.array(X), np.array(Y)

# fallback: if filter too small, use full bank
if len(X) < 200:
    X, Y = [], []
    for _, r in df.iterrows():
        X.append(extract_features(int(r['contest']), pd.to_datetime(r['date'])))
        Y.append(encode_labels(r['draw_set']))
    X, Y = np.array(X), np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, shuffle=True
)

clf = MultiOutputClassifier(RandomForestClassifier(
    n_estimators=200, random_state=42, n_jobs=-1
))
clf.fit(X_train, y_train)

y_pred_bin = clf.predict(X_test)

# ---------------- Evaluation ----------------
ham_loss = hamming_loss(y_test, y_pred_bin)
jac_micro = jaccard_score(y_test, y_pred_bin, average='micro', zero_division=0)
f1_micro = f1_score(y_test, y_pred_bin, average='micro', zero_division=0)

print("\n📊 Model Evaluation (Multi-label):")
print(f"Hamming Loss (lower is better): {ham_loss:.4f}")
print(f"Jaccard Score (micro):          {jac_micro:.4f}")
print(f"F1 Score (micro):               {f1_micro:.4f}")

# ---------------- Historical frequency ----------------
all_nums = [x for s in df['draw_set'] for x in s]
hist_counts = np.bincount(all_nums, minlength=61)[1:]  # 1..60
hist_freq = hist_counts / hist_counts.sum()
hist_freq = np.where(hist_freq == 0, 1e-6, hist_freq)

# ---------------- Prediction helpers ----------------
def parse_user_date(s: str) -> datetime:
    """
    Accept multiple user formats:
    - DD/MM/YYYY   - MM/DD/YYYY   - YYYY-MM-DD
    - DD-MM-YYYY   - MM-DD-YYYY   - Month DD, YYYY (e.g., December 31, 2025)
    Tries day-first then month-first safely.
    """
    fmts = [
        "%d/%m/%Y", "%m/%d/%Y",
        "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y",
        "%B %d, %Y", "%b %d, %Y"
    ]
    for f in fmts:
        try:
            return datetime.strptime(s.strip(), f)
        except Exception:
            continue
    # last resort: pandas is lenient but we want a datetime
    dt = pd.to_datetime(s, errors='coerce', dayfirst=True)
    if pd.isna(dt):
        raise ValueError("Unrecognized date format.")
    return dt.to_pydatetime()

def proba_for_input(contest_input: int, user_date_str: str):
    dt = parse_user_date(user_date_str)
    feats = np.array(extract_features(contest_input, dt)).reshape(1, -1)
    proba_list = clf.predict_proba(feats)
    p = np.array([pp[0, 1] for pp in proba_list])  # probability that each 1..60 is present
    return p, dt

def build_weights(p_ml: np.ndarray, contest_input: int, dt: datetime):
    """
    Combine ML probs + relation boosts + historical freq to make final weights.
    """
    w = p_ml.copy()
    rel_set, rel_reasons = related_numbers_and_reasons(contest_input, dt)

    # boost relation numbers
    for n in rel_set:
        w[n - 1] *= 1.5

    # blend with history
    w = 0.6 * w + 0.4 * hist_freq

    # normalize
    w = np.where(w <= 0, 1e-12, w)
    w = w / w.sum()
    return w, rel_set, rel_reasons

def sample_unique_games(weights: np.ndarray, rel_set: set, seed: int,
                        k_games=N_GAMES, k_per_game=NUMBERS_PER_GAME,
                        min_related=MIN_RELATED_PER_GAME, max_tries=20000):
    """
    Sample unique combos using weights, enforcing that each game includes at least
    `min_related` numbers from relation set.
    Deterministic given seed.
    """
    rng = np.random.default_rng(seed)
    games = set()
    tries = 0
    nums = np.arange(1, 61)

    # Pre-split pool to help constraint
    rel_nums = np.array(sorted(list(rel_set))) if rel_set else np.array([], dtype=int)

    while len(games) < k_games and tries < max_tries:
        # Try to ensure constraint by picking some from rel first (if available)
        if len(rel_nums) >= min_related and min_related > 0:
            pick_rel = rng.choice(rel_nums, size=min_related, replace=False)
            # fill remaining from global weights (excluding already picked)
            mask = np.ones_like(nums, dtype=bool)
            mask[pick_rel - 1] = False
            pool = nums[mask]
            # renormalize weights on remaining pool
            w_pool = weights[mask]
            w_pool = w_pool / w_pool.sum()
            pick_rest = rng.choice(pool, size=k_per_game - min_related, replace=False, p=w_pool)
            pick = np.concatenate([pick_rel, pick_rest])
        else:
            pick = rng.choice(nums, size=k_per_game, replace=False, p=weights)

        combo = tuple(sorted(int(x) for x in pick))
        # hard check (in case rel_set is small/empty)
        if len(set(combo).intersection(rel_set)) >= min_related:
            games.add(combo)
        tries += 1

    return [list(g) for g in sorted(games)]

# ---------------- CLI / Main ----------------
def main():
    try:
        # Inputs per client spec
        contest_str = input("Enter contest number (e.g., 2895): ").strip()
        if not contest_str.isdigit():
            raise ValueError("Contest number must be numeric (e.g., 2895).")
        contest_input = int(contest_str)  # 4-digit supported (or any integer)

        date_input = input("Enter contest date (supports DD/MM/YYYY, MM/DD/YYYY, or 'December 31, 2025'): ").strip()

        # probs + weights
        p_ml, dt = proba_for_input(contest_input, date_input)
        weights, rel_set, rel_reasons = build_weights(p_ml, contest_input, dt)

        # deterministic seed from contest+date
        seed = int(f"{contest_input}{dt.strftime('%Y%m%d')}")
        games = sample_unique_games(weights, rel_set, seed,
                                    k_games=N_GAMES, k_per_game=NUMBERS_PER_GAME,
                                    min_related=MIN_RELATED_PER_GAME)

        # ML ranking (for fallback reasons)
        ml_rank = np.argsort(-p_ml) + 1
        rank_pos = {int(n): i + 1 for i, n in enumerate(ml_rank)}

        def fallback_reason_for(n):
            r = rank_pos.get(int(n), 9999)
            if r <= 10:
                return f"High ML likelihood (top {r})"
            elif r <= 20:
                return f"Moderate ML likelihood (top {r})"
            else:
                return f"Weighted by history/ML (rank {r})"

        # Print & save
        rows = []
        print("\n🎯 Predicted 50 Games (each includes relation-based numbers):\n")
        for i, g in enumerate(games, 1):
            reasons_inline = []
            for n in g:
                if n in rel_reasons:
                    reasons_inline.append(f"{n:02d}: {rel_reasons[n]}")
                elif n in rel_set:
                    reasons_inline.append(f"{n:02d}: Related to date/contest by rule")
                else:
                    reasons_inline.append(f"{n:02d}: {fallback_reason_for(n)}")
            print(f"Game {i:02d}: {' '.join(f'{x:02d}' for x in g)}")
            print("  Reasons: " + " | ".join(reasons_inline))
            rows.append({
                "Game": i,
                "Numbers": " ".join(f"{x:02d}" for x in g),
                "Reasons": " | ".join(reasons_inline)
            })

        out_df = pd.DataFrame(rows)
        out_df.to_csv("predicted_games.csv", index=False)
        print("\n✅ Results saved to predicted_games.csv")

        # brief help text for client
        print("\nℹ️ Input help:")
        print("  - Contest: any integer (e.g., 2895).")
        print("  - Date: accepts DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD, DD-MM-YYYY, or 'December 31, 2025'.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
