import re
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
from data import D, GAZ

# ---------- preprocessing
def clean(t):
    t = re.sub(r"http\S+|@\w+|#", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# ---------- classifiers (intent + urgency)
def make_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )

@lru_cache(maxsize=1)
def get_models():
    X = [clean(r[0]) for r in D]
    intent_m = make_model().fit(X, [r[2] for r in D])
    urg_m = make_model().fit(X, [r[3] for r in D])
    return intent_m, urg_m

# ---------- keyword rules
EMO = {
 "Panic": r"help|trapped|immediately|rapid|urgent|hurry|now|screaming|stuck|send",
 "Fear": r"scared|afraid|worried|rising|danger|warning|collapse|aftershock|may breach",
 "Sadness": r"lost|died|dead|crying|missing|victims|gone",
 "Anger": r"angry|useless|ignored|shame|furious",
 "Frustration": r"no one has come|nobody came|still waiting|no response|tired|for two days",
 "Relief": r"safe|rescued|thank|reopened|cleared|restored",
 "Hope": r"pray|hope|stay strong|love and strength|will be fine",
}
POS = r"safe|thank|rescued|restored|reopened|cleared|going down|easing|reduced|strength|returned"
NEG = r"trapped|injur|collapse|died|dead|blocked|rising|stranded|missing|bleeding|help|angry|warning|fallen|buried|danger"
HELP = {
 "Rescue": r"trapped|rescue|stuck|stranded|rubble|boat|buried|roof",
 "Medical": r"injur|bleed|ambulance|doctor|medicine|blood|labour|heart|hurt|pregnan|unconscious",
 "Food": r"food|hungry|ration",
 "Water": r"drinking water|no water",
 "Shelter": r"shelter|homeless|exposed to rain",
 "Transportation": r"transport|evacuat",
 "Infrastructure": r"bridge|road|electricity|power|embankment|building",
 "Missing Person": r"missing|find him|find her|last seen",
}
VULN = {
 "Children": r"child|kid|baby|infant",
 "Elderly": r"elderly|old man|old woman|grandm|grandf",
 "Pregnant": r"pregnan|labour",
 "Disabled": r"disabled|wheelchair",
}
DANGER = r"trapped|rubble|buried|bleeding|collapsed|rapidly|rising quickly|labour|heart problem|unconscious"
NUMW = dict(zip("one two three four five six seven eight nine ten".split(), range(1, 11)))

def get_emotion(low):
    for emo, pat in EMO.items():
        need = 2 if emo == "Panic" else 1
        if len(re.findall(pat, low)) >= need:
            return emo
    return "Neutral"

def get_sentiment(low):
    p, n = len(re.findall(POS, low)), len(re.findall(NEG, low))
    return "Negative" if n > p else "Positive" if p > n else "Neutral"

def get_help(low):
    return [h for h, pat in HELP.items() if re.search(pat, low)] or ["General Information"]

def get_vulnerable(low):
    return [v for v, pat in VULN.items() if re.search(pat, low)]

def get_people(low):
    for pat in (r"family of (\d+)", r"(\d+)\s+(?:people|persons|members|villagers|of us|children|kids)", r"there are (\d+)"):
        m = re.search(pat, low)
        if m:
            return int(m.group(1))
    m = re.search(r"\b(%s)\s+(?:people|persons|members|of us)" % "|".join(NUMW), low)
    return NUMW[m.group(1)] if m else None

def get_location(text):
    for name in GAZ:
        if name.lower() in text.lower():
            return name
    m = re.search(r"\b(?:near|at|in|around)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", text)
    if m and m.group(1).split()[0] not in {"The", "Our", "My", "We", "Please", "Flood", "A", "An"}:
        return m.group(1)
    return "Not mentioned"   # never invent a location

# ---------- explainable priority score (weights sum to 100)
W = {"urgency": 25, "intent": 20, "medical": 20, "people_vuln": 15, "help_type": 10, "emotion": 10}

def analyze(text):
    t = clean(text)
    low = t.lower()
    intent_m, urg_m = get_models()
    intent, urgency = intent_m.predict([t])[0], urg_m.predict([t])[0]

    # Safety rule: never under-rate danger keywords in a help request
    if intent == "Emergency Help" and re.search(DANGER, low):
        urgency = "Critical"

    emotion, sentiment = get_emotion(low), get_sentiment(low)
    helps, vuln = get_help(low), get_vulnerable(low)
    medical = "Medical" in helps
    people, loc = get_people(low), get_location(text)

    s_urg = {"Critical": 1, "High": .6, "Low": .05}[urgency] * W["urgency"]
    s_int = {"Emergency Help": 1, "Warning": .5, "Information": .15, "Status Update": .05, "Other": 0}[intent] * W["intent"]
    s_med = W["medical"] if medical else 0
    s_ppl = min(people or 0, 6) / 6 * 8 + (7 if vuln else 0)
    s_help = (1 if "Rescue" in helps else .8 if medical else .5 if helps != ["General Information"] else 0) * W["help_type"]
    s_emo = {"Panic": 1, "Fear": .8, "Sadness": .5, "Anger": .5, "Frustration": .6}.get(emotion, .1) * W["emotion"]
    score = s_urg + s_int + s_med + s_ppl + s_help + s_emo
    if urgency == "Critical" and intent == "Emergency Help":
        score = max(score, 80)
    score = int(round(min(score, 100)))
    level = "CRITICAL" if score >= 80 else "HIGH" if score >= 55 else "MEDIUM" if score >= 30 else "LOW"

    reasons = []
    if re.search(DANGER, low): reasons.append("Immediate danger indicators in text")
    if "Rescue" in helps: reasons.append("Rescue requested / people trapped or stranded")
    if medical: reasons.append("Medical emergency detected")
    if people and people > 1: reasons.append(f"Multiple people affected ({people})")
    if vuln: reasons.append("Vulnerable people: " + ", ".join(vuln))
    if loc != "Not mentioned": reasons.append(f"Location identified: {loc}")
    if intent != "Emergency Help": reasons.append(f"Classified as '{intent}', not a direct help request")

    return dict(text=text, emotion=emotion, sentiment=sentiment, intent=intent, urgency=urgency,
                help=", ".join(helps), location=loc, medical="Yes" if medical else "No",
                people=people, vulnerable=", ".join(vuln) or "-", score=score, level=level, reasons=reasons)

# ---------- duplicate detection
def mark_duplicates(df, thr=0.6):
    vec = TfidfVectorizer().fit_transform(df["text"].map(clean))
    sim = cosine_similarity(vec)
    group = list(range(len(df)))
    for i in range(len(df)):
        for j in range(i):
            if sim[i, j] >= thr and group[j] == j:
                group[i] = j
                break
    df["dup_group"] = group
    df["duplicates"] = df.groupby("dup_group")["text"].transform("count") - 1
    return df
