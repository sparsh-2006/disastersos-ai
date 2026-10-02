import pandas as pd
import streamlit as st
from pipeline import analyze, mark_duplicates
from data import D, GAZ

st.set_page_config(page_title="DisasterSOS AI", page_icon="🆘", layout="wide")
st.title("🆘 DisasterSOS AI")
st.caption("Decision support for human responders. All values are AI predictions, not verified facts.")

ICON = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
tab1, tab2 = st.tabs(["Analyze a post", "Dashboard"])

with tab1:
    txt = st.text_area("Enter disaster post:", height=130,
        value="Please help! We are trapped in our house. There are 6 people and my mother is injured. Water is entering the house rapidly.")
    if st.button("Analyze Emergency", type="primary") and txt.strip():
        r = analyze(txt)
        st.subheader("AI ANALYSIS")
        a, b, c = st.columns(3)
        a.metric("Emotion", r["emotion"]); a.metric("Sentiment", r["sentiment"]); a.metric("Intent", r["intent"])
        b.metric("Urgency", r["urgency"].upper()); b.metric("Help Required", r["help"])
        b.metric("Medical Emergency", r["medical"].upper())
        c.metric("People Affected", r["people"] or "Not mentioned"); c.metric("Location", r["location"])
        c.metric("Priority", f"{ICON[r['level']]} {r['score']}/100 ({r['level']})")
        st.markdown("**Reasons**\n" + "\n".join(f"- ✓ {x}" for x in r["reasons"]))
        st.warning(f"⚠ Potentially {r['level'].lower()} report, requires human verification")

@st.cache_data
def build_feed():
    rows = []
    for i, (text, disaster, _, _) in enumerate(D):
        rows.append({"post_id": 101 + i, **analyze(text), "disaster": disaster})
    return mark_duplicates(pd.DataFrame(rows))

with tab2:
    df = build_feed()

    cols = st.columns(5)
    cols[0].metric("Total Posts", len(df))
    for col, lvl in zip(cols[1:], ICON):
        col.metric(f"{ICON[lvl]} {lvl.title()}", int((df.level == lvl).sum()))

    f1, f2, f3, f4 = st.columns(4)
    lv = f1.multiselect("Priority", list(ICON))
    em = f2.multiselect("Emotion", sorted(df.emotion.unique()))
    dz = f3.multiselect("Disaster", sorted(df.disaster.unique()))
    lc = f4.multiselect("Location", sorted(df.location.unique()))
    hp = st.multiselect("Help category", sorted({h for s in df.help for h in s.split(", ")}))
    med = st.checkbox("Medical emergencies only")

    v = df
    if lv: v = v[v.level.isin(lv)]
    if em: v = v[v.emotion.isin(em)]
    if dz: v = v[v.disaster.isin(dz)]
    if lc: v = v[v.location.isin(lc)]
    if hp: v = v[v.help.map(lambda s: any(h in s.split(", ") for h in hp))]
    if med: v = v[v.medical == "Yes"]
    v = v.sort_values("score", ascending=False)

    st.markdown("#### Priority feed (AI-ranked, unverified)")
    st.dataframe(v[["post_id", "text", "emotion", "intent", "urgency", "help", "location",
                    "medical", "people", "score", "level", "duplicates"]],
                 use_container_width=True, hide_index=True)
    st.caption("`duplicates` = number of similar earlier posts (TF-IDF cosine ≥ 0.6).")

    m = v[v.location.isin(GAZ)].drop_duplicates("dup_group").copy()
    if len(m):
        m["lat"] = m.location.map(lambda x: GAZ[x][0])
        m["lon"] = m.location.map(lambda x: GAZ[x][1])
        m["color"] = m.level.map({"CRITICAL": "#e53935", "HIGH": "#fb8c00",
                                  "MEDIUM": "#fdd835", "LOW": "#43a047"})
        st.markdown("#### Map: 🧪 DEMO DATA (approximate coordinates, not verified)")
        st.map(m, latitude="lat", longitude="lon", color="color", size=120)
