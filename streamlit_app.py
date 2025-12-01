import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import numpy as np

# ==============================
# Global UI configuration
# ==============================

st.set_page_config(
    page_title="String Matching Algorithms Benchmark",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle, premium-looking dark-on-light theme overrides
PREMIUM_CSS = """
<style>
/* Global */
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
}

.main {
    padding: 2rem 3rem 3rem 3rem;
    background: radial-gradient(circle at top left, #0f172a 0, #020617 40%, #020617 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #020617 40%, #020617 100%);
}

/* Cards */
.premium-card {
    padding: 1.2rem 1.4rem;
    border-radius: 0.9rem;
    background: rgba(15,23,42,0.9);
    border: 1px solid rgba(148,163,184,0.18);
    box-shadow: 0 18px 45px rgba(15,23,42,0.55);
}

.premium-card-light {
    padding: 1.1rem 1.3rem;
    border-radius: 0.9rem;
    background: rgba(15,23,42,0.9);
    border: 1px solid rgba(148,163,184,0.18);
    box-shadow: 0 18px 45px rgba(15,23,42,0.55);
}

.metric-card {
    padding: 0.9rem 1.1rem;
    border-radius: 0.8rem;
    background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,64,175,0.88));
    border: 1px solid rgba(129,140,248,0.55);
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    font-weight: 650 !important;
    letter-spacing: -0.02em;
}

.stMarkdown, .stText, p {
    color: #e5e7eb !important;
}

/* Tabs */
div[data-baseweb="tab-list"] {
    gap: 0.5rem;
}

button[data-baseweb="tab"] {
    border-radius: 999px !important;
    padding: 0.5rem 1rem !important;
    background-color: rgba(15,23,42,0.7) !important;
    border: 1px solid rgba(148,163,184,0.55) !important;
    color: #e5e7eb !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #22c55e) !important;
    color: #0b1120 !important;
    border-color: transparent !important;
}

/* Sidebar widgets */
section[data-testid="stSidebar"] .stMarkdown, 
section[data-testid="stSidebar"] p, 
section[data-testid="stSidebar"] label {
    color: #e5e7eb !important;
}

section[data-testid="stSidebar"] .stTextInput > div > div > input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stMultiSelect > div > div {
    background-color: #020617 !important;
    border-radius: 0.5rem;
    border: 1px solid rgba(148,163,184,0.55);
    color: #e5e7eb !important;
}

/* Buttons */
button[kind="primary"] {
    border-radius: 999px !important;
    padding: 0.4rem 1.1rem !important;
    background: linear-gradient(135deg, #4f46e5, #22c55e) !important;
    border: none !important;
    color: #0f172a !important;
    font-weight: 600 !important;
}

button[kind="secondary"] {
    border-radius: 999px !important;
}

/* Tables */
.stDataFrame table {
    border-radius: 0.8rem;
    overflow: hidden;
}

.stDataFrame table thead tr {
    background: #020617 !important;
}

.stDataFrame table tbody tr:nth-child(even) {
    background: rgba(15,23,42,0.55) !important;
}

.stDataFrame table tbody tr:nth-child(odd) {
    background: rgba(15,23,42,0.85) !important;
}
</style>
"""

st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

# Algorithm implementations
def naive_search(text, pattern):
    matches, comparisons = [], 0
    n, m = len(text), len(pattern)
    start = time.perf_counter()
    
    for i in range(n - m + 1):
        j = 0
        while j < m:
            comparisons += 1
            if text[i + j] != pattern[j]:
                break
            j += 1
        if j == m:
            matches.append(i)
    
    elapsed = (time.perf_counter() - start) * 1000
    return {'matches': matches, 'time_ms': elapsed, 'comparisons': comparisons, 'space': 4}

def kmp_search(text, pattern):
    matches, comparisons = [], 0
    n, m = len(text), len(pattern)
    
    # Build LPS
    lps = [0] * m
    length, i = 0, 1
    while i < m:
        comparisons += 1
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length != 0:
            length = lps[length - 1]
        else:
            lps[i] = 0
            i += 1
    
    start = time.perf_counter()
    i = j = 0
    while i < n:
        comparisons += 1
        if pattern[j] == text[i]:
            i += 1
            j += 1
            if j == m:
                matches.append(i - j)
                j = lps[j - 1]
        elif j != 0:
            j = lps[j - 1]
        else:
            i += 1
    
    elapsed = (time.perf_counter() - start) * 1000
    return {'matches': matches, 'time_ms': elapsed, 'comparisons': comparisons, 'space': m * 4}

def rabin_karp_search(text, pattern):
    matches, comparisons = [], 0
    n, m = len(text), len(pattern)
    base, mod = 256, 101
    
    if m > n:
        return {'matches': [], 'time_ms': 0, 'comparisons': 0, 'space': 12}
    
    start = time.perf_counter()
    pattern_hash = text_hash = h = 0
    
    for i in range(m):
        pattern_hash = (base * pattern_hash + ord(pattern[i])) % mod
        text_hash = (base * text_hash + ord(text[i])) % mod
        if i > 0:
            h = (h * base) % mod
        else:
            h = 1
    
    for i in range(n - m + 1):
        if pattern_hash == text_hash:
            match = True
            for j in range(m):
                comparisons += 1
                if text[i + j] != pattern[j]:
                    match = False
                    break
            if match:
                matches.append(i)
        
        if i < n - m:
            text_hash = (base * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % mod
            if text_hash < 0:
                text_hash += mod
    
    elapsed = (time.perf_counter() - start) * 1000
    return {'matches': matches, 'time_ms': elapsed, 'comparisons': comparisons, 'space': 12}

def boyer_moore_horspool_search(text, pattern):
    matches, comparisons = [], 0
    n, m = len(text), len(pattern)
    
    if m > n:
        return {'matches': [], 'time_ms': 0, 'comparisons': 0, 'space': 256 * 4}
    
    bad_char = {c: m for c in set(text)}
    for i in range(m - 1):
        bad_char[pattern[i]] = m - 1 - i
    
    start = time.perf_counter()
    i = 0
    while i <= n - m:
        j = m - 1
        while j >= 0:
            comparisons += 1
            if pattern[j] != text[i + j]:
                break
            j -= 1
        
        if j < 0:
            matches.append(i)
            i += 1
        else:
            i += bad_char.get(text[i + m - 1], m)
    
    elapsed = (time.perf_counter() - start) * 1000
    return {'matches': matches, 'time_ms': elapsed, 'comparisons': comparisons, 'space': 256 * 4}

# Predefined test cases
PREDEFINED_TESTS = {
    "DNA Sequence": {
        "text": "GATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC",
        "pattern": "GATCGATC",
        "description": "Repetitive DNA sequence with multiple matches"
    },
    "Worst Case - Naive": {
        "text": "a" * 100,
        "pattern": "a" * 9 + "b",
        "description": "Worst case for Naive algorithm - pattern almost matches at every position"
    },
    "Best Case - Boyer-Moore": {
        "text": "b" * 100,
        "pattern": "a" * 10,
        "description": "Best case for Boyer-Moore - immediate mismatch at every position"
    },
    "Natural Language": {
        "text": "the quick brown fox jumps over the lazy dog the quick brown fox",
        "pattern": "quick",
        "description": "Natural English text with common word pattern"
    },
    "Code Search": {
        "text": "def function(): return value; def another_function(): return value;",
        "pattern": "return",
        "description": "Searching for keyword in code"
    },
    "No Match": {
        "text": "abcdefghijklmnopqrstuvwxyz" * 4,
        "pattern": "xyz123",
        "description": "Pattern not present in text"
    }
}

# Algorithm recommendation system
def recommend_algorithm(n, m, text_type, expected_matches):
    recommendations = []
    
    # Analyze characteristics
    ratio = n / m if m > 0 else 0
    
    if text_type == "repetitive" or expected_matches == "many":
        recommendations.append({
            "algorithm": "KMP",
            "score": 95,
            "reason": "Optimal for repetitive text with many matches. Linear time O(n+m) guaranteed."
        })
        recommendations.append({
            "algorithm": "Boyer-Moore-Horspool",
            "score": 70,
            "reason": "Good for longer patterns but may not skip as much in repetitive text."
        })
        recommendations.append({
            "algorithm": "Rabin-Karp",
            "score": 60,
            "reason": "Decent performance but hash collisions may occur in repetitive text."
        })
        recommendations.append({
            "algorithm": "Naive",
            "score": 30,
            "reason": "Poor choice - O(n×m) worst case likely with repetitive patterns."
        })
    elif m > 10 and ratio > 100:
        recommendations.append({
            "algorithm": "Boyer-Moore-Horspool",
            "score": 95,
            "reason": "Best for long patterns in large text. Can skip many positions."
        })
        recommendations.append({
            "algorithm": "KMP",
            "score": 85,
            "reason": "Reliable linear time performance for any input."
        })
        recommendations.append({
            "algorithm": "Rabin-Karp",
            "score": 75,
            "reason": "Good average case but preprocessing overhead for long patterns."
        })
        recommendations.append({
            "algorithm": "Naive",
            "score": 40,
            "reason": "Simple but inefficient for long patterns."
        })
    elif m <= 3:
        recommendations.append({
            "algorithm": "Naive",
            "score": 85,
            "reason": "Simple and efficient for very short patterns. Low overhead."
        })
        recommendations.append({
            "algorithm": "Boyer-Moore-Horspool",
            "score": 80,
            "reason": "Good performance even with short patterns."
        })
        recommendations.append({
            "algorithm": "KMP",
            "score": 70,
            "reason": "Preprocessing overhead may not be worth it for short patterns."
        })
        recommendations.append({
            "algorithm": "Rabin-Karp",
            "score": 65,
            "reason": "Hash computation overhead for short patterns."
        })
    else:
        recommendations.append({
            "algorithm": "KMP",
            "score": 90,
            "reason": "Balanced choice with guaranteed O(n+m) performance."
        })
        recommendations.append({
            "algorithm": "Boyer-Moore-Horspool",
            "score": 85,
            "reason": "Good average case performance for medium patterns."
        })
        recommendations.append({
            "algorithm": "Rabin-Karp",
            "score": 75,
            "reason": "Decent performance with low space usage."
        })
        recommendations.append({
            "algorithm": "Naive",
            "score": 50,
            "reason": "Simple but not optimal for medium-sized patterns."
        })
    
    return sorted(recommendations, key=lambda x: x['score'], reverse=True)

# Statistical analysis function
def calculate_statistics(df):
    stats = df.groupby(['Algorithm', 'Text Type', 'n', 'm']).agg({
        'Time (ms)': ['mean', 'median', 'std', 'min', 'max'],
        'Comparisons': ['mean', 'median', 'std', 'min', 'max']
    }).reset_index()
    
    stats.columns = ['Algorithm', 'Text Type', 'n', 'm', 
                     'Time Mean', 'Time Median', 'Time Std', 'Time Min', 'Time Max',
                     'Comp Mean', 'Comp Median', 'Comp Std', 'Comp Min', 'Comp Max']
    return stats

# Step-by-step visualization generators
def naive_steps(text, pattern):
    n, m = len(text), len(pattern)
    comparisons = 0
    for i in range(n - m + 1):
        for j in range(m):
            comparisons += 1
            yield {
                'step': comparisons,
                'text_pos': i,
                'pattern_pos': j,
                'comparing': (i + j, j),
                'match': text[i + j] == pattern[j],
                'message': f"Comparing text[{i+j}]='{text[i+j]}' with pattern[{j}]='{pattern[j]}'",
                'found_match': False
            }
            if text[i + j] != pattern[j]:
                yield {
                    'step': comparisons,
                    'text_pos': i,
                    'pattern_pos': j,
                    'comparing': None,
                    'match': False,
                    'message': f"Mismatch! Shifting pattern by 1 position",
                    'found_match': False
                }
                break
        else:
            yield {
                'step': comparisons,
                'text_pos': i,
                'pattern_pos': m,
                'comparing': None,
                'match': True,
                'message': f"✅ Match found at position {i}!",
                'found_match': True
            }

def kmp_steps(text, pattern):
    n, m = len(text), len(pattern)
    
    # Build LPS
    lps = [0] * m
    length, i = 0, 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length != 0:
            length = lps[length - 1]
        else:
            lps[i] = 0
            i += 1
    
    yield {
        'step': 0,
        'text_pos': 0,
        'pattern_pos': 0,
        'comparing': None,
        'match': None,
        'message': f"LPS Array computed: {lps}",
        'found_match': False,
        'lps': lps
    }
    
    i = j = 0
    step = 0
    while i < n:
        step += 1
        if pattern[j] == text[i]:
            yield {
                'step': step,
                'text_pos': i - j,
                'pattern_pos': j,
                'comparing': (i, j),
                'match': True,
                'message': f"Match: text[{i}]='{text[i]}' == pattern[{j}]='{pattern[j]}'",
                'found_match': False,
                'lps': lps
            }
            i += 1
            j += 1
            if j == m:
                yield {
                    'step': step,
                    'text_pos': i - j,
                    'pattern_pos': j,
                    'comparing': None,
                    'match': True,
                    'message': f"✅ Match found at position {i - j}! Using LPS to continue",
                    'found_match': True,
                    'lps': lps
                }
                j = lps[j - 1]
        else:
            yield {
                'step': step,
                'text_pos': i - j,
                'pattern_pos': j,
                'comparing': (i, j),
                'match': False,
                'message': f"Mismatch: text[{i}]='{text[i]}' != pattern[{j}]='{pattern[j]}'. Using LPS[{j}]={lps[j-1] if j > 0 else 0}",
                'found_match': False,
                'lps': lps
            }
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

def rabin_karp_steps(text, pattern):
    n, m = len(text), len(pattern)
    base, mod = 256, 101
    
    pattern_hash = text_hash = h = 0
    for i in range(m):
        pattern_hash = (base * pattern_hash + ord(pattern[i])) % mod
        text_hash = (base * text_hash + ord(text[i])) % mod
        if i > 0:
            h = (h * base) % mod
        else:
            h = 1
    
    yield {
        'step': 0,
        'text_pos': 0,
        'pattern_pos': 0,
        'comparing': None,
        'match': None,
        'message': f"Pattern hash: {pattern_hash}, Initial window hash: {text_hash}",
        'found_match': False,
        'pattern_hash': pattern_hash,
        'text_hash': text_hash
    }
    
    step = 0
    for i in range(n - m + 1):
        step += 1
        if pattern_hash == text_hash:
            yield {
                'step': step,
                'text_pos': i,
                'pattern_pos': 0,
                'comparing': None,
                'match': None,
                'message': f"Hash match! Verifying character by character...",
                'found_match': False,
                'pattern_hash': pattern_hash,
                'text_hash': text_hash
            }
            match = True
            for j in range(m):
                if text[i + j] != pattern[j]:
                    match = False
                    break
            if match:
                yield {
                    'step': step,
                    'text_pos': i,
                    'pattern_pos': m,
                    'comparing': None,
                    'match': True,
                    'message': f"✅ Match found at position {i}!",
                    'found_match': True,
                    'pattern_hash': pattern_hash,
                    'text_hash': text_hash
                }
            else:
                yield {
                    'step': step,
                    'text_pos': i,
                    'pattern_pos': 0,
                    'comparing': None,
                    'match': False,
                    'message': f"False positive - hash collision!",
                    'found_match': False,
                    'pattern_hash': pattern_hash,
                    'text_hash': text_hash
                }
        
        if i < n - m:
            text_hash = (base * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % mod
            if text_hash < 0:
                text_hash += mod
            yield {
                'step': step,
                'text_pos': i + 1,
                'pattern_pos': 0,
                'comparing': None,
                'match': None,
                'message': f"Rolling hash: new hash = {text_hash}",
                'found_match': False,
                'pattern_hash': pattern_hash,
                'text_hash': text_hash
            }

def bmh_steps(text, pattern):
    n, m = len(text), len(pattern)
    
    bad_char = {c: m for c in set(text)}
    for i in range(m - 1):
        bad_char[pattern[i]] = m - 1 - i
    
    yield {
        'step': 0,
        'text_pos': 0,
        'pattern_pos': 0,
        'comparing': None,
        'match': None,
        'message': f"Bad character table computed",
        'found_match': False,
        'bad_char': bad_char
    }
    
    i = 0
    step = 0
    while i <= n - m:
        j = m - 1
        while j >= 0:
            step += 1
            yield {
                'step': step,
                'text_pos': i,
                'pattern_pos': j,
                'comparing': (i + j, j),
                'match': pattern[j] == text[i + j],
                'message': f"Comparing text[{i+j}]='{text[i+j]}' with pattern[{j}]='{pattern[j]}' (right to left)",
                'found_match': False,
                'bad_char': bad_char
            }
            if pattern[j] != text[i + j]:
                shift = bad_char.get(text[i + m - 1], m)
                yield {
                    'step': step,
                    'text_pos': i,
                    'pattern_pos': j,
                    'comparing': None,
                    'match': False,
                    'message': f"Mismatch! Shifting by {shift} using bad char '{text[i + m - 1]}'",
                    'found_match': False,
                    'bad_char': bad_char
                }
                break
            j -= 1
        
        if j < 0:
            yield {
                'step': step,
                'text_pos': i,
                'pattern_pos': 0,
                'comparing': None,
                'match': True,
                'message': f"✅ Match found at position {i}!",
                'found_match': True,
                'bad_char': bad_char
            }
            i += 1
        else:
            i += bad_char.get(text[i + m - 1], m)

def render_visualization(text, pattern, state):
    """Render the current state of pattern matching"""
    n, m = len(text), len(pattern)
    text_pos = state.get('text_pos', 0)
    comparing = state.get('comparing', None)
    found_match = state.get('found_match', False)
    
    # Create HTML visualization
    html = '<div style="font-family: monospace; font-size: 16px; line-height: 2;">'
    
    # Render text
    html += '<div><b>Text:</b> '
    for i, char in enumerate(text):
        color = '#ddd'
        if comparing and i == comparing[0]:
            color = '#fbbf24' if state['match'] else '#ef4444'
        elif found_match and text_pos <= i < text_pos + m:
            color = '#10b981'
        html += f'<span style="background-color: {color}; padding: 2px 6px; margin: 1px; border-radius: 3px;">{char}</span>'
    html += '</div>'
    
    # Render pattern alignment
    html += '<div style="margin-top: 10px;"><b>Pattern:</b> '
    html += '&nbsp;' * 10 * text_pos
    for j, char in enumerate(pattern):
        color = '#ddd'
        if comparing and j == comparing[1]:
            color = '#fbbf24' if state['match'] else '#ef4444'
        elif found_match:
            color = '#10b981'
        html += f'<span style="background-color: {color}; padding: 2px 6px; margin: 1px; border-radius: 3px;">{char}</span>'
    html += '</div></div>'
    
    return html

# Streamlit App Title & Hero Section
hero_left, hero_right = st.columns([3, 2])
with hero_left:
    st.markdown("### 🔍 String Matching Algorithms Benchmark")
    st.markdown(
        "A premium, interactive lab to **benchmark** and **visualize** classic string "
        "matching algorithms – Naive, KMP, Rabin-Karp, and Boyer-Moore-Horspool – "
        "across realistic scenarios."
    )
with hero_right:
    st.markdown(
        """
        <div class="premium-card">
            <div style="font-size:0.8rem; text-transform:uppercase; letter-spacing:0.16em; color:#9ca3af;">
                Overview
            </div>
            <div style="margin-top:0.4rem; font-size:0.95rem; color:#e5e7eb;">
                • Compare execution time, comparisons, and space usage<br/>
                • Explore worst-, average-, and best-case inputs<br/>
                • Step visually through each algorithm’s internal logic
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Sidebar configuration
st.sidebar.markdown("### ⚙️ Benchmark Configuration")

input_mode = st.sidebar.radio(
    "Input Mode",
    ["Synthetic", "Manual", "Predefined Tests"],
    help="Synthetic is best for large-scale benchmarking; Manual and Predefined are ideal for focused experiments.",
)

# Robust input parsing helpers
def _parse_int_list(raw: str, field_label: str):
    try:
        values = [int(x.strip()) for x in raw.split(",") if x.strip()]
        values = [v for v in values if v > 0]
    except ValueError:
        st.sidebar.error(f"'{field_label}' must be a comma-separated list of positive integers.")
        return []
    if not values:
        st.sidebar.error(f"Please provide at least one positive integer for '{field_label}'.")
    return values

text = ""
pattern = ""

if input_mode == "Manual":
    text = st.sidebar.text_area("Text", "abcabcabcabc", height=120)
    pattern = st.sidebar.text_input("Pattern", "abc")

    if not text.strip() or not pattern:
        st.sidebar.warning("Enter both **Text** and **Pattern** to run a manual benchmark.")

    text_sizes = [len(text)]
    pattern_sizes = [len(pattern)]
    text_types = ["manual"]
    runs = 1
elif input_mode == "Predefined Tests":
    selected_test = st.sidebar.selectbox("Select Test Case", list(PREDEFINED_TESTS.keys()))
    test_info = PREDEFINED_TESTS[selected_test]
    st.sidebar.info(f"**Description:** {test_info['description']}")
    text = test_info["text"]
    pattern = test_info["pattern"]
    text_sizes = [len(text)]
    pattern_sizes = [len(pattern)]
    text_types = ["predefined"]
    runs = st.sidebar.slider("Number of Runs", 1, 10, 5)
else:
    raw_text_sizes = st.sidebar.text_input("Text Sizes (comma-separated)", "1000,5000,10000")
    text_sizes = _parse_int_list(raw_text_sizes, "Text Sizes")

    raw_pattern_sizes = st.sidebar.text_input("Pattern Sizes (comma-separated)", "3,10,50")
    pattern_sizes = _parse_int_list(raw_pattern_sizes, "Pattern Sizes")

    text_types = st.sidebar.multiselect(
        "Text Types",
        ["random", "repetitive"],
        default=["random", "repetitive"],
        help="Random ≈ typical text; Repetitive ≈ worst-case for some algorithms.",
    )
    runs = st.sidebar.slider("Number of Runs", 1, 10, 3)

algorithms = st.sidebar.multiselect(
    "Select Algorithms",
    ["Naive", "KMP", "Rabin-Karp", "Boyer-Moore-Horspool"],
    default=["Naive", "KMP", "Rabin-Karp", "Boyer-Moore-Horspool"]
)

# Algorithm Recommendation
if input_mode != "Synthetic":
    with st.sidebar.expander("🧠 Algorithm Recommendation", expanded=False):
        n_val = text_sizes[0] if text_sizes else 1000
        m_val = pattern_sizes[0] if pattern_sizes else 10
        text_type_val = text_types[0] if text_types else "random"
        expected = st.selectbox("Expected Matches", ["few", "many"], index=0)

        if st.button("Get Recommendation"):
            recs = recommend_algorithm(n_val, m_val, text_type_val, expected)
            st.markdown("### 🏆 Recommendations")
            for i, rec in enumerate(recs, 1):
                st.markdown(f"**{i}. {rec['algorithm']}** &nbsp; | &nbsp; Score: `{rec['score']}/100`")
                st.caption(rec["reason"])
                st.progress(rec["score"] / 100)

run_benchmark_clicked = st.sidebar.button("🚀 Run Benchmark", type="primary")

if run_benchmark_clicked:
    # Basic guard rails before running heavy loops
    if not algorithms:
        st.error("Please select at least **one algorithm** to benchmark.")
    elif input_mode == "Synthetic" and (not text_sizes or not pattern_sizes or not text_types):
        st.error("Please provide valid **text sizes**, **pattern sizes**, and **text types** for synthetic benchmarking.")
    elif input_mode in ("Manual", "Predefined Tests") and (len(pattern) == 0 or len(text) == 0):
        st.error("Text and Pattern must be non-empty to run the benchmark.")
    else:
        algo_map = {
            "Naive": naive_search,
            "KMP": kmp_search,
            "Rabin-Karp": rabin_karp_search,
            "Boyer-Moore-Horspool": boyer_moore_horspool_search,
        }

        results = []
        progress_bar = st.progress(0)
        total_tests = len(text_types) * len(text_sizes) * len(pattern_sizes) * len(algorithms) * runs
        current = 0

        for text_type in text_types:
            for n in text_sizes:
                for m in pattern_sizes:
                    if m > n:
                        continue

                    # Generate text and pattern
                    if input_mode == "Manual" or input_mode == "Predefined Tests":
                        test_text, test_pattern = text, pattern
                    elif text_type == "random":
                        test_text = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=n))
                        start_pos = random.randint(0, n - m)
                        test_pattern = test_text[start_pos:start_pos + m]
                    else:
                        test_text = 'a' * n
                        test_pattern = 'a' * (m - 1) + 'b'

                    # Store sample for first test case
                    if len(results) == 0:
                        st.session_state['sample_text'] = test_text[:100] + ('...' if len(test_text) > 100 else '')
                        st.session_state['sample_pattern'] = test_pattern
                        st.session_state['text_type'] = text_type

                    for run in range(runs):
                        for algo_name in algorithms:
                            result = algo_map[algo_name](test_text, test_pattern)
                            results.append({
                                'Algorithm': algo_name,
                                'Text Type': text_type,
                                'n': n,
                                'm': m,
                                'Run': run + 1,
                                'Time (ms)': result['time_ms'],
                                'Comparisons': result['comparisons'],
                                'Space (bytes)': result['space'],
                                'Matches': len(result['matches']),
                            })
                            current += 1
                            progress_bar.progress(current / total_tests)

        if not results:
            st.warning(
                "No valid test cases were generated (for example, all pattern lengths were larger than the text lengths). "
                "Please adjust your configuration and try again."
            )
        else:
            st.session_state['results'] = pd.DataFrame(results)
            st.success(f"✅ Benchmark completed! {len(results)} tests run.")

            # Show sample text/pattern
            if 'sample_text' in st.session_state:
                with st.expander("📝 View Sample Text/Pattern (First Test Case)"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Text Type:** {st.session_state['text_type']}")
                        st.markdown("**Sample Text (first 100 chars):**")
                        st.code(st.session_state['sample_text'], language='text')
                    with col2:
                        st.markdown("**Pattern:**")
                        st.code(st.session_state['sample_pattern'], language='text')

# Step-by-step visualization section
st.sidebar.markdown("---")
st.sidebar.header("🎬 Step-by-Step Visualization")
if st.sidebar.checkbox("Enable Visualization Mode"):
    viz_text = st.sidebar.text_input("Text (max 100 chars)", "abcabcabc")
    viz_pattern = st.sidebar.text_input("Pattern (max 20 chars)", "abc")
    viz_algo = st.sidebar.selectbox("Algorithm", ["Naive", "KMP", "Rabin-Karp", "Boyer-Moore-Horspool"])
    
    if len(viz_text) == 0 or len(viz_pattern) == 0:
        st.sidebar.warning("Provide both a **Text** and **Pattern** to start the visualization.")
    elif len(viz_text) > 100 or len(viz_pattern) > 20:
        st.sidebar.error("Text or pattern too long for visualization — keep text ≤ 100 and pattern ≤ 20 characters.")
    elif len(viz_pattern) > len(viz_text):
        st.sidebar.error("Pattern cannot be longer than the text for visualization.")
    else:
        if 'viz_steps' not in st.session_state:
            st.session_state['viz_steps'] = []
            st.session_state['viz_index'] = 0
        
        if st.sidebar.button("🔄 Reset Visualization"):
            algo_steps = {
                "Naive": naive_steps,
                "KMP": kmp_steps,
                "Rabin-Karp": rabin_karp_steps,
                "Boyer-Moore-Horspool": bmh_steps
            }
            st.session_state['viz_steps'] = list(algo_steps[viz_algo](viz_text, viz_pattern))
            st.session_state['viz_index'] = 0
            st.session_state['viz_text'] = viz_text
            st.session_state['viz_pattern'] = viz_pattern
            st.session_state['viz_algo'] = viz_algo

# Display results or visualization
if 'viz_steps' in st.session_state and len(st.session_state['viz_steps']) > 0 and 'results' not in st.session_state:
    # Show only visualization tab
    st.subheader(f"🎬 {st.session_state['viz_algo']} Algorithm - Step by Step")
    
    total_steps = len(st.session_state['viz_steps'])
    current_step = st.session_state['viz_index']
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
    
    with col1:
        if st.button("⏮️ First"):
            st.session_state['viz_index'] = 0
            st.rerun()
    
    with col2:
        if st.button("◀️ Previous"):
            if st.session_state['viz_index'] > 0:
                st.session_state['viz_index'] -= 1
                st.rerun()
    
    with col3:
        if st.button("▶️ Next"):
            if st.session_state['viz_index'] < total_steps - 1:
                st.session_state['viz_index'] += 1
                st.rerun()
    
    with col4:
        if st.button("⏭️ Last"):
            st.session_state['viz_index'] = total_steps - 1
            st.rerun()
    
    with col5:
        st.markdown(f"**Step {current_step + 1} / {total_steps}**")
    
    st.markdown("---")
    
    state = st.session_state['viz_steps'][current_step]
    
    # Render visualization
    viz_html = render_visualization(st.session_state['viz_text'], st.session_state['viz_pattern'], state)
    st.markdown(viz_html, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display message
    st.info(f"**{state['message']}**")
    
    # Display algorithm-specific info
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Text Position", state['text_pos'])
        st.metric("Pattern Position", state.get('pattern_pos', 'N/A'))
    
    with col_b:
        if 'lps' in state:
            st.write("**LPS Array:**", state['lps'])
        elif 'pattern_hash' in state:
            st.write(f"**Pattern Hash:** {state['pattern_hash']}")
            st.write(f"**Window Hash:** {state['text_hash']}")
        elif 'bad_char' in state and current_step == 0:
            st.write("**Bad Character Table:**")
            st.json(state['bad_char'])
    
    # Progress bar
    st.progress((current_step + 1) / total_steps)
    
    st.markdown("""
    ### Legend:
    - 🟡 **Yellow**: Currently comparing
    - 🔴 **Red**: Mismatch  
    - 🟢 **Green**: Match found
    - ⬜ **Gray**: Not yet processed
    """)

if 'results' in st.session_state:
    df = st.session_state['results']
    
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Results Overview",
            "📈 Performance Charts",
            "💾 Export & Complexity",
            "🎬 Step-by-Step (Interactive)",
        ]
    )
    
    with tab1:
        st.subheader("📊 Statistical Analysis")
        
        if len(df['Run'].unique()) > 1:
            stats = calculate_statistics(df)
            st.dataframe(stats.style.format({
                'Time Mean': '{:.3f}',
                'Time Median': '{:.3f}',
                'Time Std': '{:.3f}',
                'Time Min': '{:.3f}',
                'Time Max': '{:.3f}',
                'Comp Mean': '{:.0f}',
                'Comp Median': '{:.0f}',
                'Comp Std': '{:.0f}',
                'Comp Min': '{:.0f}',
                'Comp Max': '{:.0f}'
            }), use_container_width=True)
            
            st.markdown("**Key Insights:**")
            best_algo = stats.loc[stats['Time Mean'].idxmin(), 'Algorithm']
            best_time = stats['Time Mean'].min()
            st.success(f"✅ **Fastest Algorithm:** {best_algo} (Avg: {best_time:.3f} ms)")
            
            most_stable = stats.loc[stats['Time Std'].idxmin(), 'Algorithm']
            st.info(f"🎯 **Most Consistent:** {most_stable} (Lowest std deviation)")
            
            least_comp = stats.loc[stats['Comp Mean'].idxmin(), 'Algorithm']
            st.info(f"⚡ **Fewest Comparisons:** {least_comp}")
        else:
            st.info("Run multiple iterations to see statistical analysis (mean, median, std dev)")
        
        st.markdown("---")
        st.subheader("Aggregated Results")
        agg = df.groupby(['Algorithm', 'Text Type', 'n', 'm']).agg({
            'Time (ms)': 'mean',
            'Comparisons': 'mean',
            'Space (bytes)': 'first',
            'Matches': 'first'
        }).reset_index()
        st.dataframe(agg.style.format({
            'Time (ms)': '{:.3f}',
            'Comparisons': '{:.0f}',
            'Space (bytes)': '{:.0f}'
        }), use_container_width=True)
        
        st.markdown("---")
        st.subheader("Detailed Results")
        st.dataframe(df, use_container_width=True)
    
    with tab2:
        st.subheader("Performance Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Time vs Text Length**")
            fig, ax = plt.subplots(figsize=(8, 5))
            for algo in df['Algorithm'].unique():
                data = df[df['Algorithm'] == algo].groupby('n')['Time (ms)'].mean()
                ax.plot(data.index, data.values, marker='o', label=algo, linewidth=2)
            ax.set_xlabel('Text Length (n)', fontweight='bold')
            ax.set_ylabel('Time (ms)', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.markdown("**Comparisons vs Text Length**")
            fig, ax = plt.subplots(figsize=(8, 5))
            for algo in df['Algorithm'].unique():
                data = df[df['Algorithm'] == algo].groupby('n')['Comparisons'].mean()
                ax.plot(data.index, data.values, marker='o', label=algo, linewidth=2)
            ax.set_xlabel('Text Length (n)', fontweight='bold')
            ax.set_ylabel('Comparisons', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**Time vs Pattern Length**")
            fig, ax = plt.subplots(figsize=(8, 5))
            for algo in df['Algorithm'].unique():
                data = df[df['Algorithm'] == algo].groupby('m')['Time (ms)'].mean()
                ax.plot(data.index, data.values, marker='o', label=algo, linewidth=2)
            ax.set_xlabel('Pattern Length (m)', fontweight='bold')
            ax.set_ylabel('Time (ms)', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col4:
            st.markdown("**Space Usage**")
            fig, ax = plt.subplots(figsize=(8, 5))
            space_data = df.groupby('Algorithm')['Space (bytes)'].first()
            ax.bar(space_data.index, space_data.values, color=['#ef4444', '#3b82f6', '#10b981', '#f59e0b'])
            ax.set_ylabel('Space (bytes)', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
    
    with tab3:
        st.subheader("Export Data")
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="benchmark_results.csv",
            mime="text/csv"
        )
        
        st.subheader("Complexity Summary")
        st.markdown("""
        | Algorithm | Time (Preprocess) | Time (Search) | Space |
        |-----------|-------------------|---------------|-------|
        | Naive | O(1) | O(n×m) | O(1) |
        | KMP | O(m) | O(n+m) | O(m) |
        | Rabin-Karp | O(m) | O(n+m) avg | O(1) |
        | Boyer-Moore-Horspool | O(m+σ) | O(n/m) best | O(σ) |
        """)
    
    with tab4:
        if 'viz_steps' in st.session_state and len(st.session_state['viz_steps']) > 0:
            st.subheader(f"🎬 {st.session_state['viz_algo']} Algorithm - Step by Step")
            
            total_steps = len(st.session_state['viz_steps'])
            current_step = st.session_state['viz_index']
            
            col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
            
            with col1:
                if st.button("⏮️ First"):
                    st.session_state['viz_index'] = 0
                    st.rerun()
            
            with col2:
                if st.button("◀️ Previous"):
                    if st.session_state['viz_index'] > 0:
                        st.session_state['viz_index'] -= 1
                        st.rerun()
            
            with col3:
                if st.button("▶️ Next"):
                    if st.session_state['viz_index'] < total_steps - 1:
                        st.session_state['viz_index'] += 1
                        st.rerun()
            
            with col4:
                if st.button("⏭️ Last"):
                    st.session_state['viz_index'] = total_steps - 1
                    st.rerun()
            
            with col5:
                st.markdown(f"**Step {current_step + 1} / {total_steps}**")
            
            st.markdown("---")
            
            state = st.session_state['viz_steps'][current_step]
            
            # Render visualization
            viz_html = render_visualization(st.session_state['viz_text'], st.session_state['viz_pattern'], state)
            st.markdown(viz_html, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Display message
            st.info(f"**{state['message']}**")
            
            # Display algorithm-specific info
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Text Position", state['text_pos'])
                st.metric("Pattern Position", state.get('pattern_pos', 'N/A'))
            
            with col_b:
                if 'lps' in state:
                    st.write("**LPS Array:**", state['lps'])
                elif 'pattern_hash' in state:
                    st.write(f"**Pattern Hash:** {state['pattern_hash']}")
                    st.write(f"**Window Hash:** {state['text_hash']}")
                elif 'bad_char' in state and current_step == 0:
                    st.write("**Bad Character Table:**")
                    st.json(state['bad_char'])
            
            # Progress bar
            st.progress((current_step + 1) / total_steps)
            
            st.markdown("""
            ### Legend:
            - 🟡 **Yellow**: Currently comparing
            - 🔴 **Red**: Mismatch
            - 🟢 **Green**: Match found
            - ⬜ **Gray**: Not yet processed
            """)
        else:
            st.info("👈 Enable 'Step-by-Step Visualization' in the sidebar and click 'Reset Visualization' to start!")
            st.markdown("""
            ### How to use:
            1. Check **'Enable Visualization Mode'** in sidebar
            2. Enter text (max 100 chars) and pattern (max 20 chars)
            3. Select an algorithm
            4. Click **'Reset Visualization'**
            5. Use navigation buttons to step through the algorithm
            """)
else:
    if 'viz_steps' not in st.session_state or len(st.session_state.get('viz_steps', [])) == 0:
        st.info("👈 Configure settings in the sidebar and click 'Run Benchmark' or enable 'Step-by-Step Visualization' to get started!")
