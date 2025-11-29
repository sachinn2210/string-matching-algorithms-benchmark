import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

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

# Streamlit App
st.set_page_config(page_title="String Matching Algorithms Benchmark", page_icon="🔍", layout="wide")

st.title("🔍 String Matching Algorithms Benchmark")
st.markdown("Compare performance of **Naive**, **KMP**, **Rabin-Karp**, and **Boyer-Moore-Horspool** algorithms")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

input_mode = st.sidebar.radio("Input Mode", ["Synthetic", "Manual"])

if input_mode == "Manual":
    text = st.sidebar.text_area("Text", "abcabcabcabc", height=100)
    pattern = st.sidebar.text_input("Pattern", "abc")
    text_sizes = [len(text)]
    pattern_sizes = [len(pattern)]
    text_types = ["manual"]
    runs = 1
else:
    text_sizes = st.sidebar.text_input("Text Sizes (comma-separated)", "1000,5000,10000")
    text_sizes = [int(x.strip()) for x in text_sizes.split(',')]
    
    pattern_sizes = st.sidebar.text_input("Pattern Sizes (comma-separated)", "3,10,50")
    pattern_sizes = [int(x.strip()) for x in pattern_sizes.split(',')]
    
    text_types = st.sidebar.multiselect("Text Types", ["random", "repetitive"], default=["random", "repetitive"])
    runs = st.sidebar.slider("Number of Runs", 1, 10, 3)

algorithms = st.sidebar.multiselect(
    "Select Algorithms",
    ["Naive", "KMP", "Rabin-Karp", "Boyer-Moore-Horspool"],
    default=["Naive", "KMP", "Rabin-Karp", "Boyer-Moore-Horspool"]
)

if st.sidebar.button("🚀 Run Benchmark", type="primary"):
    algo_map = {
        "Naive": naive_search,
        "KMP": kmp_search,
        "Rabin-Karp": rabin_karp_search,
        "Boyer-Moore-Horspool": boyer_moore_horspool_search
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
                if input_mode == "Manual":
                    test_text, test_pattern = text, pattern
                elif text_type == "random":
                    import random
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
                            'Matches': len(result['matches'])
                        })
                        current += 1
                        progress_bar.progress(current / total_tests)
    
    st.session_state['results'] = pd.DataFrame(results)
    st.success(f"✅ Benchmark completed! {len(results)} tests run.")
    
    # Show sample text/pattern
    if 'sample_text' in st.session_state:
        with st.expander("📝 View Sample Text/Pattern (First Test Case)"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Text Type:** {st.session_state['text_type']}")
                st.markdown(f"**Sample Text (first 100 chars):**")
                st.code(st.session_state['sample_text'], language='text')
            with col2:
                st.markdown(f"**Pattern:**")
                st.code(st.session_state['sample_pattern'], language='text')

# Display results
if 'results' in st.session_state:
    df = st.session_state['results']
    
    tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Visualizations", "💾 Export"])
    
    with tab1:
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
