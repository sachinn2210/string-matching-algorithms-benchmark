# 🔍 String Matching Algorithms Benchmark

A comprehensive benchmarking tool for comparing the performance of four classic string matching algorithms: **Naive**, **KMP (Knuth-Morris-Pratt)**, **Rabin-Karp**, and **Boyer-Moore-Horspool**.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration Parameters](#configuration-parameters)
- [Algorithms](#algorithms)
- [Output](#output)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- 🚀 **4 Algorithm Implementations**: Naive, KMP, Rabin-Karp, Boyer-Moore-Horspool
- 📊 **Real-time Benchmarking**: Instant performance comparison
- 📈 **Interactive Visualizations**: 4 different performance charts
- 🎯 **Dual Input Modes**: Manual text input or synthetic generation
- 📉 **Comprehensive Metrics**: Time, comparisons, space usage, match count
- 💾 **CSV Export**: Download results for further analysis
- 🔄 **Multiple Test Scenarios**: Random and repetitive text patterns
- 📱 **Responsive UI**: Clean, modern Streamlit interface

## 🎬 Demo

![App Screenshot](https://via.placeholder.com/800x400?text=String+Matching+Benchmark+Demo)

## 🔧 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/sachinn2210/string-matching-benchmark.git
cd string-matching-benchmark
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run streamlit_app.py
```

Or simply double-click `run_streamlit.bat` (Windows)

4. **Open your browser**
The app will automatically open at `http://localhost:8501`

## 📖 Usage

### Step 1: Choose Input Mode

**Synthetic Mode** (Recommended for benchmarking)
- Automatically generates test cases
- Configurable text and pattern sizes
- Multiple test scenarios

**Manual Mode**
- Enter your own text and pattern
- Test specific use cases
- Single test execution

### Step 2: Configure Parameters

See [Configuration Parameters](#configuration-parameters) section below.

### Step 3: Select Algorithms

Choose one or more algorithms to benchmark:
- ✅ Naive (Brute Force)
- ✅ KMP (Knuth-Morris-Pratt)
- ✅ Rabin-Karp
- ✅ Boyer-Moore-Horspool

### Step 4: Run Benchmark

Click **"🚀 Run Benchmark"** and wait for completion.

### Step 5: Analyze Results

View results in three tabs:
- **📊 Results**: Aggregated and detailed performance data
- **📈 Visualizations**: Interactive performance charts
- **💾 Export**: Download CSV and view complexity summary

## ⚙️ Configuration Parameters

### 1. **Input Mode**
- **Synthetic**: Auto-generates test cases for comprehensive benchmarking
- **Manual**: Use your own text and pattern

### 2. **Text Sizes (n)** _(Synthetic mode)_
- **What**: Length of text strings to test
- **Format**: Comma-separated values (e.g., `1000,5000,10000`)
- **Purpose**: Analyze how algorithms scale with input size
- **Example**: `1000,5000,10000` tests with 1K, 5K, and 10K character texts

### 3. **Pattern Sizes (m)** _(Synthetic mode)_
- **What**: Length of patterns to search for
- **Format**: Comma-separated values (e.g., `3,10,50`)
- **Purpose**: Understand pattern length impact on performance
- **Example**: `3,10,50` tests with 3, 10, and 50 character patterns

### 4. **Text Types** _(Synthetic mode)_
- **Random**: Randomly generated text (a-z) - realistic scenario
- **Repetitive**: Repeated characters (e.g., "aaaa...b") - worst-case scenario
- **Purpose**: Expose different algorithm behaviors

### 5. **Number of Runs**
- **What**: Repetitions per test case
- **Range**: 1-10
- **Default**: 3
- **Purpose**: Statistical reliability through averaging

### 6. **Select Algorithms**
- **What**: Which algorithms to benchmark
- **Options**: Naive, KMP, Rabin-Karp, Boyer-Moore-Horspool
- **Purpose**: Compare specific algorithms or test all

### Example Configuration

```
Text Sizes: 1000, 5000, 10000
Pattern Sizes: 10, 50
Text Types: random, repetitive
Algorithms: All 4
Runs: 3

Total Tests = 3 × 2 × 2 × 4 × 3 = 144 tests
```

## 🧮 Algorithms

### 1. Naive (Brute Force)
- **Time Complexity**: O(n×m)
- **Space Complexity**: O(1)
- **Best For**: Small texts, educational purposes

### 2. KMP (Knuth-Morris-Pratt)
- **Time Complexity**: O(n+m)
- **Space Complexity**: O(m)
- **Best For**: Guaranteed linear time, repetitive patterns

### 3. Rabin-Karp
- **Time Complexity**: O(n+m) average, O(n×m) worst
- **Space Complexity**: O(1)
- **Best For**: Multiple pattern matching, plagiarism detection

### 4. Boyer-Moore-Horspool
- **Time Complexity**: O(n/m) best, O(n×m) worst
- **Space Complexity**: O(σ) where σ = alphabet size
- **Best For**: Large texts, random patterns, practical applications

## 📊 Output

### Metrics Collected
- ⏱️ **Execution Time**: Preprocessing + matching time (milliseconds)
- 🔢 **Comparisons**: Number of character comparisons
- 💾 **Space Usage**: Memory consumption (bytes)
- 🎯 **Matches**: Number and positions of pattern occurrences

### Visualizations
1. **Time vs Text Length**: How execution time scales with input size
2. **Comparisons vs Text Length**: Character comparison analysis
3. **Time vs Pattern Length**: Pattern size impact
4. **Space Usage**: Memory consumption comparison

### Export Options
- 📥 **CSV Download**: Full results for Excel/Python analysis
- 📋 **Complexity Table**: Theoretical time/space complexity reference

## 📁 Project Structure

```
streamlit_version/
├── streamlit_app.py      # Main application
├── requirements.txt      # Python dependencies
├── run_streamlit.bat     # Quick start script (Windows)
└── README.md            # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

Created as part of Design and Analysis of Algorithms coursework.

## 🙏 Acknowledgments

- Algorithms based on classic computer science literature
- Built with [Streamlit](https://streamlit.io/)
- Visualizations powered by [Matplotlib](https://matplotlib.org/)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**⭐ Star this repository if you found it helpful!**
