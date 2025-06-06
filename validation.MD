# Tokenizer Validation Documentation

## Overview
This documentation describes the tokenizer validation process for European languages. The system consists of three main components:
1. A shell script for training the tokenizer (`run_european_tokenizer.sh`)
2. A configuration file (`config.yaml`)
3. A validation script (`validate_tokenizer.py`)

## Purpose
The validation process ensures that the trained tokenizer performs well across multiple European languages by:
- Testing tokenization quality on real-world text samples
- Measuring key metrics like token efficiency and character coverage
- Detecting anomalies in tokenization patterns
- Generating visualizations and reports for analysis

## Components

### 1. Training Script (`run_european_tokenizer.sh`)
This script handles the initial tokenizer training:

- **Environment Setup**:
  - Activates the conda environment "TerraNex"
  - Sets up necessary dependencies

- **Language Coverage**:
  - Includes 20 European languages (ISO 639-1 codes)
  - German (de), French (fr), Spanish (es), Italian (it), Dutch (nl)
  - Polish (pl), Portuguese (pt), Russian (ru), Swedish (sv), Danish (da)
  - Finnish (fi), Norwegian (no), Czech (cs), Hungarian (hu), Romanian (ro)
  - Bulgarian (bg), Greek (el), Croatian (hr), Slovak (sk), Slovenian (sl)

- **Data Sources**:
  - Uses two datasets for each language:
    - `oscar-corpus/mOSCAR`
    - `statmt/cc100`
  - Samples 5% from each dataset to create a balanced corpus

- **Training Parameters**:
  - Vocabulary size: 100,000 tokens
  - Embedding dimension: 1024
  - Number of workers: 8
  - Outputs:
    - Corpus file: `european_corpus.txt`
    - Tokenizer directory: `european_tokenizer`

### 2. Configuration File (`config.yaml`)
Controls the validation process parameters:

```yaml
# Paths
tokenizer_path: "european_tokenizer"
output_dir: "tokenizer_analysis"
log_dir: "logs"

# Processing settings
sample_size: 100
max_workers: 4
z_threshold: 2.0

# Visualization settings
dpi: 300
figure_size:
  boxplot: [15, 6]
  heatmap: [12, 8]
  scatter: [12, 8]

# Languages to validate
languages:
  - de  # German
  - fr  # French
  # ... (all 20 languages)

# Logging settings
logging:
  level: INFO
  format: "%(asctime)s - %(levelname)s - %(message)s"
  file_format: "tokenizer_validation_%Y%m%d_%H%M%S.log"
```

### 3. Validation Script (`validate_tokenizer.py`)
Performs the actual validation process:

- **Features**:
  - Parallel processing using ThreadPoolExecutor
  - Comprehensive error handling and logging
  - Detailed metrics calculation
  - Visualization generation
  - HTML report creation

- **Key Metrics**:
  - Average tokens per character
  - Standard deviation of tokenization
  - Token length distribution
  - Character coverage
  - Anomaly detection using Z-scores

## Workflow

1. **Training Phase**:
   ```bash
   ./run_european_tokenizer.sh
   ```
   - Creates the tokenizer and corpus
   - Outputs: `european_tokenizer/` and `european_corpus.txt`

2. **Validation Phase**:
   ```bash
   python validate_tokenizer.py -c config.yaml
   ```
   - Tests the tokenizer on each language
   - Generates analysis and visualizations
   - Creates HTML report

## Outputs and Analysis

### 1. JSON Results (`tokenizer_validation_results.json`)
Contains detailed metrics for each language:
- Number of samples processed
- Average tokens per character
- Standard deviation
- Anomaly counts
- Character coverage
- Sample analysis

### 2. Visualizations

#### Box Plot (`boxplot_by_language.png`)
- Shows tokenization ratio distribution
- Helps identify outliers and consistency
- X-axis: Languages
- Y-axis: Z-scores of tokens per character

#### Heatmap (`metrics_heatmap.png`)
- Normalized view of key metrics
- Metrics shown:
  - Average tokens per character
  - Standard deviation
  - Anomaly count
  - Character coverage
- Color intensity indicates relative performance

#### Scatter Plot (`consistency_scatter.png`)
- Shows relationship between efficiency and consistency
- X-axis: Average tokens per character
- Y-axis: Standard deviation
- Each point represents a language
- Helps identify optimal performance regions

### 3. HTML Report (`report.html`)
Comprehensive report including:
- Summary statistics table
- Status indicators (✅ for good, ⚠️ for anomalies)
- Embedded visualizations
- Detailed metrics for each language

### 4. Logs
- Located in `logs/` directory
- Timestamped log files
- Contains:
  - Processing progress
  - Error messages
  - Warning notifications
  - Success confirmations

## Usage Examples

1. **Basic Validation**:
   ```bash
   python validate_tokenizer.py
   ```

2. **Custom Sample Size**:
   ```bash
   python validate_tokenizer.py -s 200
   ```

3. **Specific Languages**:
   ```bash
   python validate_tokenizer.py -l "de fr es"
   ```

4. **Custom Configuration**:
   ```bash
   python validate_tokenizer.py -c custom_config.yaml
   ```

5. **No Browser Report**:
   ```bash
   python validate_tokenizer.py --no-browser
   ```

## Interpreting Results

1. **Good Performance Indicators**:
   - Low average tokens per character
   - Low standard deviation
   - High character coverage
   - No anomalies detected

2. **Warning Signs**:
   - High token-to-character ratios
   - Large standard deviations
   - Low character coverage
   - Multiple anomalies

3. **Action Items**:
   - Investigate languages with anomalies
   - Consider retraining for languages with poor coverage
   - Adjust tokenizer parameters if needed
   - Add more training data for problematic languages 