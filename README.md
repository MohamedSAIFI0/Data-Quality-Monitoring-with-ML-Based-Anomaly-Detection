# ETL Data Quality Monitoring with ML-Based Anomaly Detection

Machine learning-powered anomaly detection system for monitoring ETL pipeline data quality using Isolation Forest algorithm.

---

## Quick Start

### Prerequisites

**Required Python packages:**

```bash
pandas
numpy
scikit-learn
matplotlib
```

### Installation

**Step 1: Clone the repository**

```bash
git clone https://github.com/MohamedSAIFI0/Data-Quality-Monitoring-with-ML-Based-Anomaly-Detection.git
cd Data-Quality-Monitoring-with-ML-Based-Anomaly-Detection
```

**Step 2: Install dependencies**

```bash
pip install pandas numpy scikit-learn matplotlib
```

### Running the Script

**Execute the anomaly detection:**

```bash
python Data-Quality-Monitoring-with-ML-Based-Anomaly-Detection.py
```

**Expected output:**
- Console report with detected anomalies
- Visualization saved as `anomaly_detection_results.png`
- Alert recommendations with severity levels

---

## Features

### Core Capabilities

**Monitors multiple data quality dimensions:**
- Transaction volume spikes and drops
- Pricing anomalies and outliers
- Data quality issues (null rates, missing values)
- ETL performance degradation
- Schema changes and unexpected patterns

### Alert System

**Three-tier alerting:**

**CRITICAL** - Blocks pipeline execution
- Null rate exceeds 10%
- Data corruption detected

**HIGH** - Requires immediate investigation
- Transaction volume spike (>3σ)
- Processing delays (>300 seconds)

**MEDIUM** - Monitor closely
- Unusual pricing patterns
- Minor data quality issues

---

## Configuration

### Adjusting Detection Sensitivity

**Modify contamination rate (expected anomaly frequency):**

```python
iso_forest = IsolationForest(
    contamination=0.05,  # 5% expected anomaly rate
    random_state=42,
    n_estimators=100
)
```

### Custom Alert Thresholds

**Set your own thresholds:**

```python
CRITICAL_NULL_THRESHOLD = 10.0
HIGH_PROCESSING_TIME = 300
VOLUME_SPIKE_SIGMA = 2
```

---

## Integration Examples

### With Apache Airflow

**Add to your DAG:**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def run_anomaly_detection(**context):
    import subprocess
    result = subprocess.run(['python', 'Data-Quality-Monitoring-with-ML-Based-Anomaly-Detection.py'], 
                          capture_output=True)
    if result.returncode != 0:
        raise AirflowException("Anomaly detected!")

with DAG('etl_monitoring', 
         start_date=datetime(2024, 1, 1),
         schedule_interval='@daily') as dag:
    
    quality_check = PythonOperator(
        task_id='anomaly_detection',
        python_callable=run_anomaly_detection
    )
```

### With Great Expectations

**Integrate ML checks:**

```python
import great_expectations as gx

context = gx.get_context()

def ml_anomaly_check(df):
    anomaly_score = iso_forest.score_samples(scaler.transform(df))
    if np.min(anomaly_score) < -0.5:
        raise ValueError("Anomaly detected!")
    return True
```

---

## Usage Examples

### Basic Execution

**Run with default settings:**

```bash
python Data-Quality-Monitoring-with-ML-Based-Anomaly-Detection.py
```

### Custom Data Source

**Modify the script to load your data:**

```python
# Replace the sample data generation with your data source
df = pd.read_csv('your_etl_metrics.csv')

# Ensure these columns exist:
required_columns = ['total_transactions', 'avg_order_value', 
                   'null_percentage', 'processing_time_sec', 
                   'unique_customers']
```

### Scheduled Monitoring

**Set up cron job for daily monitoring:**

```bash
# Open crontab
crontab -e

# Add daily execution at 2 AM
0 2 * * * cd /path/to/project && python Data-Quality-Monitoring-with-ML-Based-Anomaly-Detection.py >> logs/monitoring.log 2>&1
```

---

## Output Files

### Console Report

**Detailed anomaly analysis:**

```
ETL DATA QUALITY MONITORING WITH ISOLATION FOREST
============================================================
Dataset: 90 days of sales data

DETECTED ANOMALIES:
------------------------------------------------------------
Date: 2024-01-31 (Day 30)
   Anomaly Score: -0.234
   Transactions: 15,000
   HIGH: Transaction volume spike
   HIGH: ETL processing delay
```

### Visualization

**Generated chart: `anomaly_detection_results.png`**
- Time series plots for each metric
- Red markers highlighting detected anomalies
- Visual trend analysis

---

## Customization

### Adding New Metrics

**Extend monitoring to additional features:**

```python
# Add new columns to your dataframe
df['new_metric'] = your_data

# Update features list
features = ['total_transactions', 'avg_order_value', 
            'null_percentage', 'processing_time_sec', 
            'unique_customers', 'new_metric']
```

### Changing Algorithm

**Try alternative anomaly detection methods:**

**One-Class SVM:**
```python
from sklearn.svm import OneClassSVM
model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
predictions = model.fit_predict(X_scaled)
```

**Autoencoder:**
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

input_layer = Input(shape=(n_features,))
encoded = Dense(32, activation='relu')(input_layer)
decoded = Dense(n_features, activation='sigmoid')(encoded)
autoencoder = Model(input_layer, decoded)
```

---

## Use Cases

**Industry applications:**

- **E-commerce**: Daily sales pattern monitoring
- **Financial Services**: Fraud detection and data quality
- **Healthcare**: Patient data completeness tracking
- **IoT**: Sensor data quality monitoring
- **Logistics**: Shipment volume and timing anomalies

---

## Contributing

**Pull requests welcome!**

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## Resources

**Learn more:**

- [Isolation Forest Algorithm Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)

- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/outlier_detection.html)

##Created By:
Mohamed SAIFI: Cloud data Engineer
