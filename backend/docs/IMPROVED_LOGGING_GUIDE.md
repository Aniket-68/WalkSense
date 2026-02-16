# WalkSense - Improved Logging & Graph Generation System

## 📊 Overview

I've created a **comprehensive metrics logging and graph generation system** that captures real performance data and generates grounded evaluation graphs.

---

## 🆕 What's New

### 1. **Enhanced Metrics Logger** (`infrastructure/metrics_logger.py`)

A structured logging system that captures:

#### Component Performance
- ✅ YOLO detection latency (ms)
- ✅ VLM inference latency (ms)
- ✅ STT transcription latency (ms)
- ✅ LLM reasoning latency (ms)
- ✅ Frame processing latency (ms)

#### Detection Metrics
- ✅ Object labels with confidence scores
- ✅ Bounding boxes
- ✅ Spatial direction (left/center/right)
- ✅ Timestamps for temporal analysis

#### Alert Metrics
- ✅ Alert type (CRITICAL/WARNING/INFO)
- ✅ Severity level
- ✅ Suppression status (for redundancy filter effectiveness)
- ✅ Full message content

#### Query Metrics
- ✅ User query text
- ✅ End-to-end response time
- ✅ VLM usage flag (two-stage response tracking)

#### System Metrics
- ✅ FPS samples over time
- ✅ GPU memory usage (MB and %)
- ✅ Spatial prediction accuracy

### 2. **Real Data Graph Generator** (`scripts/generate_graphs_from_metrics.py`)

Generates **8 evaluation graphs** from actual logged data:

1. **Component Latency Distribution** - Box plots with mean/median
2. **YOLO Performance Timeline** - Time-series with moving average
3. **Alert Distribution** - Pie chart + suppression effectiveness
4. **Detection Frequency** - Top 15 objects with safety color coding
5. **Query Response Time** - Histogram split by VLM usage
6. **FPS Stability** - Real-time throughput over session
7. **Spatial Direction Accuracy** - Per-direction accuracy breakdown
8. **GPU Memory Usage** - Memory consumption timeline

---

## 🚀 How to Use

### Step 1: Integrate MetricsLogger into Your Code

```python
from infrastructure.metrics_logger import metrics_logger

# In your main loop:

# Start timer
metrics_logger.start_timer("yolo")

# ... do YOLO detection ...
detections = detector.detect(frame)

# Stop timer and log
metrics_logger.stop_timer("yolo")

# Log detections
metrics_logger.log_detection_batch(detections, frame_width=1280)

# Log alerts
metrics_logger.log_alert(
    alert_type="WARNING",
    severity="medium",
    message="Dog ahead. Proceed carefully.",
    suppressed=False
)

# Log queries
metrics_logger.start_timer("llm")
answer = llm.answer_query(query, context)
response_time = metrics_logger.stop_timer("llm")
metrics_logger.log_query(query, response_time, vlm_used=True)

# Log FPS
metrics_logger.log_fps(current_fps)

# Log GPU memory (if available)
try:
    import torch
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        metrics_logger.log_gpu_memory(used, total)
except:
    pass
```

### Step 2: Save Metrics at End of Session

```python
# At the end of your session (e.g., on KeyboardInterrupt):
metrics_logger.save_metrics()      # Saves raw data
metrics_logger.save_summary()      # Saves statistics
metrics_logger.print_summary()     # Prints to console
```

This creates:
- `logs/metrics/session_YYYYMMDD_HHMMSS.json` - Full metrics
- `logs/metrics/summary_YYYYMMDD_HHMMSS.json` - Statistics

### Step 3: Generate Graphs

```bash
python scripts/generate_graphs_from_metrics.py
```

Output: `plots/evaluation_real/` with 8 PNG graphs

---

## 📈 Example Integration

Here's how to add to `scripts/run_enhanced_camera.py`:

```python
# At the top
from infrastructure.metrics_logger import metrics_logger

# In main loop:
while True:
    # Capture frame
    frame = camera.read()
    
    # YOLO Detection
    metrics_logger.start_timer("yolo")
    detections = detector.detect(frame)
    metrics_logger.stop_timer("yolo")
    metrics_logger.log_detection_batch(detections, frame.shape[1])
    
    # Safety evaluation
    for det in detections:
        alert = safety_rules.evaluate(det)
        if alert:
            suppressed = redundancy_filter.should_suppress(
                alert.message, alert.type
            )
            metrics_logger.log_alert(
                alert_type=alert.type,
                severity="high" if alert.priority == 3 else "medium",
                message=alert.message,
                suppressed=suppressed
            )
    
    # VLM (if triggered)
    if should_run_vlm:
        metrics_logger.start_timer("vlm")
        description = qwen.describe_scene(frame, context)
        metrics_logger.stop_timer("vlm")
    
    # User query
    if user_query:
        query_start = time.time()
        
        # Stage 1: LLM
        metrics_logger.start_timer("llm")
        quick_answer = llm.answer_query(user_query, spatial_context)
        metrics_logger.stop_timer("llm")
        
        # Stage 2: VLM grounding (if available)
        vlm_used = description is not None
        
        response_time = (time.time() - query_start) * 1000
        metrics_logger.log_query(user_query, response_time, vlm_used)
    
    # FPS tracking
    fps = 1.0 / (time.time() - frame_start)
    metrics_logger.log_fps(fps)
    
    # GPU memory (every 10 frames)
    if frame_count % 10 == 0:
        try:
            import torch
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1024**2
                total = torch.cuda.get_device_properties(0).total_memory / 1024**2
                metrics_logger.log_gpu_memory(used, total)
        except:
            pass

# On exit
try:
    # ... main loop ...
except KeyboardInterrupt:
    print("\n\n🛑 Stopping WalkSense...")
    metrics_logger.save_metrics()
    metrics_logger.save_summary()
    metrics_logger.print_summary()
```

---

## 📊 Generated Graphs Preview

### 1. Component Latency Distribution
- Box plots showing min/max/median/mean for each component
- Sample counts displayed
- Mean markers for quick reference

### 2. YOLO Performance Timeline
- Raw latency data (light blue)
- 20-sample moving average (red)
- Mean line (green dashed)
- Shows performance stability over time

### 3. Alert Distribution
- **Left**: Pie chart of active alerts by type
- **Right**: Bar chart showing suppression effectiveness
- Displays reduction percentage

### 4. Detection Frequency
- Horizontal bar chart of top 15 objects
- Color-coded by safety level:
  - 🔴 Red: Critical (car, knife, gun)
  - 🟠 Orange: Warning (dog, pole, stairs)
  - 🔵 Blue: Info (person, chair, laptop)

### 5. Query Response Time
- Histogram split by VLM usage
- Shows two-stage response system effectiveness
- Mean response time line

### 6. FPS Stability
- Time-series of FPS over session
- Target FPS line (30)
- Mean FPS line
- Filled area for visual clarity

### 7. Spatial Direction Accuracy
- Bar chart for left/center/right accuracy
- Overall accuracy line
- Percentage labels on bars

### 8. GPU Memory Usage
- Time-series of VRAM usage
- Total memory line
- Mean usage with percentage
- Shows memory stability

---

## 🎯 Benefits

### For Your Report

1. **Grounded Evidence**: Every graph is based on actual logged data
2. **Reproducible**: Anyone can verify by running the system
3. **Comprehensive**: Covers all key performance aspects
4. **Professional**: Publication-quality visualizations

### For Development

1. **Performance Monitoring**: Track regressions
2. **Optimization Targets**: Identify bottlenecks
3. **A/B Testing**: Compare configurations
4. **Debugging**: Correlate issues with metrics

---

## 📝 Sample Output

```
======================================================================
  WalkSense Metrics Summary - Session 20260131_023045
======================================================================

⏱️  Duration: 127.3s

🔧 Component Latencies:
   YOLO    :  285.3ms avg (234-684ms, n=423)
   VLM     : 2341.7ms avg (1890-3120ms, n=25)
   STT     : 8234.2ms avg (6420-13680ms, n=7)
   LLM     : 1156.8ms avg (710-2340ms, n=12)

👁️  Detections: 1847 total, 23 unique objects

🚨 Alerts: 45 active, 1802 suppressed (97.6% reduction)

❓ Queries: 12 total, 5234ms avg response

📊 FPS: 28.7 avg (24.3-31.2)
   GPU: 62.3% avg, 5847MB peak

🎯 Spatial Accuracy: 94.2% (113/120)

======================================================================
```

---

## 🔄 Next Steps

1. **Run a test session** with the enhanced logging
2. **Generate graphs** from the metrics
3. **Include in report** with proper citations
4. **Compare** before/after optimization

---

**Created**: January 31, 2026  
**Version**: 2.0  
**Files**:
- `infrastructure/metrics_logger.py` (426 lines)
- `scripts/generate_graphs_from_metrics.py` (584 lines)
