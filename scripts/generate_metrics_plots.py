import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import numpy as np
import traceback

# Log file path
LOG_FILE = r'd:\Github\WalkSense\logs\performance.log'
OUTPUT_DIR = r'd:\Github\WalkSense\docs\plots'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_logs(file_path):
    perf_data = []
    alert_data = []
    
    perf_pattern = re.compile(r"Performance Stats: ({.*})")
    alert_pattern = re.compile(r"Safety Alert: (Danger|Warning|Info)!")
    time_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            time_match = time_pattern.search(line)
            if not time_match:
                continue
            
            timestamp_str = time_match.group(1)
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
            except:
                continue
            
            # Parse Performance
            perf_match = perf_pattern.search(line)
            if perf_match:
                try:
                    stats_json = perf_match.group(1).replace("'", '"')
                    stats = json.loads(stats_json)
                    
                    entry = {
                        'timestamp': timestamp,
                        'yolo_avg': float(stats.get('yolo', {}).get('avg_ms', 0)),
                        'stt_avg': float(stats.get('stt', {}).get('avg_ms', 0)),
                        'llm_avg': float(stats.get('llm_reasoning', {}).get('avg_ms', 0)),
                        'frame_total_avg': float(stats.get('frame_total', {}).get('avg_ms', 0))
                    }
                    perf_data.append(entry)
                except Exception:
                    pass
            
            # Parse Alerts
            alert_match = alert_pattern.search(line)
            if alert_match:
                alert_type = alert_match.group(1)
                alert_data.append({'timestamp': timestamp, 'type': alert_type})

    return pd.DataFrame(perf_data), pd.DataFrame(alert_data)

def generate_plots(df_perf, df_alerts):
    try:
        if df_perf.empty:
            print("No performance data found in logs.")
            return

        # Ensure numeric
        cols = ['yolo_avg', 'frame_total_avg', 'stt_avg', 'llm_avg']
        for col in cols:
            df_perf[col] = pd.to_numeric(df_perf[col], errors='coerce').fillna(0)

        # Calculate FPS (1000ms / frame_time)
        df_perf['fps'] = 1000 / df_perf['frame_total_avg'].replace(0, np.nan)
        
        # Set style
        sns.set_theme(style="darkgrid")
        df_perf = df_perf.set_index('timestamp')
        
        # --- ORIGINAL PLOTS ---
        
        # 1. Latency Evolution
        plt.figure(figsize=(12, 6))
        df_resampled = df_perf[cols].rolling('10s').mean()
        sns.lineplot(data=df_resampled, x=df_resampled.index, y='yolo_avg', label='YOLO Detection')
        sns.lineplot(data=df_resampled, x=df_resampled.index, y='frame_total_avg', label='Total Frame Pipeline')
        plt.title('System Latency Evolution (10s Moving Average)')
        plt.xlabel('Time')
        plt.ylabel('Latency (ms)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '01_latency_evolution.png'))
        plt.close()
        
        # 2. Interaction Latency
        plt.figure(figsize=(10, 6))
        interaction_df = df_perf[df_perf['stt_avg'] > 0]
        if not interaction_df.empty:
            mean_stt = interaction_df['stt_avg'].mean()
            mean_llm = interaction_df['llm_avg'].mean()
            plt.bar(['STT', 'LLM'], [mean_stt, mean_llm], color=['#3498db', '#e74c3c'])
            plt.title('Average Interaction Latency')
            plt.ylabel('Time (ms)')
            plt.text(0, mean_stt, f"{mean_stt:.0f} ms", ha='center', va='bottom')
            plt.text(1, mean_llm, f"{mean_llm:.0f} ms", ha='center', va='bottom')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, '02_interaction_latency.png'))
        plt.close()
        
        # 3. Pipeline Responsibility
        yolo_val = df_perf['yolo_avg'].mean()
        frame_val = df_perf['frame_total_avg'].mean()
        overhead = max(0, frame_val - yolo_val)
        labels = ['YOLO Model', 'Processing Overhead']
        sizes = [yolo_val, overhead]
        colors = ['#2ecc71', '#95a5a6']
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.title('Frame Processing Breakdown')
        plt.savefig(os.path.join(OUTPUT_DIR, '03_pipeline_responsibility.png'))
        plt.close()

        # --- NEW PLOTS ---

        # 4. FPS Stability
        plt.figure(figsize=(12, 6))
        # Remove massive outliers for cleaner plot
        df_fps = df_perf['fps'][df_perf['fps'] < 60] 
        df_fps_resampled = df_fps.rolling('10s').mean()
        
        sns.lineplot(data=df_fps_resampled, color='#8e44ad', linewidth=2)
        plt.axhline(y=20, color='r', linestyle='--', label='Target (20 FPS)')
        plt.axhline(y=30, color='g', linestyle='--', label='Ideal (30 FPS)')
        plt.title('System Real-Time Stability (FPS)')
        plt.xlabel('Time')
        plt.ylabel('Frames Per Second')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '06_fps_stability.png'))
        plt.close()

        # 5. Alert Distribution
        if not df_alerts.empty:
            plt.figure(figsize=(8, 6))
            sns.countplot(data=df_alerts, x='type', palette='viridis', order=['Info', 'Warning', 'Danger'])
            plt.title('Safety Alert Distribution')
            plt.xlabel('Alert Severity')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, '07_alert_distribution.png'))
            plt.close()
            
        # 6. Component Latency Boxplot (Variance Analysis)
        plt.figure(figsize=(10, 6))
        # Melt data for boxplot
        stats_box = df_perf[['yolo_avg', 'frame_total_avg']].copy()
        stats_box.columns = ['YOLO', 'Total Frame']
        # Add LLM if available
        if df_perf['llm_avg'].sum() > 0:
            stats_box['LLM'] = df_perf['llm_avg'].replace(0, np.nan)
        
        df_melted = stats_box.melt(var_name='Component', value_name='Latency (ms)')
        df_melted = df_melted.dropna()
        
        sns.boxplot(x='Component', y='Latency (ms)', data=df_melted, palette="Set2")
        plt.yscale('log') # Log scale because LLM is much slower than YOLO
        plt.title('Component Latency Distribution (Log Scale)')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '08_latency_boxplot.png'))
        plt.close()

        print(f"Plots saved to {OUTPUT_DIR}")
        
    except Exception as e:
        print("Error generating plots:")
        traceback.print_exc()

if __name__ == "__main__":
    if not os.path.exists(LOG_FILE):
        print(f"Log file not found at {LOG_FILE}")
    else:
        print(f"Parsing {LOG_FILE}...")
        df_perf, df_alerts = parse_logs(LOG_FILE)
        print(f"Found {len(df_perf)} performance points and {len(df_alerts)} alerts.")
        generate_plots(df_perf, df_alerts)
