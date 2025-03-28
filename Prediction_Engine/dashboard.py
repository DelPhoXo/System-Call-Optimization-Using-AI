import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import io
import base64

app = Flask(__name__)

# Load optimized data from Module 2
df = pd.read_csv("optimized_logs.csv")

# Function to generate plot and convert to base64
def plot_to_base64(plot_func):
    img = io.BytesIO()
    plot_func()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_base64 = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_base64

# Plot 1: Latency Trend (Line Chart)
def plot_latency_trend(filtered_df):
    plt.figure(figsize=(8, 4))
    plt.plot(filtered_df['time'], filtered_df['latency_before'], label="Before Optimization", color="red", marker='o')
    plt.plot(filtered_df['time'], filtered_df['latency_after'], label="After Optimization", color="green", marker='o')
    plt.xlabel("Time")
    plt.ylabel("Latency (seconds)")
    plt.title("Latency Trend Over Time")
    plt.legend()
    plt.grid(True)

# Plot 2: Before vs After Comparison (Bar Chart)
def plot_comparison(filtered_df):
    plt.figure(figsize=(6, 4))
    plt.bar(["Before", "After"], 
            [filtered_df['latency_before'].mean(), filtered_df['latency_after'].mean()],
            color=["red", "green"])
    plt.ylabel("Average Latency (seconds)")
    plt.title("Latency Before vs After Optimization")

# Flask Route
@app.route('/', methods=['GET', 'POST'])
def dashboard():
    filtered_df = df
    
    if request.method == 'POST':
        pid_filter = request.form.get('pid')
        time_start = request.form.get('time_start')
        time_end = request.form.get('time_end')
        
        if pid_filter:
            filtered_df = filtered_df[filtered_df['pid'] == int(pid_filter)]
        if time_start and time_end:
            filtered_df = filtered_df[(filtered_df['time'] >= time_start) & 
                                    (filtered_df['time'] <= time_end)]
    
    latency_plot = plot_to_base64(lambda: plot_latency_trend(filtered_df))
    comparison_plot = plot_to_base64(lambda: plot_comparison(filtered_df))
    
    return render_template('dashboard.html', 
                         latency_plot=latency_plot, 
                         comparison_plot=comparison_plot, 
                         data=filtered_df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
