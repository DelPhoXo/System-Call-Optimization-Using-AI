import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import io
import base64

app = Flask(__name__)

# Load data
df = pd.read_csv("optimized_logs.csv")

# Convert 'time' to string (in case it's not)
df['time'] = df['time'].astype(str)

# Plot to Base64
def plot_to_base64(plot_func):
    img = io.BytesIO()
    plot_func()
    plt.savefig(img, format='png', bbox_inches='tight', facecolor='white')
    img.seek(0)
    plot_base64 = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_base64

# Plot 1: Latency Trend
def plot_latency_trend(filtered_df):
    plt.figure(figsize=(10, 4))
    plt.plot(filtered_df['time'], filtered_df['latency_before'], label="Before Optimization", color="red", marker='o')
    plt.plot(filtered_df['time'], filtered_df['latency_after'], label="After Optimization", color="green", marker='o')
    plt.xlabel("Time")
    plt.ylabel("Latency (s)")
    plt.title("Latency Trend Over Time", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True)

# Plot 2: Bar Comparison
def plot_comparison(filtered_df):
    plt.figure(figsize=(5, 4))
    plt.bar(["Before", "After"], 
            [filtered_df['latency_before'].mean(), filtered_df['latency_after'].mean()],
            color=["red", "green"])
    plt.ylabel("Average Latency (s)")
    plt.title("Latency Before vs After Optimization", fontsize=12)

@app.route('/', methods=['GET', 'POST'])
def dashboard():
    filtered_df = df.copy()
    
    if request.method == 'POST':
        pid = request.form.get('pid')
        t_start = request.form.get('time_start')
        t_end = request.form.get('time_end')

        if pid:
            filtered_df = filtered_df[filtered_df['pid'] == int(pid)]
        if t_start and t_end:
            filtered_df = filtered_df[(filtered_df['time'] >= t_start) & (filtered_df['time'] <= t_end)]

    latency_plot = plot_to_base64(lambda: plot_latency_trend(filtered_df))
    comparison_plot = plot_to_base64(lambda: plot_comparison(filtered_df))

    return render_template('dashboard.html',
                           latency_plot=latency_plot,
                           comparison_plot=comparison_plot,
                           data=filtered_df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
