import subprocess
import pandas as pd
import time
import os
from collections import defaultdict

# Function to run strace and capture system calls
def monitor_system_calls(pid=None, duration=5):
    if pid is None:
        cmd = ["strace", "-T", "-o", "strace_output.txt", "ls"]
    else:
        cmd = ["strace", "-T", "-o", "strace_output.txt", "-p", str(pid)]
    
    print("Monitoring system calls for {} seconds...".format(duration))
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(duration)
        process.terminate()
    except Exception as e:
        print("Error running strace:", e)
        return

# Function to parse strace output and extract system call data
def parse_strace_output():
    syscall_data = defaultdict(lambda: {"frequency": 0, "total_time": 0.0})
    
    try:
        with open("strace_output.txt", "r") as f:
            for line in f:
                if "<" in line and ">" in line:
                    syscall_name = line.split("(")[0].strip()
                    time_str = line.split("<")[1].split(">")[0]
                    try:
                        time_taken = float(time_str)
                    except ValueError:
                        continue
                    syscall_data[syscall_name]["frequency"] += 1
                    syscall_data[syscall_name]["total_time"] += time_taken
    except FileNotFoundError:
        print("strace_output.txt not found. Make sure strace ran successfully.")
        return None
    
    return syscall_data

# Function to generate structured data with timestamps
def generate_data(syscall_data, duration=5):
    data = {
        "time": [],
        "syscall": [],
        "frequency": [],
        "latency": [],
        "pid": []
    }
    
    for i in range(duration):
        timestamp = f"10:0{i}"
        for syscall, stats in syscall_data.items():
            data["time"].append(timestamp)
            data["syscall"].append(syscall)
            data["frequency"].append(stats["frequency"])
            latency = stats["total_time"] / stats["frequency"] if stats["frequency"] > 0 else 0
            data["latency"].append(latency)
            data["pid"].append(123)
    
    return data

# Function to save data to CSV
def save_to_csv(data):
    df = pd.DataFrame({
        "time": data["time"],
        "syscall": data["syscall"],
        "frequency": data["frequency"],
        "latency_before": data["latency"],
        "pid": data["pid"]
    })
    df.to_csv("syscall_logs.csv", index=False)
    print("Data saved to syscall_logs.csv")

# Main function
def main():
    monitor_system_calls(pid=None, duration=5)
    syscall_data = parse_strace_output()
    if not syscall_data:
        print("No system call data collected.")
        return
    data = generate_data(syscall_data, duration=5)
    save_to_csv(data)

if __name__ == "__main__":
    main()
