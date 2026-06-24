import pandas as pd
from attend import mark_attendance, load_data

def run_simulation():
    print("--- SIMULATING ATTENDANCE MARKING ---")
    
    # Simulate scanning 3 different users
    users = ["STUDENT_001", "STUDENT_002", "STUDENT_003"]
    
    for user in users:
        success, msg = mark_attendance(user, "Python")
        print(f"Action: Scanned {user} -> Result: {success} | Message: {msg}")
        
    print("\n--- SIMULATING DUPLICATE SCAN ---")
    # Try to scan STUDENT_001 again in the same lab today
    success, msg = mark_attendance("STUDENT_001", "Python")
    print(f"Action: Re-scanned STUDENT_001 -> Result: {success} | Message: {msg}")

    print("\n--- READING STORED ATTENDANCE RECORDS (CSV) ---")
    df = load_data()
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_simulation()
