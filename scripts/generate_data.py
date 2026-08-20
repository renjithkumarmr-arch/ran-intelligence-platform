from app.data import generate_telemetry, ROOT
if __name__ == "__main__":
    path=ROOT/"data"/"ran_telemetry.csv"
    generate_telemetry().to_csv(path,index=False)
    print(f"Generated {path}")
