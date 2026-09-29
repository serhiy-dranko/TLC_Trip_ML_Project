import argparse

def main():
    parser = argparse.ArgumentParser(description="NYC Taxi data pipeline")
    parser.add_argument("--stage", type=str, required=True, choices=["ingest"],
                        help="Which pipeline stage to run")
    args = parser.parse_args()
    print(f"Stage requested: {args.stage}")

if __name__ == "__main__":
    main()