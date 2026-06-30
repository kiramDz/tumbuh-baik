import os
import sys


def main():
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    from jobs.run_lstm import run_lstm_background_worker

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python jobs/lstm_worker_entrypoint.py <config_id> [db_name]")

    config_id = sys.argv[1]
    db_name = sys.argv[2] if len(sys.argv) > 2 else os.getenv("MONGODB_DB_NAME", "tugas_akhir")

    run_lstm_background_worker(
        config_id=config_id,
        mongo_uri=os.getenv("MONGODB_URI"),
        db_name=db_name,
    )


if __name__ == "__main__":
    main()
