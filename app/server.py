import os

import uvicorn


def get_port() -> int:
    port = os.environ.get("PORT")
    if port is None:
        raise RuntimeError("PORT environment variable is required")

    try:
        return int(port)
    except ValueError as exc:
        raise RuntimeError("PORT environment variable must be an integer") from exc


def main() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=get_port())


if __name__ == "__main__":
    main()
