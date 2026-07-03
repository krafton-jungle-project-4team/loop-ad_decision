import uvicorn


def main() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
