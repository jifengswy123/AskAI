import uvicorn

HOST = "127.0.0.1"
PORT = 8000


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
