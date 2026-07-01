import uvicorn


if __name__ == "__main__":
    uvicorn.run("src.faketrace_app.api.app:app", host="127.0.0.1", port=7860, reload=False)
