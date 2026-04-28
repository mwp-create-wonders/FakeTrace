import uvicorn


if __name__ == "__main__":
    uvicorn.run("src.marc_app.web:app", host="127.0.0.1", port=7860, reload=False)
