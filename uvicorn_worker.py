from uvicorn.workers import UvicornWorker

class AsyncioWorker(UvicornWorker):
    CONFIG_KWARGS = {"loop": "asyncio", "http": "h11"}
