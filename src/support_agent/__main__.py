"""Run the API server: `uv run support-agent` or `python -m support_agent`."""

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "support_agent.app:create_app",
        factory=True,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
