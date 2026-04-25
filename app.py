"""
Hugging Face Space entrypoint wrapper.

Keeps Dockerfile compatibility by importing the FastAPI app from server.py.
"""

from server import app