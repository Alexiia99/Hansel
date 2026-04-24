# Dockerfile

FROM python:3.11-slim

# uv para instalar dependencias — más rápido que pip
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copiar solo los archivos de dependencias primero para aprovechar
# el cache de Docker — si el código cambia pero no las deps, esta
# capa no se reconstruye.
COPY pyproject.toml .
COPY README.md .
COPY src/ src/

# Instalar dependencias de producción (sin dev tools)
RUN uv pip install --system  .

# Directorio para CVs subidos en runtime
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "hansel.api.main:app", "--host", "0.0.0.0", "--port", "8000"]