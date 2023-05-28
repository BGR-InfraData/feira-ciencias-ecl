FROM python:3.10.10-slim
ENV PYTHONUNBUFFERED 1
WORKDIR /app
COPY . /app
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi
EXPOSE 5000
CMD streamlit run --server.port 5000 app.p
