FROM python:3.12.12-slim-bookworm

# Only the execution API is used: no generation clients or dataset downloads.
# These versions match uv.lock. The runner checks Python/EvalPlus at startup.
RUN pip install --no-cache-dir --no-deps evalplus==0.3.1 numpy==2.5.3 psutil==7.2.2
COPY runner.py /opt/oracle/runner.py
ENV PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
USER 65534:65534
WORKDIR /tmp
ENTRYPOINT ["python", "-I", "/opt/oracle/runner.py"]
