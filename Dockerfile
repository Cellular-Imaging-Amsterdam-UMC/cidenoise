ARG RUNTIME_CACHE_IMAGE=w_cidenoise-runtime-cache:latest
FROM ${RUNTIME_CACHE_IMAGE}
COPY cidenoise/ /app/cidenoise/
COPY wrapper.py bilayers_cli.py config.yaml model_manifest.json version.txt /app/
COPY LICENSE THIRD_PARTY.md README.md /app/
COPY tools/cuda_smoke.py /app/tools/cuda_smoke.py
RUN python -m compileall -q /app
ENTRYPOINT ["python", "/app/wrapper.py"]
