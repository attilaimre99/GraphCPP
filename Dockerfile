FROM mambaorg/micromamba:1.4.5

COPY --chown=$MAMBA_USER:$MAMBA_USER . /tmp/.

RUN micromamba install -y -n base -f /tmp/cpu.yml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]