FROM mambaorg/micromamba:1.4.5

COPY --chown=$MAMBA_USER:$MAMBA_USER . /tmp/.

RUN micromamba install -y -n base -f /tmp/cpu.yml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)

EXPOSE 80

HEALTHCHECK CMD curl --fail http://localhost:80/_stcore/health

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "streamlit", "run", "dashboard.py", "--server.port=80", "--server.address=0.0.0.0", "--global.developmentMode=False"]