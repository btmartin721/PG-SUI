# Base image with Conda
FROM continuumio/miniconda3

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONDA_ENV=pgsuienv

# Install system-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    curl \
    wget \
    libbz2-dev \
    liblzma-dev \
    libz-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libncurses-dev \
    libhdf5-dev \
    ca-certificates \
    unzip \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create a new Conda environment and install dependencies
RUN conda create -y -n $CONDA_ENV -c conda-forge -c btmartin721 \
    python=3.12 \
    numpy=2.2.6 \
    pandas=2.2.3 \
    pip && \
    conda clean -afy && \
    conda init bash && \
    echo "conda activate $CONDA_ENV" > ~/.bashrc

ENV PATH /opt/conda/envs/$CONDA_ENV/bin:$PATH

RUN conda run -n $CONDA_ENV pip install --no-cache-dir \
    pg-sui \
    pytest \
    jupyterlab && \
    conda clean -afy

# Create a non-root user and set home directory
RUN useradd -ms /bin/bash pgsuiuser && \
    mkdir -p /home/pgsuiuser/.config/matplotlib /app/results /app/docs /app/example_data && \
    chown -R pgsuiuser:pgsuiuser /app /home/pgsuiuser

# Set working directory
WORKDIR /app

# Copy application files with correct permissions
COPY --chown=pgsuiuser:pgsuiuser tests/ tests/
COPY --chown=pgsuiuser:pgsuiuser pgsui/example_data/ example_data/
COPY --chown=pgsuiuser:pgsuiuser README.md docs/README.md
COPY --chown=pgsuiuser:pgsuiuser configs/ configs/
COPY --chown=pgsuiuser:pgsuiuser workflow/ workflow/
COPY --chown=pgsuiuser:pgsuiuser scripts/ scripts/

# Switch to non-root user
USER pgsuiuser
ENV HOME=/home/pgsuiuser
ENV MPLCONFIGDIR=$HOME/.config/matplotlib
RUN chmod -R u+w $HOME/.config/matplotlib

# Run tests (non-blocking; allows image to build even if tests fail)
RUN conda run -n $CONDA_ENV pytest tests/ || echo "Tests failed during build; continuing..."

# Default container command
CMD ["bash"]
