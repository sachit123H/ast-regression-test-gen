# Reproducible research environment for CS.91000 Graduation Research
# Ubuntu 22.04 + OpenJDK 8 (Defects4J) + OpenJDK 17 (GumTree 4.x) + Defects4J v2.0.1 + GumTree + Python 3.11
# Supports all 5 Defects4J projects: Lang, Math, Chart (Xvfb/AWT), Time (TZ/locales), Closure (Ant heap)
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV ANT_OPTS="-Xmx2048m -Dfile.encoding=UTF-8"
ENV JAVA_TOOL_OPTIONS="-Djava.awt.headless=true"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget unzip zip build-essential \
    openjdk-8-jdk openjdk-17-jdk ant maven \
    perl cpanminus libdbi-perl libdbd-csv-perl \
    python3 python3-venv python3-pip python3-dev \
    subversion patch diffutils \
    xvfb xauth x11-utils libxi6 libxrender1 libxtst6 \
    locales tzdata \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Set Java 8 as default for Defects4J (GumTree can use JAVA17_HOME)
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV JAVA17_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Install Defects4J v2.0.1
WORKDIR /opt
RUN git clone --depth 1 --branch v2.0.1 https://github.com/rjust/defects4j.git /opt/defects4j && \
    cd /opt/defects4j && \
    cpanm --installdeps . && \
    ./init.sh
ENV D4J_HOME=/opt/defects4j
ENV PATH="${D4J_HOME}/framework/bin:${PATH}"

# Install GumTree (v3.0.0 works on Java 11/17; v4.0.0-beta6 works on Java 17)
RUN mkdir -p /opt/gumtree && \
    wget -q https://github.com/GumTreeDiff/gumtree/releases/download/v3.0.0/gumtree-3.0.0.zip -O /tmp/gumtree.zip && \
    unzip -q /tmp/gumtree.zip -d /opt/gumtree && \
    rm /tmp/gumtree.zip && \
    printf '#!/bin/sh\nJAVA_HOME="${JAVA17_HOME:-/usr/lib/jvm/java-17-openjdk-amd64}" exec /opt/gumtree/gumtree-3.0.0/bin/gumtree "$@"\n' > /usr/local/bin/gumtree && \
    chmod +x /usr/local/bin/gumtree
ENV GUMTREE_BIN=/usr/local/bin/gumtree
ENV PATH="/usr/local/bin:${PATH}"

# Setup Python 3.11 workspace
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /workspace/requirements.txt

COPY . /workspace
ENV PYTHONPATH=/workspace

CMD ["bash"]
