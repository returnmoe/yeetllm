# syntax=docker/dockerfile:1.18@sha256:dabfc0969b935b2080555ace70ee69a5261af8a8f1b4df97b9e7fbcf6722eddf

# vLLM is consumed exactly as published by the vLLM project. The manifest-list
# digest makes the CUDA/PyTorch/Python runtime immutable while still selecting
# the correct architecture at build time.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.27.0@sha256:07ea4e292adf3a26b05ac97114b28849cf4551a26beb1fbe7decd3842d752ed7
ARG TOOLS_IMAGE=ubuntu:22.04@sha256:3b06811b2afd352be909dd088a004166d665dc76d38b13eada33522a9d915c6f

# Build the small operating-system overlay independently. The final stage has
# no RUN instruction, so BuildKit can reuse the official vLLM layers without
# unpacking the multi-gigabyte CUDA filesystem on a standard CI runner.
FROM ${TOOLS_IMAGE} AS runtime-tools

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --download-only --no-install-recommends \
        iproute2 \
        jq \
        less \
        openssh-client \
        openssh-server \
        pciutils \
        procps \
        tini \
        util-linux \
    && install -d -m 0755 /payload \
    && find /var/cache/apt/archives -maxdepth 1 -type f -name '*.deb' \
        -exec dpkg-deb --extract '{}' /payload ';' \
    && install -D -m 0755 /usr/bin/setpriv /payload/usr/bin/setpriv \
    && rm -f /payload/etc/ssh/ssh_host_* \
    && test -x /payload/usr/sbin/sshd \
    && test -x /payload/usr/bin/ssh-keygen \
    && test -x /payload/usr/bin/tini

FROM ${VLLM_IMAGE}

ARG YEETLLM_VERSION=0.1.0.dev0
ARG SOURCE_REVISION=development
ARG VLLM_VERSION=0.27.0
ARG VLLM_COMMIT=4bdc8a788d2e2ce9165d552b3d4d8b72604626bf

LABEL org.opencontainers.image.title="YeetLLM" \
      org.opencontainers.image.description="RunPod-ready multi-model vLLM appliance" \
      org.opencontainers.image.source="https://github.com/returnmoe/yeetllm" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.version="${YEETLLM_VERSION}" \
      ai.yeetllm.cuda.variant="cu130" \
      ai.yeetllm.cuda.expected="13.0" \
      ai.yeetllm.vllm.version="${VLLM_VERSION}" \
      ai.yeetllm.vllm.commit="${VLLM_COMMIT}" \
      ai.yeetllm.vllm.provenance="official-vllm-image"

USER root

# Linked copies keep YeetLLM's small additions independent of the inherited
# CUDA layers. This lets BuildKit rebase the overlay without unpacking the
# official vLLM root filesystem when the exporter supports lazy base reuse.
COPY --link --from=runtime-tools /payload/usr/ /usr/
# Ubuntu 22.04's package archives still contain legacy /bin, /sbin, and /lib
# paths, while the vLLM base exposes those as merged-/usr symlinks.
COPY --link --from=runtime-tools /payload/bin/ /usr/bin/
COPY --link --from=runtime-tools /payload/sbin/ /usr/sbin/
COPY --link --from=runtime-tools /payload/lib/ /usr/lib/
COPY --link --from=runtime-tools /payload/etc/ssh/ /etc/ssh/
COPY --link --from=runtime-tools /payload/etc/iproute2/ /etc/iproute2/
COPY --link --chmod=0644 docker/sshd-yeetllm.conf /etc/ssh/sshd_config
COPY --link --chmod=0755 docker/entrypoint.sh /usr/local/bin/yeetllm-entrypoint
COPY --link --chmod=0755 docker/healthcheck.sh /usr/local/bin/yeetllm-healthcheck
COPY --link --chmod=0755 docker/yeetllm /usr/local/bin/yeetllm
COPY --link --chown=2000:0 src/yeetllm /opt/yeetllm/yeetllm

ENV PYTHONPATH=/opt/yeetllm \
    YEETLLM_CONFIG=/workspace/yeetllm/config.yaml \
    YEETLLM_SSH_ENABLE=auto \
    YEETLLM_SSH_PORT=22 \
    HF_HOME=/workspace/yeetllm/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/yeetllm/cache/huggingface/hub \
    VLLM_CACHE_ROOT=/workspace/yeetllm/cache/vllm \
    XDG_CACHE_HOME=/workspace/yeetllm/cache \
    TRITON_CACHE_DIR=/workspace/yeetllm/cache/vllm/triton \
    TORCHINDUCTOR_CACHE_DIR=/workspace/yeetllm/cache/vllm/torchinductor \
    PYTHONUNBUFFERED=1 \
    VLLM_ALLOW_RUNTIME_LORA_UPDATING=0 \
    YEETLLM_IMAGE_VERSION=${YEETLLM_VERSION} \
    YEETLLM_CUDA_VARIANT=cu130 \
    YEETLLM_EXPECTED_CUDA=13.0 \
    YEETLLM_VLLM_VERSION=${VLLM_VERSION} \
    YEETLLM_VLLM_COMMIT=${VLLM_COMMIT}

EXPOSE 22
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=10s --start-period=60m --retries=3 \
    CMD ["/usr/local/bin/yeetllm-healthcheck"]
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/yeetllm-entrypoint"]
CMD ["yeetllm", "serve"]
