# Docker — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Docker/containers.
**Expert sugerido**: família `docker` dentro de `devops_experts` (fase 11). **Total est.**: ~65 lições.
**Convenção**: `treinamento_devops/docker/<subsetor>/*.md` → path = [docker, subsetor].

## fundamentos/ — ~12
o que são containers vs VMs; o que é o Docker (arquitetura: engine/daemon/client); instalação; imagens vs containers; o Docker Hub e registries; ciclo de vida de um container; `docker run`; `docker ps`/`stop`/`rm`; `docker exec`; logs (`docker logs`); inspecionar (`docker inspect`); estatísticas.

## imagens/ — ~14
o Dockerfile; instruções (`FROM`/`RUN`/`COPY`/`ADD`); `CMD` vs `ENTRYPOINT`; `WORKDIR`/`ENV`/`EXPOSE`; camadas (layers) e cache; `docker build` e tags; `.dockerignore`; multi-stage builds; imagens base (alpine/slim/distroless); reduzir tamanho de imagem; `ARG` e build args; push/pull em registry; healthcheck; labels.

## runtime-dados/ — ~12
volumes (named e anonymous); bind mounts; `tmpfs`; persistência de dados; portas e `-p`; variáveis de ambiente; `--env-file`; restart policies; limites de recursos (CPU/memória); usuário não-root; `docker cp`; working with stdin/tty.

## redes/ — ~10
redes no Docker; bridge network; host network; none; redes customizadas; comunicação entre containers; DNS interno; `docker network`; expor vs publicar portas; overlay (visão geral).

## compose/ — ~12
o que é o Docker Compose; `docker-compose.yml`; serviços; `depends_on`; volumes no compose; redes no compose; variáveis de ambiente; `up`/`down`/`logs`; escalar serviços; override files; profiles; multi-container (app + banco + cache).

## producao-seguranca/ — ~5
boas práticas de imagem; scan de vulnerabilidades; secrets no Docker; BuildKit; registries privados.
