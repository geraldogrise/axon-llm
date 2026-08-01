"""axon-desk: o backend do aplicativo de mesa do axon-lang.

Fica ao lado do `pyaxon`, não dentro: o `pyaxon` é uma biblioteca com postura de zero
dependências e ciclo de vida próprio; o app depende dele numa direção só.

Sobe um servidor HTTP em 127.0.0.1 numa porta escolhida pelo sistema, anuncia porta e
token pelo stdout, e serve o Electron. Tudo que não é protocolo vai para o stderr.
"""

__version__ = "0.1.0"

# A única linha que o Electron lê do stdout. Qualquer print solto no meio disso quebra
# o aperto de mão -- por isso o resto do backend escreve em stderr.
HANDSHAKE = "AXON_DESK_READY"
