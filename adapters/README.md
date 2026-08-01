# adapters/

Os adapters LoRA baixados do Drive, prontos para registrar no Ollama.

**Uma pasta por modelo base.** Não é organização estética: um adapter só encaixa na
arquitetura e no tamanho em que foi treinado. Um adapter do 7B não carrega no 1.5B — as
matrizes têm dimensões diferentes — e misturá-los numa pasta só é convite a registrar o
`FROM` errado e ficar caçando por que a resposta saiu estranha.

```
adapters/
├── qwen7b/          treinados em Qwen2.5-Coder-7B-Instruct
│   ├── go-lora.gguf   (18 arquivos, ~80 MB cada)
│   └── ...
└── qwen15b/         treinados em Qwen2.5-Coder-1.5B-Instruct
```

Os `.gguf` **não são versionados** — 1,4 GB por conjunto. Só este README e os
`Modelfile` entram no git.

## Registrar no Ollama

O base, uma vez só — e tem que ser o mesmo em que o adapter foi treinado:

```powershell
ollama pull qwen2.5-coder:7b-instruct      # para os de qwen7b/
ollama pull qwen2.5-coder:1.5b             # para os de qwen15b/
```

Um `Modelfile` por expert:

```
FROM qwen2.5-coder:7b-instruct
ADAPTER ./go-lora.gguf
PARAMETER temperature 0.3
```

```powershell
ollama create axon-go -f Modelfile.go
ollama list
```

O Ollama guarda o base uma vez e o referencia, então os 4,7 GB não se multiplicam a cada
expert — cada um custa só os ~80 MB do adapter.

## Usar

O `axon-chat` e o `axon-code` leem a lista do Ollama e mostram num seletor no cabeçalho;
não precisa configurar nada. Pelo pyaxon direto:

```python
import pyaxon as ax

sistema = ax.system.AxonSystem.load(r"C:\caminho\para\axon_experts")
r = sistema.answer("como uso goroutines?", model="axon-go")
print(r["mode"])     # "generated" -- se vier "extractive", o Ollama não respondeu
```

## Qual base escolher

Medido na máquina de desenvolvimento (CPU, sem GPU utilizável), mesma pergunta e mesmo
material recuperado:

| | entrada | prefill | geração | total |
|---|---:|---:|---:|---:|
| 7B com adapter | 283 tok | 55,4 s | 66 tok em 46 s | **105 s** |
| 1.5B puro | 283 tok | 10,8 s | 120 tok em 21 s | **48 s** |

O 7B responde melhor — foi direto ao ponto onde o 1.5B fez uma lista genérica. O 1.5B
responde em menos da metade do tempo. Com uma NVIDIA de 8 GB o 7B passaria a responder em
poucos segundos e a escolha deixaria de existir.

Os adapters do 1.5B saem do `notebooks/finetune_1.5b_colab.ipynb`.
