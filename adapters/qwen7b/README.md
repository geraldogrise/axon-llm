# adapters/

Onde ficam os adapters LoRA baixados do Drive, prontos pra registrar no Ollama.

Os arquivos `.gguf` **não são versionados** (~100 MB cada) — só este README e os
`Modelfile` entram no git.

## Como chegam aqui

O `finetune_expert_colab.ipynb` treina o adapter e a seção 11 converte pra GGUF, salvando
em `MyDrive/axon_gguf/<base>/<expert>-lora.gguf`. Baixe o arquivo do Drive e ponha nesta
pasta.

```
adapters/
├── go-lora.gguf
├── rust-lora.gguf
├── Modelfile.go
└── Modelfile.rust
```

## Registrar no Ollama

Uma vez só, o modelo base (o mesmo em que os adapters foram treinados):

```powershell
ollama pull qwen2.5-coder:7b-instruct
```

Um `Modelfile` por expert — só a linha do `ADAPTER` muda:

```
FROM qwen2.5-coder:7b-instruct
ADAPTER ./go-lora.gguf
```

```powershell
ollama create axon-go -f Modelfile.go
ollama list
```

O base fica guardado uma vez e é referenciado por todos, então os ~4 GB não se
multiplicam a cada expert.

## Usar com o pyaxon

O `pyaxon.generate` já fala com o Ollama em `localhost:11434` — é só passar o nome do
modelo:

```python
import pyaxon as ax

sistema = ax.system.AxonSystem.load(r"C:\caminho\para\axon_experts")
r = sistema.answer("como uso goroutines e canais?", model="axon-go")

print(r["mode"])     # "generated" -- se vier "extractive", o Ollama não respondeu
print(r["answer"])
```

`mode` em `generated` quer dizer que o ciclo fechou: o router escolheu o expert, o KB
recuperou a lição e o seu modelo escreveu a resposta em cima dela, tudo offline.

## Se a conversão do adapter falhar

O caminho alternativo é exportar o modelo já fundido, no próprio notebook:

```python
model.save_pretrained_gguf(destino, tokenizer, quantization_method="q4_k_m")
```

Dá ~4 GB por expert em vez de ~100 MB, porque cada arquivo carrega uma cópia do modelo
base — mas é um comando só e não depende do llama.cpp.
