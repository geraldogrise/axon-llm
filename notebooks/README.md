# Notebooks — rodar o axon-lang no Google Colab

Dois repositórios, um de cada lado:

| Repo | O que tem |
|---|---|
| [`axon-llm`](https://github.com/geraldogrise/axon-llm) (este) | código: C++ core, pyaxon, `examples/build_*_experts.py` |
| [`treinamento`](https://github.com/geraldogrise/treinamento) | dados: as lições, **uma branch por fase** (`fase-1` … `fase-12`); a `main` é vazia de propósito |

O mapa "expert → branch → script" vive em [`axon_colab.py`](./axon_colab.py) — é a única
fonte da tabela, os notebooks importam de lá. `ac.tabela()` imprime tudo.

Os dois repos são **públicos**, então o Colab clona sem credencial nenhuma — é só abrir
o notebook e rodar.

> Se um dia você fechar os repos, defina um secret `GH_TOKEN` nos Secrets do Colab (🔑):
> o `axon_colab.py` usa se existir e ignora se não existir.

## Ordem

### 1. `train_expert_colab.ipynb` — o expert (retrieval)
Compila o `_axon.so`, clona **só** a branch daquele expert, roda o `build_<x>_experts.py`
e salva `router.*.json` + `kb.sparse.json.gz` em `MyDrive/axon_experts/<expert>_experts/`.

Um expert por vez: troque `EXPERT` na célula 4 e rode 4 → 5 → 6 de novo. Treinar `go`
não mexe em `rust`.

### 2. `finetune_expert_colab.ipynb` — o gerador (QLoRA)
Um adapter LoRA por expert **e** por modelo-base, a partir das mesmas lições.

Rode inteiro com `BASE = "deepseek"`; depois reinicie a sessão, ponha `BASE = "qwen"`
e rode de novo. Os dois convivem e a célula 9 compara o `eval_loss` lado a lado.

```
MyDrive/axon_lora/
  deepseek/go/    adapter + metrica.json
  qwen/go/        adapter + metrica.json
MyDrive/axon_ckpt/
  deepseek/go/    checkpoint-50, checkpoint-100 ...   <- retomável
```

**Checkpoints:** o treino grava no Drive a cada 50 passos. Se a sessão do Colab cair,
rode a mesma célula outra vez — ela acha o último `checkpoint-<n>` e retoma dali.

A célula 10 junta as duas metades: o router escolhe a família, o KB recupera a lição,
o LoRA responde com esse material no prompt.

### 3. `cuda_colab.ipynb` — opcional
Compila o pyaxon com CUDA/cuBLAS e mede o speedup do `matmul` (e do treino) na GPU.
Não é necessário pros dois de cima; é a demonstração de que o backend de GPU funciona.

## Modelos-base

| Chave | Modelo | Nota |
|---|---|---|
| `deepseek` | `deepseek-ai/deepseek-coder-6.7b-instruct` | fp16; o `load_in_4bit` quantiza na hora (baixa ~13 GB) |
| `deepseek-1b` | `deepseek-ai/deepseek-coder-1.3b-instruct` | idem, bem menor |
| `qwen` | `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit` | já vem em 4-bit (~4 GB) |
| `qwen-14b` | `unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit` | apertado na T4 |
| `qwen-3b` | `unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit` | se a T4 estourar memória |
| `qwen-1.5b` | `unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit` | o mais folgado |

O Unsloth não publica um 4-bit do `deepseek-coder` — os DeepSeek dele são R1/Prover. Por
isso essa entrada aponta pro repositório oficial em fp16.

Na T4, 7B em 4-bit com `max_seq_length=2048` roda com `batch_size=1` +
`gradient_accumulation=8`. Se der OOM, caia pro `qwen-3b` ou baixe o `MAX_LEN` pra 1024.
