# Notebooks — rodar o axon-lang no Google Colab

Dois repositórios, um de cada lado:

| Repo | O que tem |
|---|---|
| [`axon-llm`](https://github.com/geraldogrise/axon-llm) (este) | código: C++ core, pyaxon, `examples/build_*_experts.py` |
| [`treinamento`](https://github.com/geraldogrise/treinamento) | dados: as lições, **uma branch por fase** (`fase-1` … `fase-12`); a `main` é vazia de propósito |

O mapa "expert → branch → script" vive em [`axon_colab.py`](./axon_colab.py) — é a única
fonte da tabela, os notebooks importam de lá. `ac.tabela()` imprime tudo.

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
| `deepseek` | `unsloth/deepseek-coder-6.7b-instruct-bnb-4bit` | cabe na T4 grátis em 4-bit |
| `qwen` | `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit` | idem |
| `qwen-3b` | `unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit` | se a T4 estourar memória |

Na T4, 7B em 4-bit com `max_seq_length=2048` roda com `batch_size=1` +
`gradient_accumulation=8`. Se der OOM, caia pro `qwen-3b` ou baixe o `MAX_LEN` pra 1024.

## Legado

`finetune_axon_colab.ipynb` é a versão antiga: pede upload manual de um zip
`treinamento_portugues/` e usa Llama-3.2-3B. Foi substituído pelo
`finetune_expert_colab.ipynb`, que puxa os dados direto da branch e trabalha por expert.
Mantido só como referência.
