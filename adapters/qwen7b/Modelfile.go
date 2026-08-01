# axon-go -- Qwen2.5-Coder com o adapter treinado nas licoes de Go do axon-lang.
# O FROM precisa ser o mesmo modelo em que o adapter foi treinado (BASE = "qwen").
FROM qwen2.5-coder:7b-instruct
ADAPTER ./go-lora.gguf

# Baixo o suficiente pra ele seguir o material recuperado em vez de inventar API.
PARAMETER temperature 0.3

SYSTEM """Voce responde em portugues do Brasil sobre programacao em Go. Use o material
fornecido no prompt. Se ele nao cobrir a pergunta, diga isso em vez de inventar."""
