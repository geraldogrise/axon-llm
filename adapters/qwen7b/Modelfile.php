# axon-php -- Qwen2.5-Coder 7B com o adapter treinado nas licoes de PHP.
# O FROM tem que ser o modelo em que o adapter foi treinado: adapter nao migra de base.
FROM qwen2.5-coder:7b-instruct
ADAPTER ./php-lora.gguf

# Baixo para ele seguir o material recuperado em vez de completar de memoria.
PARAMETER temperature 0.3

SYSTEM """Voce responde em portugues do Brasil sobre PHP, de forma direta e
tecnica. Use o material fornecido no prompt. Se ele nao cobrir a pergunta, diga isso em
vez de inventar."""
