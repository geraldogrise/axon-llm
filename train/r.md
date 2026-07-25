# R — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre R (estatística/dados).
**Expert sugerido**: `r_experts` ou família em `data_science_experts`. **Total est.**: ~100 lições.
**Convenção**: `treinamento_r/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~30
o que é R e RStudio; vetores; tipos de dados; operadores; atribuição (`<-`); indexação de vetores; fatores (factors); listas; matrizes; data frames; NA e valores faltantes; coerção de tipos; controle de fluxo (`if`/`for`/`while`); funções; apply family (apply/lapply/sapply); vetorização; escopo; pacotes (install/library); pipe (`|>` e `%>%`); datas; strings básicas; ler CSV/Excel; ambiente e workspace.

## tidyverse/ — ~26
introdução ao tidyverse; dplyr: filter/select/mutate; dplyr: group_by/summarise; dplyr: arrange/joins; tidyr: pivot_longer/wider; tidyr: separate/unite; readr; purrr (map); stringr (strings); forcats (factors); lubridate (datas); tibbles; magrittr pipes; manipulação de dados; limpeza de dados; missing data; reshape.

## visualizacao/ — ~18
ggplot2: gramática de gráficos; aes e geoms; scatter plots; bar charts; histogramas; boxplots; line charts; facets; temas e estilos; cores e escalas; anotações; gráficos combinados; plotly (interativo); base R plots; heatmaps; customização.

## estatistica-modelagem/ — ~26
estatística descritiva; distribuições; testes de hipótese (t-test); ANOVA; correlação; regressão linear (`lm`); regressão logística (`glm`); modelos mistos; séries temporais; clustering; PCA; caret (machine learning); random forest; validação cruzada; métricas; inferência; bootstrap; R Markdown (relatórios); Shiny (apps web); reprodutibilidade; boas práticas.
