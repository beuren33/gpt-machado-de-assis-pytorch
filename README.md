# GPT em Nível de Caractere Treinado em Português com PyTorch

Este projeto implementa do zero um modelo de linguagem no estilo GPT, em PyTorch, que aprende a gerar texto em português prevendo um caractere de cada vez. O treino foi feito em duas etapas: primeiro um pré-treino em artigos da Wikipédia em português, para o modelo aprender o idioma de forma geral, e depois um fine-tuning em obras de Machado de Assis, para puxar o estilo para o do autor. Um exemplo prático do uso seria dar um caractere inicial ao modelo e pedir que ele continue escrevendo, e o texto que sai segue a ortografia, a pontuação e o ritmo do que viu no treino.

## Como funciona

A arquitetura segue a do artigo "Attention is All You Need", na versão apresentada no curso, só com o decoder do Transformer. Cada caractere vira um índice, que passa por uma tabela de embeddings, e a posição dele no contexto passa por outra tabela de embeddings de posição. A soma dos dois entra numa pilha de blocos, e cada bloco tem uma camada de atenção com várias cabeças (multi-head attention) e uma rede feed-forward, ambas com LayerNorm e conexão residual. Na atenção, cada token gera um vetor key, um query e um value: o query diz o que o token procura, o key diz o que ele tem a oferecer e a afinidade entre os dois decide quanto de cada token anterior entra na resposta. Ao final, uma camada linear devolve a pontuação de cada caractere do vocabulário para escolher o próximo.

Os hiperparâmetros usados no pré-treino foram estes:

| Parâmetro | Valor |
| --- | --- |
| Contexto (block_size) | 364 caracteres |
| Batch | 32 |
| Dimensão do embedding | 256 |
| Cabeças de atenção | 16 |
| Blocos empilhados | 8 |
| Dropout | 0,4 |
| Otimizador | AdamW, taxa de aprendizado 0,001 |
| Iterações | 6.000 |
| Parâmetros | cerca de 6,9 milhões |

Foi usado ainda o ReduceLROnPlateau, que reduz pela metade a taxa de aprendizado quando a loss de validação não melhora em 3 avaliações seguidas.

## Dados

O pré-treino usa a Wikipédia em português (wikimedia/wikipedia, versão 20231101.pt, carregada por streaming), limpa por expressões regulares (REGEX) com técnicas vistas no card de NLP: remoção de marcadores de referência como [1], de espaços repetidos e de linhas em branco em excesso, ficando só artigos com pelo menos 200 caracteres. Sobraram 982.003 artigos, e como o conjunto é grande a limpeza foi feita em streaming, escrevendo direto no arquivo conforme limpava, pois guardar tudo em memória não foi possível nem na GPU do Colab. Do arquivo final foram usados os primeiros 25 milhões de caracteres, divididos em 90% para treino e 10% para validação.

O fine-tuning usa o corpus de Machado de Assis do NLTK, com cerca de 14,5 milhões de caracteres depois da remoção dos cabeçalhos das obras e da limpeza dos espaços, também separado em 90% para treino e 10% para validação. O vocabulário do pré-treino foi reaproveitado (arquivo 'modelo/vocab.json'), e antes de treinar foi checado que o corpus do Machado não tinha nenhum caractere fora dele.

## Resultados

No pré-treino a loss de validação caiu de 6,93 (modelo aleatório) para 1,36 depois de 6.000 iterações, com loss de treino de 1,27. No fine-tuning, com taxa de aprendizado bem menor (0,00003) e só 500 iterações, a loss de validação no corpus do Machado foi de 1,71 para 1,52.

| Etapa | Loss de validação inicial | Loss de validação final |
| --- | --- | --- |
| Pré-treino (Wikipédia pt) | 6,93 | 1,36 |
| Fine-tuning (Machado de Assis) | 1,71 | 1,52 |

O texto gerado imita bem a forma do português, com pontuação, parágrafos, maiúsculas e palavras que parecem reais, mas muitas das palavras são inventadas e as frases não têm sentido. Esse comportamento é esperado para um modelo de apenas 6,9 milhões de parâmetros que enxerga caractere por caractere e treinou por pouco tempo, e ele serve mais para entender o mecanismo do GPT do que para gerar texto útil.

## Como rodar

Instale as dependências:

```bash
pip install -r requirements.txt
```

Os notebooks foram executados no Google Colab com GPU e leem os arquivos do Google Drive, então os caminhos (por exemplo '/content/drive/MyDrive/...') precisam ser ajustados para o seu ambiente. A ordem é esta:

1. 'notebook/limpeza_wikipedia.ipynb' gera o arquivo de texto da Wikipédia.
2. 'scripts/corpus_machado.py' baixa o corpus de Machado de Assis pelo NLTK e gera o 'machado.txt'.
3. 'notebook/gpt_pretrain_finetune.ipynb' faz o pré-treino, salva o 'pretrain.pt' e o 'vocab.json', e depois faz o fine-tuning, salvando o 'finetune.pt'.

Os pesos ('.pt', cerca de 96 MB cada) e os textos não estão no repositório. Para gerar texto basta recriar o modelo com os mesmos hiperparâmetros, carregar o 'finetune.pt' e chamar o método 'generate'.

## Estrutura do projeto

```
gpt-machado-de-assis-pytorch/
├── notebook/
│   ├── limpeza_wikipedia.ipynb       # limpeza da Wikipedia em streaming
│   └── gpt_pretrain_finetune.ipynb   # modelo, pre-treino e fine-tuning
├── scripts/
│   └── corpus_machado.py             # baixa e limpa o corpus do Machado
├── modelo/
│   └── vocab.json                    # vocabulario de caracteres
├── requirements.txt
└── .gitignore
```

## Observações e próximos passos

Como próximos passos, vale trocar o nível de caractere por um tokenizador de subpalavras, que reduz o contexto necessário e deixa as palavras mais consistentes, treinar por mais iterações com mais dados da Wikipédia (só 25 milhões de caracteres foram usados) e avaliar o fine-tuning de forma mais objetiva, por exemplo com perplexidade, já que a avaliação do texto gerado foi só visual.

Os resultados deste projeto podem ser melhores. O foco foi o aprendizado do conteúdo do bootcamp e não obter o melhor desempenho possível, por isso não foi feita uma busca exaustiva por hiperparâmetros, arquiteturas ou dados.
