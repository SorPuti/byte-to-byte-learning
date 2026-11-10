# byte-to-byte — um GPT em português, do zero

Modelo de linguagem tipo GPT (~4,4 milhões de parâmetros) treinado em obras de Machado de Assis.
Feito só com Python e NumPy: o tokenizador, o backpropagation e o AdamW são escritos à mão.

## 1. Instalar

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Gerar o dataset em português

```bash
python prepare_data.py
```

Isso baixa 14 obras de Machado de Assis do Projeto Gutenberg (domínio público)
e junta tudo em `data/corpus_pt.txt` (~4,7 milhões de caracteres).
Os downloads ficam em `data/raw/`, então rodar de novo é rápido.

## 3. Conferir que está tudo certo

```bash
python -m pytest -q
```

Deve aparecer `passed` e nenhum `failed`.

## 4. Teste rápido (recomendado primeiro)

```bash
python train.py --smoke --epochs 1 --checkpoint-dir checkpoints_smoke
python -m src.generate --checkpoint checkpoints_smoke/epoch_00.npz --tokens 100
```

O primeiro comando leva uns 2 minutos (inclui treinar o tokenizador, na primeira vez).
O texto gerado nesse teste ainda vai ser ruim, porque o modelo viu só 5 passos. É esperado.

## 5. Treino

Treino completo com a configuração do vídeo (75 épocas):

```bash
python train.py
```

Para um teste de verdade sem esperar dias, rode poucas épocas:

```bash
python train.py --epochs 3
```

**Tempo estimado nesta máquina:** cerca de 15 segundos por passo, ~200 passos por época,
ou seja ~50 minutos por época. As 75 épocas levam ~60 horas.
Com `--epochs 3` são ~2,5 horas.

Os checkpoints são salvos em `checkpoints/epoch_XX.npz` a cada época.
Se você rodar `python train.py` de novo, ele continua de onde parou.

Observação: o cronograma de aprendizado (cosine) é calculado para 75 épocas.
Com `--epochs 3` o aprendizado fica alto durante o teste, então o resultado não é o final.

## 6. Gerar texto

```bash
python -m src.generate --checkpoint checkpoints/epoch_74.npz --prompt "— Mas, senhora" --tokens 200 --temp 0.8
```

- `--prompt`: começo do texto (use caracteres que existem no livro, como letras e pontuação portuguesa).
- `--tokens`: quantos tokens gerar.
- `--temp`: criatividade. Valores menores (0,5) são mais conservadores; maiores (1,0) são mais variados.

## Estrutura

```
prepare_data.py      baixa e monta o dataset em português
config.py            todos os hiperparâmetros
train.py             loop de treino (com retomada automática)
src/tokenizer.py     tokenizador BPE
src/gpt.py           modelo GPT com backpropagation manual
src/optimizer.py     AdamW e cosine learning rate
src/dataset.py       sorteio de lotes
src/generate.py      geração de texto
tests/               testes automatizados
docs/superpowers/    spec e plano de implementação
```

## Limitações

- Só caracteres que aparecem no corpus podem ser usados no prompt. Outro caractere dá erro.
- Treino só em CPU, com NumPy. É lento por design: o objetivo é entender cada passo.
