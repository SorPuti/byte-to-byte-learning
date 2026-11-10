# byte-to-byte — um GPT em português, do zero

Modelo de linguagem tipo GPT (~4,4 milhões de parâmetros) treinado em obras de Machado de Assis.
Feito só com Python e NumPy: o tokenizador, o backpropagation e o AdamW são escritos à mão.

## 1. Instalar

```bash
uv sync                          # cria o .venv e instala as dependências do pyproject.toml
uv run python train.py --help    # ou: source .venv/bin/activate
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

Treino completo com a configuração do vídeo (75 épocas), na sua máquina:

```bash
python train.py
```

Para rodar na nuvem em vez da sua máquina, veja a seção 5b.

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

## 5b. Treino no Modal (opcional, na nuvem)

O Modal não tem um processo de treino próprio. Ele executa **o mesmo loop** de `src/training.py`
dentro de um container remoto. O ponto de entrada continua sendo o `train.py`:
a flag `--backend` só decide **onde** o código roda.

```
python train.py                    → src/training.py   (seu computador)
python train.py --backend modal    → modal_app.py::train_remote → src/training.py  (container Modal)
modal run modal_app.py             → modal_app.py::main → train_remote → src/training.py  (mesmo caminho)
```

Os dados e checkpoints ficam em dois Volumes do Modal (discos persistentes na nuvem):

| Volume | Conteúdo |
|---|---|
| `byte-to-byte-data` | `corpus_pt.txt` (enviado por você) e `tokenizer.json` (criado na primeira execução) |
| `byte-to-byte-checkpoints` | `epoch_XX.npz`, um por época |

### Passo a passo

**Uma vez só:**

```bash
pip install modal
modal setup                                                       # login na sua conta Modal
python prepare_data.py                                            # gera data/corpus_pt.txt localmente, se ainda não existe
modal volume put byte-to-byte-data data/corpus_pt.txt /corpus_pt.txt
```

**Rodar o treino** (escolha uma forma; as duas executam a mesma coisa):

```bash
python train.py --backend modal --epochs 3          # pela CLI do projeto (recomendado)
modal run modal_app.py --epochs 3                   # direto pelo Modal
```

Teste rápido remoto antes do treino de verdade:

```bash
python train.py --backend modal --epochs 1 --smoke
```

**Retomar:** rode o mesmo comando de novo. O treino continua do último `epoch_XX.npz` do Volume de checkpoints.

**Baixar um checkpoint** para gerar texto localmente:

```bash
modal volume get byte-to-byte-checkpoints epoch_02.npz ./checkpoints/
python -m src.generate --checkpoint checkpoints/epoch_02.npz --tokens 100
```

### Pontos de atenção

- **`--epochs` é obrigatório na prática.** O padrão é 75 épocas (`config.EPOCHS`), ou seja, horas de
  cobrança se você esquecer a flag. Use sempre `--epochs`.
- **Corpus precisa estar no Volume** antes da primeira execução. Se não estiver, o treino falha ao abrir o arquivo.
- **Tokenizer:** se `tokenizer.json` não existe no Volume de dados, o tokenizador é treinado no container
  (mais lento) e salvo lá para as próximas execuções.
- **`--checkpoint-dir` não se aplica** ao backend `modal`; os checkpoints vão sempre para o Volume.
- **Recursos e custo:** vCPUs, memória e timeout estão no topo de `modal_app.py`
  (padrão: 8 vCPUs, 16 GB, 12 h). Confira o preço no painel do Modal antes de rodar.
- **Python 3.14:** confirme que o Modal suporta essa versão. Se não, troque `PYTHON_VERSION` em `modal_app.py`.
- **Ainda não testado contra o Modal real.** Os testes automatizados usam um `modal` falso.
  A primeira execução com `--smoke` é a validação de verdade.

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
train.py             CLI de treino (--backend local|modal) e retomada automática
modal_app.py         entrypoint do treino no Modal (único arquivo que importa modal)
src/training.py      loop de treino independente de ambiente
src/backends/        backends de execução: local e Modal
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
