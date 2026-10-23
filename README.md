# face_filter

Filtra fotos: **rejeita a foto se a pessoa A estiver nela**.

Pipeline local de verificação facial (1:1) usando `face_recognition` (dlib).
Você cadastra fotos de referência da pessoa A; o script decide se uma foto
qualquer pode ser usada ou deve ser descartada.

## Instalação (Windows)

Python 3.13 não tem wheel oficial de `dlib` no PyPI (build exige Visual C++).
Use a wheel pré-compilada:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install "https://github.com/sachadee/Dlib/raw/main/dlib-20.0.0-cp313-cp313-win_amd64.whl"
pip install "setuptools<81"              # face_recognition_models usa pkg_resources
pip install git+https://github.com/ageitgey/face_recognition_models
pip install face_recognition opencv-python
```

Em Python 3.10–3.12 ou Linux/macOS, `pip install face_recognition` funciona direto.

## Uso

```powershell
# 1. Coloque 2-5 fotos da pessoa A em refs\pessoa_a\ (ângulos/iluminação variados)
python face_filter.py build_db --refs_dir refs

# 2. Verifique uma foto
python face_filter.py check --photo foto.jpg
```

### Opções do `check`

| Opção | Default | Descrição |
|---|---|---|
| `--person` | — | Verifica somente uma pessoa específica do banco |
| `--tolerance` | 0.6 | Limiar de distância (menor = mais estrito) |
| `--model` | hog | `hog` (CPU) ou `cnn` (GPU, mais preciso) |
| `--db_path` | encodings.pkl | Caminho do banco de encodings |

### Exit codes

| Código | Significado |
|---|---|
| 0 | Aprovada — pessoa A **não** está na foto |
| 1 | Rejeitada — pessoa A **detectada** |
| 2 | Erro de uso/processamento |

Facilita uso em pipelines: `python face_filter.py check --photo foto.jpg && usar_foto || descartar`.

## Resultados dos testes (dataset deepface + ageitgey)

| Teste | Distância | Decisão |
|---|---|---|
| Mesma pessoa (ref vs img5) | 0.320 | REJEITAR ✓ |
| Pessoa diferente (img42) | 0.726 | APROVAR ✓ |
| Grupo Obama+Biden | 0.96–1.04 | APROVAR ✓ |
| Obama solo | 0.986 | APROVAR ✓ |

## Ajuste fino

- **Falso positivo** (rejeita foto de outra pessoa): aumente `--tolerance` (ex. 0.65)
- **Falso negativo** (pessoa parecida passa): diminua (ex. 0.5)
- Teste com fotos reais da sua pessoa A antes de definir o valor final.

## Limitações

- Modelo treinado em adultos — desempenho menor com crianças
- Acurácia varia entre grupos étnicos (ver wiki do face_recognition)
- Verificação Facial 1:1 é *verificação*, não identificação de pessoas desconhecidas

## Privacidade / LGPD

Tudo roda localmente: nenhuma foto sai da máquina. Dados biométricos são
dados sensíveis ( LGPD art. 5º, II; GDPR art. 9) — use apenas com fotos e
pessoas sobre as quais você tem base legal para processar.