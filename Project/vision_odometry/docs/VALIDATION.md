# Validação executada em 13/11/2026

Implementação: `a0c13851fa275fbf06518e431c4a6f7dec3b9d6f`. Branch `feat/validated-mono-stereo-odometry`; main preservada. Os fontes posteriores foram apenas formatados antes do commit.

## Ambiente e verificações

Python 3.10.12, NumPy 1.26.4, OpenCV Python 4.11.0; OpenCV C++ 4.5.4, g++ (Ubuntu 9.5.0-1ubuntu1~22.04.1) 9.5.0. Sem GUI/GPU.

`make clean`, build Release, CTest (1 executável, oito grupos geométricos) e **54 testes pytest**, todos passaram. Quatorze casos pytest executam as interfaces C++/Python reais. Asserções C++ usam `CHECK` com exceção, permanecendo ativas em Release. [Logs e ambiente](evidence/). CI foi configurada; seu estado remoto é reportado separadamente.

## Dados e cobertura real

Arena 1/2/3: **199 pares por sequência**, recorte de 20 s, 20–40 s após a primeira imagem IR, stride 6 (~10 Hz). Sincronização exata por Header ns, zero pares selecionados sem direita. Baseline CameraInfo.P = 0,050007871777 m. Referência FAST-LIO do CSV, timestamps de Header e TF nominal para o centro óptico esquerdo. KITTI 00: **60 pares**, quadros 0–59, referência oficial. Vídeo herdado: **495 quadros processados**, stride 3, cobrindo os 1484 quadros/49,467 s. No total, **18 execuções reais**: quatro por dataset pareado e duas monoculares no vídeo. Todas terminaram com exit 0.

| Sequência | Versão | Modo | Aceitos/total | Segmentos | Poses pareadas no maior trecho | RMSE posição maior trecho [m] | Escala de alinhamento |
|---|---|---|---:|---:|---:|---:|---:|
| arena1 | cpp | mono | 155/199 | 5 | 146 | 0.0186 | 0.124790 |
| arena1 | cpp | stereo | 196/199 | 1 | 194 | 0.0392 | 1.000000 |
| arena1 | python | mono | 153/199 | 6 | 145 | 0.0189 | 0.124794 |
| arena1 | python | stereo | 196/199 | 1 | 194 | 0.0391 | 1.000000 |
| arena2 | cpp | mono | 116/199 | 1 | 115 | 0.0389 | 0.187393 |
| arena2 | cpp | stereo | 199/199 | 1 | 197 | 0.0378 | 1.000000 |
| arena2 | python | mono | 116/199 | 1 | 115 | 0.0393 | 0.187861 |
| arena2 | python | stereo | 199/199 | 1 | 197 | 0.0378 | 1.000000 |
| arena3 | cpp | mono | 166/199 | 4 | 157 | 0.0274 | 0.096833 |
| arena3 | cpp | stereo | 197/199 | 1 | 195 | 0.0310 | 1.000000 |
| arena3 | python | mono | 174/199 | 2 | 159 | 0.0326 | 0.123417 |
| arena3 | python | stereo | 198/199 | 1 | 196 | 0.0306 | 1.000000 |
| kitti00 | cpp | mono | 59/60 | 1 | 59 | 0.1632 | 0.722937 |
| kitti00 | cpp | stereo | 60/60 | 1 | 60 | 0.3166 | 1.000000 |
| kitti00 | python | mono | 59/60 | 1 | 59 | 0.1492 | 0.722400 |
| kitti00 | python | stereo | 60/60 | 1 | 60 | 0.3165 | 1.000000 |

**Arena: RMSE é discrepância em relação ao FAST-LIO, não precisão/ATE contra verdade física.** O maior trecho é destacado para leitura, não somado ou unido aos outros mapas. O [JSON completo](evidence/real-validation-summary.json) inclui todos os segmentos, inclusive os curtos demais para avaliação. Os CSVs completos e gráficos estão na entrega local.

**Mono: Sim(3)** ajustado em cada segmento independente usando todas as poses pareadas desse segmento; a escala absoluta continua desconhecida. **Estéreo: SE(3), escala fixada em 1**. O alinhamento é apenas para avaliação; não retroalimenta as estimativas. A menor RMSE mono não comprova superioridade, pois houve ajuste de escala, coberturas e inicializações diferentes.

## Movimento relativo (1 s), maior segmento de cada execução

| Execução | Pares RPE | RMSE translação [m] | RMSE rotação [graus] |
|---|---:|---:|---:|
| arena1_cpp_mono | 136 | 0.0219 | 0.7129 |
| arena1_cpp_stereo | 181 | 0.0141 | 0.6689 |
| arena1_python_mono | 135 | 0.0216 | 0.6966 |
| arena1_python_stereo | 181 | 0.0141 | 0.6692 |
| arena2_cpp_mono | 105 | 0.0425 | 0.8798 |
| arena2_cpp_stereo | 187 | 0.0120 | 0.6387 |
| arena2_python_mono | 105 | 0.0427 | 0.8866 |
| arena2_python_stereo | 187 | 0.0120 | 0.6387 |
| arena3_cpp_mono | 146 | 0.0212 | 0.9147 |
| arena3_cpp_stereo | 184 | 0.0143 | 0.8445 |
| arena3_python_mono | 149 | 0.0238 | 0.9431 |
| arena3_python_stereo | 185 | 0.0142 | 0.8417 |
| kitti00_cpp_mono | 49 | 0.2625 | 0.5852 |
| kitti00_cpp_stereo | 50 | 0.3426 | 0.5311 |
| kitti00_python_mono | 49 | 0.2474 | 0.5778 |
| kitti00_python_stereo | 50 | 0.3425 | 0.5310 |

Referência interpolada por posição e SO(3) geodésico, sem extrapolação; lacunas >0,35 s são excluídas. RPE considera aproximadamente 1 s (tolerância 0,08 s), sem atravessar gaps de poses aceitas >0,35 s ou segmentos. Segmentos com <8 poses pareadas ou movimento insuficiente não são alinhados. A amostra curta não representa o protocolo KITTI de 100–800 m.

## Vídeo herdado sem referência

| Versão | Aceitos | Quadros processados | Segmentos |
|---|---:|---:|---:|
| python | 438 | 495 | 2 |
| cpp | 433 | 495 | 3 |

Perfil intrínseco herdado **não verificado**; distorção zero assumida. Apenas execução funcional e rastreamento, sem alegação de precisão. Não foi tratado como estéreo.

## Limitações observadas

- O monocular perdeu o mapa em partes das Arenas 1 e 3 e abriu segmentos; não há continuidade ou escala inventadas entre eles. A diferença C++/Python é compatível com OpenCV/SIFT/RANSAC distintos e não é mascarada por alinhamento entre segmentos.
- Os testes sintéticos não validam a calibração física. CameraInfo é a calibração registrada pelo driver; TF de montagem é nominal e FAST-LIO não é verdade absoluta. O recorte não valida o percurso completo de nenhuma Arena.
- Sem bundle adjustment, loop closure, IMU, modelo de objetos móveis, rolling shutter ou relocalização global. Pontos de mapa têm filtros geométricos, mas não otimização conjunta. Inicializações planares/sem paralaxe são deliberadamente recusadas.
- Sincronização de Header e erro epipolar são verificáveis; sem metadados físicos adicionais não detectamos toda defasagem de exposição, clock offset ou baseline incorreto.
- As métricas de execução capturadas em paralelo são tempo de parede desta máquina, não benchmark de desempenho em tempo real.

## Repetir

```bash
make
ctest --test-dir build --output-on-failure
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
python scripts/download_kitti.py --frames 60
# Exportar as três Arenas conforme DATA.md antes do comando seguinte:
python scripts/run_real_validation.py --datasets arena1 arena2 arena3 kitti00 --jobs 2 --legacy-video
```

O notebook possui execução padrão pública sem arquivos locais e seção opcional para os ZIPs da Arena. A validação do notebook e a gravação no Colab são registradas em `evidence/notebook-validation.json`.


## Notebook e gravação no Colab

O notebook final processou **as 16 células**, com **sete células de código não vazias executadas sem erro**, no ambiente local isolado. O bootstrap foi exercitado com clone esparso e ambiente Python novos, build C++, instalação de dependências e download oficial completo do recorte; a repetição final desde o início reutilizou apenas o cache cujos 120 hashes foram conferidos e eram idênticos ao primeiro download independente. A seção opcional de importação do ZIP Arena também foi executada, com os 12 casos das três Arenas.

A execução integral descobriu um backend Matplotlib herdado do kernel incompatível com o ambiente CLI isolado. A revisão `a7ee87e` corrige o problema antes da importação e adiciona teste de regressão. A CI pública dessa revisão passou. O notebook fixa essa revisão imutável; a lógica dos estimadores é a mesma da revisão das 18 execuções originais.

O arquivo do **Colab de destino foi atualizado**, preservando a célula original. A leitura posterior confirmou identidade dos bytes e SHA-256. **Não foi executado em um servidor hospedado pelo Google**; a validação de execução ocorreu localmente em ambiente isolado. Nenhum bloqueio de autenticação permaneceu para publicar a branch ou gravar o Colab. [Registro detalhado](evidence/notebook-validation.json), [gravação verificada](evidence/colab-save-verification.json) e [CI](evidence/ci-verification.json).
