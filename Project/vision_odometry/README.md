# Odometria visual monocular e estéreo validada

Implementações independentes em **C++17 e Python**, sem janelas. Mono inicializa um mapa de escala arbitrária por duas vistas e mantém a escala relativa por PnP; estéreo reconstrói profundidade com pares sincronizados e baseline calibrado. Ambos exportam pose, estado, qualidade e segmentos em CSV.

[Colab de destino](https://colab.research.google.com/drive/1EHjk83__QW-TeIksE1VfLCRa2Z5ywIWf) · [Notebook para importar](notebooks/odometria_mono_stereo_colab.ipynb) · [Erros reproduzidos](docs/ERRORS.md) · [Dados e calibração](docs/DATA.md) · [Resultados executados](docs/VALIDATION.md).

## Instalação e testes

Ubuntu/Colab CPU; OpenCV ≥4.5. Python testado em 3.10. Dependências Python fixam OpenCV/NumPy para evitar a incompatibilidade ABI observada nesta máquina. Não exige ROS para estimar odometria ou ler os recortes exportados.

```bash
cd Project/vision_odometry
sudo apt-get update
sudo apt-get install -y g++ cmake pkg-config libopencv-dev python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make
ctest --test-dir build --output-on-failure
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
```

`make clean` remove somente `build/`, preservando vídeo, configurações, dados e resultados. O build utiliza core, imgproc, imgcodecs, features2d, calib3d e videoio; sem viz, HighGUI ou Eigen. Executáveis/objetos antigos já versionados foram removidos da branch e agora são ignorados. A CI executa os testes sintéticos e de entrada, sem baixar datasets.

## Executar os dois modos

Obter um conjunto público pequeno (60 pares oficiais KITTI 00, ~183 MB transferidos com leitura antecipada) ou extrair as bags da Arena conforme [DATA.md](docs/DATA.md):

```bash
python scripts/download_kitti.py --frames 60 --output data/kitti00

# Python monocular: só usa as imagens da coluna left
python python/vo.py --mode mono \
  --manifest data/kitti00/frames.csv --calibration data/kitti00/calibration.json \
  --output results/python_mono.csv

# Python estéreo: exige left e right realmente sincronizadas
python python/vo.py --mode stereo \
  --manifest data/kitti00/frames.csv --calibration data/kitti00/calibration.json \
  --output results/python_stereo.csv

# C++ tem a mesma interface
build/vision_odometry --mode mono \
  --manifest data/kitti00/frames.csv --calibration data/kitti00/calibration.json \
  --output results/cpp_mono.csv
build/vision_odometry --mode stereo \
  --manifest data/kitti00/frames.csv --calibration data/kitti00/calibration.json \
  --output results/cpp_stereo.csv

# Vídeo herdado: somente mono; calibração explicitamente NÃO VERIFICADA
python python/vo.py --mode mono --video video/video_simulacao.mp4 \
  --calibration config/legacy_video_UNVERIFIED.json \
  --stride 3 --output results/video_mono.csv
```

Para usar Arena 1, substituir manifest/calibração por `data/arena1/frames.csv` e `data/arena1/calibration.json`; idem `arena2` e `arena3`. Flags opcionais comuns: `--stride N`, `--max-frames N` (0 = até o fim), `--sync-tolerance 0.002` em segundos. Vídeo e manifest são mutuamente exclusivos; estéreo não aceita vídeo único. Erros de entrada terminam com código 2 e diagnóstico em stderr. Não há fallback para calibração ou escala inventadas.

## Entradas

Manifest CSV UTF-8, caminhos relativos ao próprio manifest (ou absolutos), índice e timestamp estritamente crescentes:

```csv
index,timestamp,left,right,right_timestamp
0,0.000000000,left/000000.png,right/000000.png,0.000000000
1,0.100000000,left/000001.png,right/000001.png,0.100000000
```

Mono requer `index,timestamp,left`; ignora direita, IMU e referência. Estéreo requer também `right,right_timestamp`, arquivos distintos não reutilizados e diferença de timestamps dentro da tolerância. Os pixels precisam ter exatamente a resolução calibrada. A tolerância não comprova sincronização física e não substitui aquisição sincronizada. Disparidade/erro epipolar adicionais ajudam a rejeitar pares inconsistentes. Não se cria estéreo duplicando, deslocando imagens ou pareando instantes sucessivos.

Calibração JSON obrigatória. Ver [synthetic.json](config/synthetic.json) para o esquema completo **apenas exemplificativo**, sem validade física. Campos:

| Campo | Mono | Estéreo | Significado |
|---|---|---|---|
| `width,height` | exige | exige | resolução exata, sem redimensionamento implícito |
| `K_left,D_left` | exige | exige | matriz 3×3 e distorção pinhole OpenCV (4/5/8/12/14 coeficientes) |
| `K_right,D_right` | — | exige | câmera direita |
| `R_right_left,T_right_left` | — | exige | `X_right = R_right_left X_left + T_right_left` |
| `translation_unit` | saída `relative` | exige | `m`, `cm`, `mm` ou `arbitrary`; saída na mesma unidade |
| `already_rectified` | ignora | opcional | true somente com K comuns, D=0, R=I e T=(-baseline,0,0) |
| `provenance` | recomendado | recomendado | origem/validade dos parâmetros |

Fluxos pinhole horizontais, câmera direita no +x da esquerda. O pipeline recusa extrínsecas inválidas, baseline zero e outras geometrias não suportadas. Dados raw são retificados com `stereoRectify`/`initUndistortRectifyMap`; verifica-se o alinhamento epipolar observado. Dados já retificados usam sua calibração de projeção, sem repetir a correção de distorção. Para entrada raw estéreo, a pose se refere à **câmera esquerda retificada**; `R1` é a rotação da câmera raw para a retificada. Uma referência externa deve ser convertida para esse mesmo frame, usando `R_body_rect=R_body_raw R1.T`. O avaliador padrão Arena/KITTI usa fluxos já retificados (`R1=I`). Fisheye/equidistant não está implementado.

## Geometria e arquitetura

`python/vo.py` e `include/odometry.hpp`/`src/odometry.cpp` implementam as mesmas etapas; `src/main.cpp` é a CLI C++. Os scripts de dados e avaliação são compartilhados. Referência e IMU não são acessadas pelos estimadores.

- Features SIFT (até ~2400), matcher L2 com razão 0,75 e associação mútua, sem limite artificial de deslocamento em pixels. Estado imagem/features/descritores é associado ao keyframe. Ausência de descritores e knn com menos de dois vizinhos são tratados.
- Mono: RANSAC da matriz essencial com limiar 1 px, mínimo 20 pontos, fração mínima 45%. A máscara é passada a `recoverPose`, e intersectada com cheiralidade, triangulação, reprojeção e paralaxe compensada pela rotação. Exige cheiralidade ≥70%, paralaxe mediana ≥1 px, cobertura espacial e mapa não planar. Homografia de rotação detecta rotação pura antes da inicialização; movimento mediano <0,4 px é tratado como repouso/baixo movimento. Nada é acumulado nessas rejeições.
- A norma unitária de `t` define **somente o gauge da primeira baseline**. Landmarks triangulados ficam no mundo do segmento. Nas vistas seguintes, suas observações 2D alimentam `solvePnPRansac` + `solvePnPRefineLM`. Não há soma recorrente de direções unitárias. Keyframes são renovados depois de oito imagens ou perda de suporte; pontos existentes são propagados e novos pontos triangulados com a baseline já estimada no mesmo mapa.
- Estéreo: correspondências esquerda/direita após retificação; erro epipolar mediano ≤1,5 px e pelo menos 60% abaixo de 2 px, disparidade >0,5 px, profundidade positiva em ambas as câmeras, reprojeção <2 px. Os pontos do quadro anterior são colocados no mundo e associados à esquerda atual. PnP estima a pose; um novo mapa estéreo é construído em cada quadro aceito.
- PnP: ≥20 inliers, fração ≥45%, cheiralidade ≥90%, reprojeção mediana ≤1,5 px, erros individuais <2 px e cobertura espacial. Refinamento não recupera pontos excluídos pelo RANSAC. Em ambos os modos, cobertura exige ≥6 células de uma grade 4×3 e extensão ≥20% em cada dimensão; é um gate conservador, documentado nas perdas reais.

Eixos ópticos: x direita, y baixo, z frente. Em cada segmento, `R_wc` é orientação câmera→mundo e `C_w` a posição do centro da câmera. Para `recoverPose` anterior→atual:

```text
X_atual = R_rel * X_anterior + t_rel
R_wc_new = R_wc * R_rel.T
C_w_new = C_w - escala * R_wc_new * t_rel
```

PnP com landmarks **mundiais** retorna `X_camera = R_cw X_world + t_cw`; portanto `R_wc=R_cw.T`, `C_w=-R_cw.T t_cw`. Os testes usam câmeras rotacionadas para verificar sinal, ordem, escala e convenção.

Três falhas consecutivas de tracking abrem novo segmento, limpam o mapa e estabelecem uma nova origem/escala. Estados geométricos sem observabilidade antes da inicialização (`STATIONARY`, `ROTATION_ONLY`, `PLANAR`, `DEGENERATE`) aguardam uma vista útil. Estimativas rejeitadas preservam a pose/mapa anteriores; a única mudança de origem ocorre explicitamente em `REINITIALIZING` com novo `segment` e `pose_valid=0`. Não há relocalização entre mapas.

## CSV e avaliação

`R_wc` é exportada em nove colunas `r00..r22`; posição em `x,y,z`. `accepted` indica se a estimativa daquele quadro foi aceita; `pose_valid` informa se existe pose inicializada no segmento, mesmo quando uma linha posterior apenas mantém a última pose. Assim, **uma pose válida com accepted=0 não é nova medição**. `ANCHOR` estéreo tem pose válida na origem; `ANCHOR` mono ainda não tem mapa/escala. Campos sem observabilidade são `nan`, sem números artificiais.

```bash
python scripts/evaluate.py --trajectory results/python_mono.csv \
  --reference data/kitti00/reference.csv --mode mono --output results/mono_eval
python scripts/evaluate.py --trajectory results/python_stereo.csv \
  --reference data/kitti00/reference.csv --mode stereo --output results/stereo_eval

# Discrepância Arena com referência FAST-LIO e conversão do ponto físico:
python scripts/evaluate.py --trajectory results/arena1_python_stereo.csv \
  --reference data/arena1/reference_fastlio.csv --mode stereo --reference-kind fastlio \
  --tf-static data/arena1/tf_static_recorded.json --output results/arena1_eval
```

Avaliação posterior e segmentada: mono Sim(3) com escala global por segmento; estéreo SE(3) sem mudar escala. Exporta JSON, CSV de poses alinhadas e gráficos de trajetória/inliers/resíduos. Computa RMSE de posição e RPE (~1 s), sem referência na estimação. Relata cobertura e segmentos recusados; não conecta segmentos nem interpola a VO através de perdas. Referência interpolada por posição/SO(3), sem extrapolação nem gaps >0,35 s. Para fluxos raw, converter antes a referência ao frame retificado, conforme a convenção acima.

## Resultados e limites

Foram executadas **18 sequências/métodos**, **54 testes Python** e a suíte C++. Na Arena, o estéreo manteve um segmento em todos os recortes. O monocular perdeu/reinicializou mapas em partes das Arenas 1 e 3; C++ e Python divergem em alguns segmentos, com OpenCV 4.5.4/4.11.0. No KITTI 00 (60 pares), RMSE estéreo ≈0,317 m com escala fixa; mono ≈0,149 m Python / 0,163 m C++ após Sim(3), não uma comparação de precisão sob as mesmas restrições. [Todas as contagens, alinhamentos, RPEs e limitações estão documentadas](docs/VALIDATION.md).

FAST-LIO na Arena é referência estimada, com TF nominal de montagem, **não ground truth**. Vídeo herdado não tem calibração validada nem referência de trajetória. Build, execução e testes não demonstram qualidade física de calibração ou superioridade científica. Não há IMU, bundle adjustment, loop closure, rolling-shutter, relocalização global ou garantias em cenas móveis/sem textura. O Colab padrão usa dados públicos; a seção opcional Arena usa os recortes privados entregues em ZIP, nunca as bags inteiras no Git.

Base técnica: [convenções PnP/recoverPose/retificação no OpenCV](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html). Código sob a licença MIT da raiz; licença KITTI separada e não comercial.
