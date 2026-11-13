# Dados, procedência e calibração

## Arena do usuário

Foram extraídos recortes das três gravações `realsense_livox_arena_{1,2,3}_20261110_*`, preservando os arquivos de origem. Seleção: 20–40 s após a primeira imagem IR, um quadro a cada seis recebidos (aproximadamente 10 Hz). Cada recorte contém 199 pares reais IR1/IR2, PNG sem alteração de resolução, manifest com timestamps de Header, IMUs e referência FAST-LIO de CSV. Os pares são associados por timestamp inteiro em nanossegundos **exatamente igual**, sem reutilização. Não se associa RGB com IR para fabricar estéreo.

Instalação adicional somente para extrair MCAP:

```bash
pip install -r requirements-bags.txt
python scripts/export_arena.py \
  --bag /caminho/realsense_livox_arena_1_20261110_103019 \
  --fastlio-csv /caminho/fast_openvis_arena_1.csv \
  --output data/arena1 --start 20 --duration 20 --stride 6
```

Repetir com Arena 2 (`20261110_103404`) e Arena 3 (`20261110_103659`). O exportador usa `mcap` e `mcap-ros2-support`, sem ROS, replay, movimentação ou conexão com robô. Aceita MCAP ROS2 com schemas. Atualmente espera os tópicos:

- `/kalibr/rgbd/infra1/image_rect_raw/compressed` e `infra2/...`;
- `infra1/camera_info`, `infra2/camera_info`;
- `/kalibr/rgbd/imu`, `gyro/sample`, `accel/sample`, `/kalibr/livox/imu`;
- `/tf_static`.

A referência é extraída das colunas de `/kalibr/fast_lio/odometry` no CSV, usando **Header stamp da odometria**, não o tempo da linha PlotJuggler, nem a saída `odometry_high_freq`. A bag original contém sensores/TF; a trajetória está nos CSVs separados. O exportador não executa FAST-LIO nem OpenVINS.

Cada diretório gerado inclui:

- `frames.csv`, `left/`, `right/`;
- `calibration.json`: projeção **das imagens retificadas**, obtida de `CameraInfo.P`;
- `camera_info_recorded.json` e `tf_static_recorded.json`: fontes de calibração/TF registradas;
- `imu.csv`: timestamps, tópico/frame, gyro e aceleração brutos;
- `reference_fastlio.csv`: pose e frames publicados pelo FAST-LIO;
- `provenance.json`: intervalo, contagens, baseline, arquivos e SHA-256 dos PNGs.

O baseline registrado é **0,050007871777 m**; `fx=fy=388,507293701`, `cx=318,007873535`, `cy=236,957580566`, resolução 640×480. Usar `P` dos pixels retificados evita aplicar novamente a distorção de um YAML Kalibr de outro domínio de imagem. É a calibração informada pelo driver, não uma nova validação física. O `frame_id` da CameraInfo direita também aponta para o frame óptico esquerdo: a geometria foi identificada pelos tópicos/P, e essa particularidade é preservada no JSON original.

As IMUs são preservadas para extensões VIO e inspeção, **não são usadas na odometria visual**. Gyro em rad/s; aceleração RealSense em m/s². A aceleração Livox bruta pode estar em g conforme a cadeia de aquisição; não foi convertida nem usada para inferir escala. Não se inventou calibração temporal/inercial.

A avaliação usa a cadeia TF registrada `livox/imu → livox/link → fcu → rgbd/link → aligned_depth_to_infra1 → infra1_optical` (nomes normalizados sem `/` inicial). Aplica o braço de alavanca e a orientação para comparar o mesmo ponto. As translações de montagem nominais (ex.: câmera 0,1 m à frente do FCU) não foram fisicamente aferidas nesta tarefa. FAST-LIO também tem deriva, hipóteses de calibração e limitações próprias: as métricas são **discrepância entre estimadores**, não ground truth. Nenhum dado de referência volta ao estimador.

As bags e imagens privadas não são versionadas nem publicadas no GitHub. Os recortes são entregues em ZIP separado para importar no Colab. O notebook executa sua demonstração padrão com dados públicos; a seção Arena é opcional e consome o ZIP, sem caminhos exclusivos desta máquina.

## KITTI público para runtime novo

Fonte oficial: [KITTI odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php), sequência **00**, imagens grayscale retificadas e sincronizadas `image_0/image_1`, frames 0–59, `calib.txt`, `times.txt`, poses oficiais. A calibração métrica deriva de `P0` e `P1`; `T_right_left.x=P1[0,3]/fx`, com sinal negativo.

```bash
python scripts/download_kitti.py --frames 60 --output data/kitti00
```

O script usa o [bucket oficial KITTI](https://registry.opendata.aws/kitti/) `avg-kitti`, HTTP Range e diretório ZIP/CRC da biblioteca padrão. Foram transferidos 42,8 MB, incluindo metadados dos ZIPs; **não** o arquivo inteiro de 23 GB. Há até quatro tentativas de rede, validação de `Content-Range`, recusa de respostas completas inesperadas e SHA-256 de cada PNG. Não depende de espelhos comunitários, login ou links temporários. Limite proposital: 20–500 frames por comando; não há download indiscriminado do benchmark.

Licença dos dados: [CC BY-NC-SA 3.0](https://www.cvlibs.net/datasets/kitti/). Adequada à demonstração acadêmica não comercial; não é a licença MIT do código. Não redistribuímos os dados no Git. Citar: Geiger, Lenz, Urtasun, *Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite*, CVPR 2012. O recorte não representa uma avaliação completa KITTI de 100–800 m nem submissão ao leaderboard.

## Vídeo herdado

`video/video_simulacao.mp4` já estava no repositório. Foi inspecionado antes do uso. Somente mono, com calibração herdada explicitamente **UNVERIFIED**. Distorção zero é uma hipótese, não uma calibração medida. Não há trajetória verdadeira para atribuir precisão ao resultado. O teste integral lê 1484 quadros e processa um a cada três (495 imagens).
