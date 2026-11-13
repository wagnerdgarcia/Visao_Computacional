# Auditoria do código original

Base imutável: `3d347e2ce05aab3ec33f911dd35b58f9e45b3411` (main ao iniciar esta tarefa). A cópia isolada estava limpa. Nenhum checkout de trabalho do usuário foi modificado. Não havia `AGENTS.md` no repositório nem nos diretórios ancestrais inspecionados.

Reproduzir sem GUI, sem alterar o checkout:

```bash
python scripts/reproduce_before.py
```

O script extrai os arquivos antigos com `git show`, compila em um diretório temporário e escreve [reproduction-before.json](evidence/reproduction-before.json). Essa reprodução requer o histórico Git, OpenCV C++ e Python. Os primeiros registros de build, clean e probe foram capturados **antes de substituir os fontes**; o script posterior os torna repetíveis no commit original.

| Hipótese | Evidência e conclusão |
|---|---|
| Build quebrado | **Não confirmado:** o `make -B` original compila com OpenCV 4.5.4 nesta máquina. O uso de viz e caminhos de Eigen é desnecessário, mas não foi a causa de falha aqui. |
| `make clean` | **Confirmado por execução:** exit 2, `Makefile:36: unterminated variable reference`. |
| Média sem matches | **Confirmado por execução:** três resultados `NaN`, `finite=0`, na chamada com vetores vazios. Não foi uma exceção de divisão inteira. |
| Acesso a vizinhos inexistentes | **Risco confirmado por inspeção:** acesso direto a `[0]`/`[1]`, sem checar tamanho. Com somente um descritor de destino, o teste observou uma **exceção do FLANN anterior ao acesso**, não um crash por acesso fora dos limites. |
| Estado das features | **Confirmado por execução e inspeção:** o retorno antecipado com quadro sem textura preserva saídas antigas (`stale_output_sizes_after_blank=2,2`). Os descritores antigos podem então corresponder a uma imagem diferente de `prev_frame`. Além disso, `processVideos()` limpa pontos usando um vetor local `keypoints` que nunca é preenchido. |
| Máscara de `findEssentialMat` descartada | **Confirmado por inspeção:** a sobrecarga chamada não recebe máscara de saída. Não há filtragem posterior pelos inliers RANSAC. |
| Retorno/máscara de `recoverPose` ignorados | **Confirmado por inspeção:** não há máscara nem teste do retorno, da cheiralidade ou da paralaxe. Não afirmamos que toda pose antiga estava incorreta. |
| Referenciais/sinal | **Confirmado algebricamente e por contraprova numérica:** para câmera já girada 90° e `R_rel=I,t_rel=(1,0,0)`, o deslocamento antigo é `(0,+1,0)`, enquanto a posição câmera→mundo exige `(0,-1,0)`. A multiplicação de orientação também não implementa `R_wc_new=R_wc R_rel.T`. |
| Translação unitária | **Confirmado por inspeção:** a direção de norma unitária de `recoverPose` é somada a cada atualização sem escala observável. Não representa distância real nem preserva velocidades relativas. |
| Filtros excessivos | **Efeito confirmado no vídeo:** gates `ratio=.4`, distância SIFT `<30`, movimento `<10 px`. Entre quadros 150→165: 0 candidatas antigas contra 202 com razão `.75` + mutual nearest. São candidatas, não precisão nem inliers geométricos; não se atribui toda perda antiga somente ao matcher. |
| Calibração fixa | **Confirmado:** intrínsecos fixos no construtor, sem resolução, distorção, procedência ou argumentos de calibração. Foram preservados apenas no perfil `legacy_video_UNVERIFIED.json`; sua validade física não foi estabelecida. |
| Documentação incompatível | **Confirmado no cabeçalho e comentários:** dizem ORB/fluxo óptico, mas a implementação usa SIFT + FLANN por descritores. O README raiz original não descrevia um algoritmo; não lhe atribuímos alegações inexistentes. |
| Vídeo estéreo | **Não confirmado:** um fluxo MPEG-4 1280×720, 1484 quadros, 30 fps, 49,467 s. Amostras 0/150/600/1200 foram inspecionadas visualmente: cena renderizada, sem evidência de um par estéreo codificado. Usado somente como vídeo monocular, sem referência nem precisão declarada. |

Correções: implementação nova sem GUI, máscaras mantidas, composição explícita, mapa e PnP monocular, triangulação estéreo métrica, calibração obrigatória, CSV com estado e segmento, dados reais e testes. Não há fallback implícito para passos unitários.
