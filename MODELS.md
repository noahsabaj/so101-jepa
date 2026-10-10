# Model registry

Every trained model has a name and a version. Reports, PLAN.md and results use them.

## Names and versions

The convention of ML model registries (MLflow, SageMaker, Vertex AI, W&B): one registered name and
integer versions in the order of registration, with the lineage in the registry.

- **SO-JEPA vN** is a world model of this project (encoder, predictor, training losses and any
  learned heads such as the value). N counts up by 1 for each new trained model: v1, v2, v3.
  The planner is not part of the version; results give the planner separately.
- N says nothing about quality or compatibility. The registry says what each version is: its
  parent, the change from the parent, data, training, code and results. Siblings (two changes
  from the same parent) are normal.
- Why not SemVer: its promise is compatibility within a major version. A learned model cannot
  keep it: any retraining changes the latent space, so value heads, probes, goal latents and
  upper H-JEPA levels made for one version never carry over to another.
- **One version is one trained artifact:** recipe, data, seed and code. It never changes. A new
  seed, a longer training or a bug fix is a new version (parent: the old one).
- **N is given when the training job is submitted.** A cancelled job's N is not used again.
- **File names:** the config is `hjepa/config/train/sojepa-vN.yaml`, its `output_model_name` is
  `sojepa-vN`, and results use `sojepa-vN` in their file names. v1 to v5 keep their old file names
  (below): they were trained or queued before this rule.
- **Code:** commit the repo and the H-JEPA fork before a training, and record both commits here.
  fleet pushes the working tree, so a dirty tree makes the record wrong.
- **Datasets** have IDs too (table at the end).
- Paper names (LeWM, LpWM, H-JEPA, EP-IDM, LeVJEPA) name recipes, losses and outside weights, not
  our models.

## Models

| Version | Parent | Recipe (change from the parent) | Data | Training | Code (repo, fork) | Config, checkpoint | Status and results |
|---|---|---|---|---|---|---|---|
| v1 | — | LeWM recipe. Latent 384 = image 256 (ViT-tiny, 8×8 patches, CLS token, MLP) + joint angles 128 (MLP). Predictor: causal transformer, 6 layers, width 384, AdaLN actions, history 4. Losses: prediction, SIGReg 0.08, one-step inverse dynamics 100. | sim-2 | 10 epochs, 27,750 steps, batch 128, lr 5e-4, seed 42 | f614b50 to a147a93 (resumed), 106986f | so101_lewm; so101_lewm | Rung 1 97/100. Rung 2 0/50. Image latent blind (A15 killed). |
| v2 | v1 | LpWM recipe: RepReLU MLP heads on encoder and predictor; RDMReg (rectified Laplace, weight 10) in place of SIGReg. | sim-2 | 10 epochs, seed 42 | f614b50 to a147a93 (resumed), 106986f | so101_lpwm; so101_lpwm | Rung 1 100/100. Rung 2 0/50. Image latent blind. |
| v3 | v1 | Image only: no joint angles in the input. Latent 384 from the image. | sim-2 | 3 epochs, seed 42 | 5b51c03, 229e1c2 | so101_lewm_vis; so101_lewm_vis | Ridge probe: table 7.12, grasp 1.66 cm. |
| v4 | v3 | Endpoint inverse dynamics (EP-IDM): an MLP gets the 10 actions (2 s) back from z_t and z_t+10, weight 10. Clips of 11 frames. | sim-2 | 3 epochs, seed 42 | 492a961, f1577c4 | so101_lewm_vis_epidm; so101_lewm_vis_epidm | Ridge probe: table 6.53, grasp 1.72 cm. Rung 1 (CEM) 96/100, median 0.22 cm. Rung 2 0/50, cube never lifted. The first tries (fork 7dcc7f4) crashed: clips were cut to 5 frames. |
| v5 | v4 | v4's world model plus a learned goal-reaching value (hjepa/value.py: hindsight goals, -1 per step, expectile regression, two heads). Planner cost -V(last predicted latent, goal latent). | sim-2 | value: 50,000 steps, batch 1,024, on frozen v4 latents | 7b01eff, 70063d2 | so101_flat_value / so101_flat_cem_value; value.pt beside v4's checkpoint | Rung 1 18/100 (median 2.0 cm), rung 2 0/50: the value is a worse cost than v4's latent distance. |
| v6 | v4 | Token latents: every ViT patch token (8×16 = 128 tokens of 64 numbers), no CLS summary. Predictor width 256, block-causal over the tokens. | sim-2 | 3 epochs, seed 42 | 1181b0c, 70063d2 | sojepa-v6; sojepa-v6 | Ridge probe: table 5.20, grasp 2.34 cm; attentive table 4.06 (rerun 3.97). Rung 1 (CEM) 98/100, median 0.19 cm. Rung 2 0/50, never lifted. Jump test (time error): 1.53 s at 2 s, 4.05 s at 4 s. |
| v7 | v6 | v6's world model plus its learned value (as v5). | sim-2 | as v5 | 1181b0c, 70063d2 | value.pt beside v6's checkpoint | Rung 1 3/100 (median 3.5 cm); rung 2 0/33 (stopped), never lifted. |
| v8 | v3 | Seed 43 (v3's seed-variance replicate). | sim-2 | as v3, seed 43 | (this commit) | sojepa-v8; sojepa-v8 | Ridge probe: table 7.45, grasp 1.80 cm. Rung 1 92/100; rung 2 0/50. |
| v9 | v4 | Seed 43 (v4's seed-variance replicate). | sim-2 | as v4, seed 43 | (this commit) | sojepa-v9; sojepa-v9 | Ridge probe: table 6.44, grasp 1.70 cm. |
| v10 | v6 | Seed 43 (v6's seed-variance replicate). | sim-2 | as v6, seed 43 | (this commit) | sojepa-v10; sojepa-v10 | Ridge probe: table 5.67, grasp 2.43 cm; attentive table 3.81 (seed spread with v6: 0.25 cm). |
| v11 | v6 | 10 epochs instead of 3 (A18: compute). | sim-2 | 10 epochs, seed 42 | gpu-portable | sojepa-v11 | Ridge table 5.05, grasp 1.96; attentive table 5.79 cm. No gain from 3x compute. |
| v12 | v6 | ViT-small encoder instead of ViT-tiny (A18: model size). | sim-2 | 3 epochs, seed 42 | gpu-portable | sojepa-v12 | Ridge table 4.87, grasp 1.89; attentive table 4.17 cm. No gain from the bigger encoder alone. |
| v13-v20 | v6 | Loss-weight search, 8 trials (A23): SIGReg, inverse-dynamics and EP-IDM weights drawn log-uniform (numpy seed 0; values in each config). | sim-2 | 3 epochs, seed 42 | gpu-portable | sojepa-v13 ... sojepa-v20 | Attentive table 3.87 (v19) to 4.84 cm: no weight set beats v6 by more than the noise. |
| v21 | v6 | Play data only: no expert, no task (A19a). | sim-3 (play) | 3 epochs, seed 42 | gpu-portable | sojepa-v21 | Attentive table 7.34, ridge 6.37 cm on the expert val set (play data has few grasps). |
| v22 | v6 | Prodigy optimizer: no learning rate (A23). | sim-2 | 3 epochs, seed 42 | gpu-portable | sojepa-v22 | Ridge table 5.54; attentive table 3.90 cm (rerun 3.80): equal to the hand-set learning rate. |
| v23-v28 | v6 | Optimizer search, 6 trials (A23): learning rate, weight decay, batch size drawn log-uniform (numpy seed 1). | sim-2 | 3 epochs, seed 42 | gpu-portable | sojepa-v23 ... sojepa-v28 | Attentive table 4.12 (v25) to 6.70 cm (v24): no setting beats v6's. |
| v29 | v6 | Prodigy + ScheduleFree: no learning rate, no schedule (A23). | sim-2 | 3 epochs, seed 42 | gpu-portable | sojepa-v29 | Ridge table 6.62; attentive table 4.04 cm: equal to v6 without a learning rate or a schedule. The weaker ridge is not from BatchNorm statistics (recomputed: no change). |
| v30 | v6 | Trained on sim-2 + community-1 (real SO-100/SO-101 teleoperation), same steps as v6 (A24). | sim-2 + community-1 | 8,154 steps, seed 42 | gpu-portable | sojepa-v30 | Ridge table 6.40; attentive table 5.68 cm on sim val: half the steps go to community data. Transfer to the real arm not tested yet. |
| v31 | v6 | Time-step-conditioned predictor: random stride 1-10 steps per clip, stride as input (A21). | sim-2 | 8,154 steps, seed 42 | gpu-portable | sojepa-v31 | Ridge table 4.14, grasp 2.09; attentive table 3.56 cm (rerun 3.35). Jump test: 0.79 s at 2 s, 2.16 s at 4 s (v6: 1.53, 4.05). Reach with stride planning 12/12 (stopped: 417 s per trial). |
| v32 | v21 | Self-improvement round 1: play data + 1,200 episodes of v21 playing toward goals sampled from the play data (A19b). | sim-3 + self-play | 8,154 steps, seed 42 | gpu-portable | sojepa-v32 | Cancelled: round 1 self-play gave 0 episodes (93 processes shared one GPU with a training). N not used again. |
| v33 | v6 | 25% of sim-2, same steps (A18: data). | sim-2 (300 episodes) | 8,154 steps, seed 42 | gpu-portable | sojepa-v33 | Attentive table 6.62 cm. |
| v34 | v6 | 50% of sim-2, same steps (A18: data). | sim-2 (600 episodes) | 8,154 steps, seed 42 | gpu-portable | sojepa-v34 | Attentive table 5.23 cm. With v6 (4.06): -1.3 cm per doubling of the data. |
| v35 | v31 | Best parts together: v31's time-step predictor, ViT-small (v12), Prodigy + ScheduleFree (v29), 3x the steps. | sim-2 | 24,462 steps, seed 42 | a19f4e4, f1eee9a | sojepa-v35; sojepa-v35 | Ridge table 3.99, grasp 2.04; attentive table 3.32 cm. Jump test 0.96 s at 2 s, 2.13 s at 4 s. Equal to v31 and v36: the extras add nothing. |
| v36 | v31 | Seed 43 (v31's replicate). | sim-2 | as v31, seed 43 | a19f4e4, f1eee9a | sojepa-v36; sojepa-v36 | Ridge table 4.16, grasp 2.05; attentive table 3.22 cm. Jump test 0.68 s at 2 s, 1.76 s at 4 s. Confirms v31. |
| v37 | v29 | Seed 43 (v29's replicate). | sim-2 | as v29, seed 43 | a19f4e4, f1eee9a | sojepa-v37; sojepa-v37 | Ridge table 6.90, grasp 2.44; attentive table 4.89 cm (v29: 4.04): schedule-free spreads ~0.9 cm between seeds. |
| v38 | v6 | ScheduleFree+ (h_jepa/sfplus.py, a port of Meta's reference, Apache-2.0): Polyak step with f* = 0, schedule-free averaging, AdamC weight decay. No learning rate, no schedule. | sim-2 | 3 epochs, seed 42 | a19f4e4, f1eee9a | sojepa-v38; sojepa-v38 | Diverged (NaN): gradient clipping ran before the Polyak step, so steps were 30-90x too large. |
| v39 | v38 | f* fitted during training instead of 0. | sim-2 | 3 epochs, seed 42 | a19f4e4, f1eee9a | sojepa-v39; sojepa-v39 | Diverged, but it inherited gradient clipping (only v40 turns it off), so it does not test the fit. Not needed: f* = 0 trains without clipping (v40). The fit code is removed. |
| v40 | v38 | No gradient clipping. | sim-2 | 3 epochs, seed 42 | a19f4e4, f1eee9a | sojepa-v40; sojepa-v40 | Ridge table 6.58, grasp 2.51; attentive table 4.57 cm (rerun 4.69): trains, 0.5 cm behind v6 and v22. |
| v41 | v40 | Seed 43 (v40's replicate). | sim-2 | as v40, seed 43 | 23998e4, 3645eb9 | sojepa-v41; sojepa-v41 | Ridge table 7.60, grasp 2.89; attentive table 3.64 cm (v40: 4.57). |
| v42 | v31 | The new default recipe: v31 (time-step predictor) with Prodigy (no learning rate, v22), SIGReg in fp32 and stride-1 clips from every start. Also Prodigy's seed replicate. | sim-2 | 8,154 steps, seed 42 | 80d6242+, 4a486a6 | sojepa-v42; sojepa-v42 | Training (National Compute, 2026-10-09) |
| v43 | v21 | Self-improvement round 1 (A19b): play data + the episodes of v21 playing toward goals from the play data (round 1c, kat-pc; rounds 1 and 1b were lost). | sim-3 + self-play | 8,154 steps, seed 42 | 80d6242+, 4a486a6 | sojepa-v43; sojepa-v43 | Self-play round 1c: 132 episodes of v21 playing (6 h, fast planner) added to the 1200 play episodes. Probe: ridge table 6.30 cm (v21 6.37), grasp 2.33; attentive table 8.04 (v21 7.34): no gain. |
| v44 | v4 | 2-level H-JEPA on v4's vision-only recipe (EP-IDM); level 2 of so101_hjepa_l2 (stride 5, history 3). The first H-JEPA config was built on v1's blind recipe and never trained. | sim-2 | 3 epochs, seed 42 | (this commit) | sojepa-v44; sojepa-v44 | The first model over the rung 2 pass mark: pick 2/50 (cube within 2 cm of the goal place, resting; median 13.2 cm), both by pushing: never picked up, success_strict 0/50. Rung 1 93/100, median 0.24 cm (v6: 98, 0.19). Probe: grasp 1.72 cm, table 6.26 cm. Offline tests: outputs/offline_sojepa-v44_so101_l2.json. |
| v45 | v4 | v4's recipe on the community data (real SO-100/SO-101, one 64x64 view; A5, A24). | community-1 | 5 epochs, seed 42 | (this commit) | sojepa-v45; sojepa-v45 | Training (National Compute, 2026-10-09) |
| v46 | v44 | v44 (2-level, vision only) on the community data. | community-1 | 5 epochs, seed 42 | (this commit) | sojepa-v46; sojepa-v46 | Training (National Compute, 2026-10-09) |
| v47 | (new) | LeWAM (A25): ResNet-18 encoder per view (64 px upsampled to 96), MoT predictor with a state stream and a goal-conditioned flow-matching action stream, SIGReg; LeWAM's gr protocol (frameskip 5, history 2, H_max 20), both views as goal, vision only. Trained with LeWAM's own code (third_party/lewam). | sim-2 | 20 epochs (49,780 batches of 128), seed 42; epoch 1 fp32, then bf16, fused AdamW, compiled trunk and loss (~75 s an epoch on an MI355X) | (this commit) | lewam/sojepa-v47 | Fails. Head alone: reach 0/100 (median 19.5 cm), pick 0/42 (13.0 cm); gradient planner reach 0/35 (16.5 cm). Probe: grasp 10.7 cm, cube 12.1 cm (joints alone: 5.9, 8.0). Offline action MSE 1.18, no better than the mean action; 1.42 with the goal 1 chunk ahead: from scratch the latents do not see the arm. |
| v47b | v47 | v47 warm-started from the authors' OGBench Cube checkpoint (huggingface.co/LeWAM/lewam-cube): encoder and predictor loaded, action input/output and view embedding fresh. | sim-2 | as v47 (20 epochs), seed 42 | 152e63f | lewam/sojepa-v47b | The first LeWAM head that works: reach head alone 1/100 within the bar, median 4.8 cm; gradient planner 6/83, 2.8 cm. Pick 0/50 (13.8 cm; picked up 13/50), gradient 0/50 (13.2 cm; picked up 9). Probe: grasp 5.4 cm, cube 8.0 cm (v47: 10.7, 12.1). Offline action MSE 0.58 with the goal 1 chunk ahead, 0.86 at 11-20 (v47: 1.42, 1.16). The warm start does in 20 epochs what training from scratch needs ~50 for (v49). |
| v48 | v47 | v47 at 224 px (sim-2 re-rendered, the same episodes), LeWAM's encoder exactly as published; bf16. | sim-2 at 224 px | 20 epochs, 1 of warmup, seed 42, bf16; ~4 min an epoch | 05cd7d8 | lewam/sojepa-v48 | Stopped after 9 epochs: val act stuck at 1.10-1.12 from epoch 2 (v48b, warm: 0.80 at epoch 8). From scratch, LeWAM's own 224 px encoder does not learn our actions in 20 epochs; GPU given to v51. |
| v48b | v48 | v48 warm-started from the Cube checkpoint (also 224 px): the closest match to the paper. | sim-2 at 224 px | as v48 (20 epochs), seed 42 | 05cd7d8 | lewam/sojepa-v48b | No better than v47b at 3.3x the training time: reach head alone 0/100, median 5.1 cm (v47b 4.8); gradient planner 3/100, 2.5 cm (v47b 8/100, 2.8); pick head alone 0/50 (12.6 cm), gradient planner 0/18 (12.1 cm). Probe: grasp 5.5 cm, cube 9.6 cm (v47b 5.4, 8.0). Offline action MSE 0.82 (v47b 0.75): best with the goal 1 chunk ahead (0.40), worst at 11-20 (1.01). Real 224 px views do not help LeWAM here; stay at 64 px. |
| v49 | v47 | v47 for the paper's 50 epochs (5 of warmup): does LeWAM gain from training as long as in the paper? | sim-2 | 50 epochs, seed 42 | 8114f41 | lewam/sojepa-v49 | Works: best val 0.722; reach head alone 2/100, median 6.4 cm; gradient planner 8/100, 2.9 cm (as v47b: 8/100, 2.8); pick head alone 0/50 (picked up 18/50, the most of any run), gradient planner 0/42 (12.6 cm). Probe: grasp 5.7 cm, cube 6.0 cm (the best LeWAM cube so far). Offline action MSE 0.73 (0.54 with the goal 1 chunk ahead). v47 failed for lack of training, not of the warm start: 50 epochs from scratch come close to v47b. |
| v50 | v47 | v47 with the 64 px views upsampled to 224 (LEWAM_UPSAMPLE=224): v48's 7 x 7 map without its 224 px detail. | sim-2 | 20 epochs, 1 of warmup, seed 42, bf16 | a9bad6c | lewam/sojepa-v50 | Stopped after 4 epochs: val act stuck at 1.12, as v48 from scratch (1.11 at epoch 7); GPU given to v50b. From scratch, the 7 x 7 map trains worse than the 3 x 3 (v47: 0.79). |
| v50b | v50 | v50 warm-started from the Cube checkpoint: does warm start need real 224 px (v48b), or only the checkpoint's map size? | sim-2 | 15 epochs, 1 of warmup, seed 42, bf16 | (this commit) | lewam/sojepa-v50b | About v47b or worse: reach head alone 1/100, median 5.4 cm; gradient planner 0/25, 3.7 cm; pick head alone 0/28 (16.3 cm). Offline action MSE 0.779 (v47b 0.754): 0.57 with the goal 1 chunk ahead, 0.91 at 11-20. Probe: grasp 5.1 cm, cube 7.6 cm (v47b 5.4, 8.0). Best val 0.774 at 15 epochs. The 7 x 7 map gives nothing over 3 x 3 with the warm start either. |
| v51 | v47b | v47b for 35 epochs: does more training help the warm-started head in closed loop? | sim-2 | 35 epochs, 2 of warmup, seed 42, bf16 | (this commit) | lewam/sojepa-v51 | Reach head alone 2/100, median 4.7 cm (v47b 4.8); gradient planner 1/22, 2.7 cm (stopped for v50b's GPU); pick head alone 0/50 (12.0 cm). Offline action MSE 0.710, the best LeWAM so far (v47b 0.754): 0.53 with the goal 1 chunk ahead, 0.84 at 11-20. More epochs help offline, not in closed loop. |
| v47c | v47b | v47b with seed 43 (its replicate): is the reach gap between v47b (4.8 cm) and v49 (6.7 cm) larger than the seed noise? | sim-2 | as v47b (20 epochs), seed 43 | (this commit) | lewam/sojepa-v47c | Seed noise is large in closed loop: reach head alone median 6.3 cm (1/100; v47b 4.8 cm), gradient planner 3.9 cm (2/58; v47b 2.8 cm), pick head alone 1/48 loose, 0 strict (12.3 cm); small offline: action MSE 0.765 (v47b 0.754). The v47b-v49 reach gap (4.8 vs 6.7 cm) is within it. |
| v53 | v47b | Data scaling: v47b's recipe at a fixed budget (8 epochs of the full train set) on 1/8, 1/4, 1/2 and all of the train episodes (repeated to the same steps; sojepa-v53-f12 ... -f100); val on a held-out 10% of the episodes for all, then the offline action check on the val set. | sim-2 (fractions) | 8 epochs, 1 of warmup, seed 42, bf16; all 4 on one GPU | (this commit) | lewam/sojepa-v53-f* | Held-out val act, 4 epochs: 1/8 0.963 (train 0.816: overfits), 1/2 0.890, all 0.882. More episodes help most up to half the data at this budget. Offline checks: outputs/v53-f*/check.log. |
| v54 | v53 | v53 at all episodes with batch 512, lr 3e-4 (sqrt scaling): 24% less time an epoch (lewam_bench). Same quality as v53-f100 (0.882)? | sim-2 | 4 epochs, seed 42, bf16 | (this commit) | lewam/sojepa-v54 | Best held-out val 0.897 (v53-f100: 0.882) in 4 epochs of 67 s (v53-f100 alone: ~85 s): a bit worse at equal epochs. SIGReg's term grows with the batch (reg 3.6 at 128, 6.1 at 256, 8.8 at 512): v54d tests that. |
| v54b | v53 | v54 with batch 256, lr 2.12e-4: 11% less time an epoch. | sim-2 | 4 epochs, seed 42, bf16 | (this commit) | lewam/sojepa-v54b | Best held-out val 0.893 (v53-f100: 0.882). |
| v54c | v54 | v54 for 5 epochs: about v53-f100's wall-clock time. | sim-2 | 5 epochs, batch 512, seed 42 | (this commit) | lewam/sojepa-v54c | Best held-out val 0.884 (v53-f100 at batch 128: 0.882) in about the same wall-clock time: batch 512 gives no real gain. |
| v54d | v54 | v54 with the SIGReg weight / 4 (0.01). | sim-2 | 4 epochs, batch 512, seed 42 | (this commit) | lewam/sojepa-v54d | Best held-out val 0.891 (v54: 0.897): the stronger SIGReg of a larger batch explains part of v54's gap. |

LeWAM runs (lewam/...): v47 to v51 and v47c used LeWAM's own train/val split, by start point: val frames come from training episodes, so their val loss and best_val reward recall, and lewam_best.pt was chosen by it (in these runs val fell almost every epoch, so it is close to the last epoch). Compare them by the offline action check (hjepa/lewam_check.py, on the separate val episodes) and the closed loop. From v53 the split is by episode (hjepa/lewam_common.py; LEWAM_SPLIT=frame reproduces the old runs).

Configs are in `hjepa/config/train/`. Checkpoints are in `data/ckpts/so101/<name>/seed<seed>/` on the
training computer (kat-pc: WSL home).

Hardware and torch (GPU.md): v1 to v3 train on kat-pc (RTX 4060 Ti, torch 2.7.1+cu128). v4 to v10
train on a National Compute node (8x AMD MI355X, torch 2.11.0+rocm7.2), one GPU per model.

H-JEPA L2 and the community models are v44 to v46 (vision only); the earlier configs on v1's
blind recipe (so101_hjepa_l2 alone, community_lewm, community_hjepa_l2) were never trained.

## Datasets

| ID | Content | Files |
|---|---|---|
| sim-1 | First MuJoCo build (1,300 episodes). Faulty expert. Retired. | — |
| sim-2 | MuJoCo, fixed expert. 1,200 train episodes (seeds 1,000,000 and up), 100 val (2,000,000 and up), 300 steps at 5 Hz. 64×128 frames (scene and wrist), joints, actions, true state. | data/so101_train.h5, data/so101_val.h5 |
| sim-3 | Play data: the same scenes as sim-2's train seeds, but motor babbling (sim/collect.py play: random joint targets held 1-15 steps), no expert, no task. 1,200 episodes. | data/so101_train_play.h5 |
| community-1 | lerobot/community_dataset_v3 (revision ab92ac3f), 441 SO-100/SO-101 datasets, one camera, 64×64, 5 Hz. Test: 43 datasets not in training. | community/ (build.py) |
