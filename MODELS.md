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
| v3 | v1 | Image only: no joint angles in the input. Latent 384 from the image. | sim-2 | 3 epochs, seed 42 | 5b51c03, 229e1c2 | so101_lewm_vis; so101_lewm_vis | Training (A15 fix: vision only). |
| v4 | v3 | Endpoint inverse dynamics (EP-IDM): an MLP gets the 10 actions (2 s) back from z_t and z_t+10, weight 10. Clips of 11 frames. | sim-2 | 3 epochs, seed 42 | 492a961, f1577c4 | so101_lewm_vis_epidm; so101_lewm_vis_epidm | Training (A15 fix: vision only plus EP-IDM). The first tries (fork 7dcc7f4) crashed: clips were cut to 5 frames. |
| v5 | v4 | v4's world model plus a learned goal-reaching value (hjepa/value.py: hindsight goals, -1 per step, expectile regression, two heads). Planner cost -V(last predicted latent, goal latent). | sim-2 | value: 50,000 steps, batch 1,024, on frozen v4 latents | 7b01eff, 70063d2 | so101_flat_value / so101_flat_cem_value; value.pt beside v4's checkpoint | Queued |
| v6 | v4 | Token latents: every ViT patch token (8×16 = 128 tokens of 64 numbers), no CLS summary. Predictor width 256, block-causal over the tokens. | sim-2 | 3 epochs, seed 42 | 1181b0c, 70063d2 | sojepa-v6; sojepa-v6 | Queued |
| v7 | v6 | v6's world model plus its learned value (as v5). | sim-2 | as v5 | 1181b0c, 70063d2 | value.pt beside v6's checkpoint | Queued |
| v8 | v3 | Seed 43 (v3's seed-variance replicate). | sim-2 | as v3, seed 43 | (this commit) | sojepa-v8; sojepa-v8 | Queued |
| v9 | v4 | Seed 43 (v4's seed-variance replicate). | sim-2 | as v4, seed 43 | (this commit) | sojepa-v9; sojepa-v9 | Queued |
| v10 | v6 | Seed 43 (v6's seed-variance replicate). | sim-2 | as v6, seed 43 | (this commit) | sojepa-v10; sojepa-v10 | Queued |
| v11 | v6 | 10 epochs instead of 3 (A18: compute). | sim-2 | 10 epochs, seed 42 | gpu-portable | sojepa-v11 | Training |
| v12 | v6 | ViT-small encoder instead of ViT-tiny (A18: model size). | sim-2 | 3 epochs, seed 42 | gpu-portable | sojepa-v12 | Training |
| v13-v20 | v6 | Loss-weight search, 8 trials (A23): SIGReg, inverse-dynamics and EP-IDM weights drawn log-uniform (numpy seed 0; values in each config). | sim-2 | 3 epochs, seed 42 | gpu-portable | sojepa-v13 ... sojepa-v20 | Training |
| v21 | v6 | Play data only: no expert, no task (A19a). | sim-3 (play) | 3 epochs, seed 42 | gpu-portable | sojepa-v21 | Queued (data) |
| v22 | v6 | Prodigy optimizer: no learning rate (A23). | sim-2 | 3 epochs, seed 42 | gpu-portable | sojepa-v22 | Training |
| v23-v28 | v6 | Optimizer search, 6 trials (A23): learning rate, weight decay, batch size drawn log-uniform (numpy seed 1). | sim-2 | 3 epochs, seed 42 | gpu-portable | sojepa-v23 ... sojepa-v28 | Training |
| v29 | v6 | Prodigy + ScheduleFree: no learning rate, no schedule (A23). | sim-2 | 3 epochs, seed 42 | gpu-portable | sojepa-v29 | Queued |
| v30 | v6 | Trained on sim-2 + community-1 (real SO-100/SO-101 teleoperation), same steps as v6 (A24). | sim-2 + community-1 | 8,154 steps, seed 42 | gpu-portable | sojepa-v30 | Training |
| v31 | v6 | Time-step-conditioned predictor: random stride 1-10 steps per clip, stride as input (A21). | sim-2 | 8,154 steps, seed 42 | gpu-portable | sojepa-v31 | Training |
| v32 | v21 | Self-improvement round 1: play data + 1,200 episodes of v21 playing toward goals sampled from the play data (A19b). | sim-3 + self-play | 8,154 steps, seed 42 | gpu-portable | sojepa-v32 | Queued |
| v33 | v6 | 25% of sim-2, same steps (A18: data). | sim-2 (300 episodes) | 8,154 steps, seed 42 | gpu-portable | sojepa-v33 | Training |
| v34 | v6 | 50% of sim-2, same steps (A18: data). | sim-2 (600 episodes) | 8,154 steps, seed 42 | gpu-portable | sojepa-v34 | Training |

Configs are in `hjepa/config/train/`. Checkpoints are in `data/ckpts/so101/<name>/seed<seed>/` on the
training computer (kat-pc: WSL home).

Hardware and torch (GPU.md): v1 to v3 train on kat-pc (RTX 4060 Ti, torch 2.7.1+cu128). v4 to v10
train on a National Compute node (8x AMD MI355X, torch 2.11.0+rocm7.2), one GPU per model.

Not trained yet, so no version: H-JEPA L2 (so101_hjepa_l2) and the community models
(community_lewm, community_hjepa_l2). Each gets a version when its training job is submitted.

## Datasets

| ID | Content | Files |
|---|---|---|
| sim-1 | First MuJoCo build (1,300 episodes). Faulty expert. Retired. | — |
| sim-2 | MuJoCo, fixed expert. 1,200 train episodes (seeds 1,000,000 and up), 100 val (2,000,000 and up), 300 steps at 5 Hz. 64×128 frames (scene and wrist), joints, actions, true state. | data/so101_train.h5, data/so101_val.h5 |
| sim-3 | Play data: the same scenes as sim-2's train seeds, but motor babbling (sim/collect.py play: random joint targets held 1-15 steps), no expert, no task. 1,200 episodes. | data/so101_train_play.h5 |
| community-1 | lerobot/community_dataset_v3 (revision ab92ac3f), 441 SO-100/SO-101 datasets, one camera, 64×64, 5 Hz. Test: 43 datasets not in training. | community/ (build.py) |
