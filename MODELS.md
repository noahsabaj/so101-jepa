# Model registry

Every trained model has a name and a version. Reports, PLAN.md and results use them.

## Names and versions

- **SO-JEPA X.Y** is a world model of this project (encoder, predictor and training losses).
  The planner is not part of the version; results give the planner separately.
- **X (major)** changes when the interface changes: the inputs (cameras, image size, joint
  angles), the action space, the latent shape or the number of levels. A planner, probe or goal
  set made for one major version may not work with another.
- **Y (minor)** changes for every other change of the recipe: architecture, losses, data,
  training length, a bug fix.
- A version is one fixed recipe. After its first training, its recipe does not change. A change
  gets a new version.
- **Checkpoint ID:** version plus seed, for example `sojepa-2.1-s42`.
- **File names:** the config is `hjepa/config/train/sojepa-X.Y.yaml` and its
  `output_model_name` is `sojepa-X.Y`. Versions 1.0 to 2.1 keep their old file names (below).
- **Code:** commit the repo and the H-JEPA fork before a training, and record both commits here.
  fleet pushes the working tree, so a dirty tree makes the record wrong.
- **Datasets** have IDs too (table at the end).
- Paper names (LeWM, LpWM, H-JEPA, EP-IDM) name recipes and losses, not our models.

## Models

| Version | Parent | Recipe (change from the parent) | Data | Training | Code (repo, fork) | Config, checkpoint | Status and results |
|---|---|---|---|---|---|---|---|
| SO-JEPA 1.0 | — | LeWM recipe. Latent 384 = image 256 (ViT-tiny, 8×8 patches, CLS token, MLP) + joint angles 128 (MLP). Predictor: causal transformer, 6 layers, width 384, AdaLN actions, history 4. Losses: prediction, SIGReg 0.08, one-step inverse dynamics 100. | sim-2 | 10 epochs, 27,750 steps, batch 128, lr 5e-4, seed 42 | f614b50 to a147a93 (resumed), 106986f | so101_lewm; so101_lewm | Rung 1 97/100. Rung 2 0/50. Image latent blind (A15 killed). |
| SO-JEPA 1.1 | 1.0 | LpWM recipe: RepReLU MLP heads on encoder and predictor; RDMReg (rectified Laplace, weight 10) in place of SIGReg. | sim-2 | 10 epochs, seed 42 | f614b50 to a147a93 (resumed), 106986f | so101_lpwm; so101_lpwm | Rung 1 100/100. Rung 2 0/50. Image latent blind. |
| SO-JEPA 2.0 | 1.0 | Image only: no joint angles in the input. Latent 384 from the image. | sim-2 | 3 epochs, seed 42 | 5b51c03, 229e1c2 | so101_lewm_vis; so101_lewm_vis | Training (variant A of the A15 fix). |
| SO-JEPA 2.1 | 2.0 | Endpoint inverse dynamics (EP-IDM): an MLP gets the 10 actions (2 s) back from z_t and z_t+10, weight 10. Clips of 11 frames. | sim-2 | 3 epochs, seed 42 | 492a961, f1577c4 | so101_lewm_vis_epidm; so101_lewm_vis_epidm | Training (variant E). The first tries (fork 7dcc7f4) crashed: clips were cut to 5 frames. |

Configs are in `hjepa/config/train/`. Checkpoints are in `data/ckpts/so101/<name>/seed42/` on the
training computer (kat-pc: WSL home).

Planned (the version is fixed at the first training):

- SO-JEPA 2.2: the 2.1 world model plus a learned goal-reaching value (hjepa/value.py; value.pt
  beside the 2.1 checkpoint). The planner cost is -V(last predicted latent, goal latent).
- SO-JEPA 3.0: 2.1 with token latents (config sojepa-3.0): 128 patch tokens of 64 numbers per
  frame, no CLS summary; predictor width 256, block-causal over the tokens.
- SO-JEPA 3.1: 3.0 plus its learned value.

Not trained yet, so no version: H-JEPA L2 (so101_hjepa_l2) and the community models
(community_lewm, community_hjepa_l2). Each gets a version at its first training.

## Datasets

| ID | Content | Files |
|---|---|---|
| sim-1 | First MuJoCo build (1,300 episodes). Faulty expert. Retired. | — |
| sim-2 | MuJoCo, fixed expert. 1,200 train episodes (seeds 1,000,000 and up), 100 val (2,000,000 and up), 300 steps at 5 Hz. 64×128 frames (scene and wrist), joints, actions, true state. | data/so101_train.h5, data/so101_val.h5 |
| community-1 | lerobot/community_dataset_v3 (revision ab92ac3f), 441 SO-100/SO-101 datasets, one camera, 64×64, 5 Hz. Test: 43 datasets not in training. | community/ (build.py) |
