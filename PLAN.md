# so101-jepa plan

Written in ASD-STE100 (Simplified Technical English).

## 1. North star

An SO-101 arm learns to assemble another SO-101 arm.

- Along the way: the robot first assembles a robot-friendly SO-101 (chamfers, snap-fits, fixtures).
  The stock SO-101 (74 screws, 6 cables) is the final test (decision of 2026-10-06).
- The system is JEPA. A JEPA world model and a planner control the arm. A small actor can propose
  start plans, but the world model and the cost select the action (decision of 2026-10-06). No VLA.

## 2. Rules: how we leave local maxima

1. One ladder metric: the highest rung (section 6) that passes on the test trials. We do not
   polish a rung that passes. (Skunkworks: reach went from 81% to 97%; pick-and-place stayed at 0%.)
2. Each assumption (section 5) has a test that can kill it. Each week, we test at least one.
3. We run 2 or 3 alternatives at small scale in parallel on the fleet, not one at a time.
4. Stop rule: if a rung stays stuck for 3 tries, we stop and do a first-principles review. We
   record it in section 8. You challenge me, and I challenge you.
5. We set each pass mark before the test. If we change a pass mark after a test, we write why,
   and we do the test again on new trials.
6. We select settings on the tuning trials (seeds 1000 to 4999). We report results on the test
   trials (seeds 5000 and higher). We compare two settings on the same trials (paired), with a 95%
   interval.
7. Sim first. Our software does not move the real arm before the sim test of that rung passes.
8. The safety checks use only kinematics and fixed limits, not the world model. A person stays at
   the power switch when the real arm can move.
9. Lean code. When an experiment concludes, we record the result in section 9 and delete its code.
10. The fleet first. I tell you the cost before I rent a GPU.

## 3. Task facts

- One follower arm: 6 STS3215 servos (1/345 gears), 11 horns, 74 screws (50 M3x6, 24 M2x6) and
  6 three-pin cables ([LeRobot SO-101 guide](https://huggingface.co/docs/lerobot/so101)).
- Servo backlash: approximately 0.9 degrees (measured; the datasheet gives 0.5 or less). This is
  several mm at the gripper. Screws need sub-mm alignment.
- Each servo reports position, speed, load (register 60) and current (register 69). The current
  is a touch sensor at no cost.
- The SO-101 URDF and MJCF models are official (SO-ARM100, `Simulation/SO101`).
- Real data: approximately 23k community SO-100 episodes (the 481 datasets of SmolVLA). Your
  leader-arm demos after the build.
- Printer: AnkerMake M5, 235 x 235 mm bed, PLA+. Print the gauge files first.

## 4. Architecture

```mermaid
flowchart TB
  G["Goal: CAD state + step order"]
  subgraph L3["Level 3: step, 10 s"]
    E3["Encoder L3"] --> P3["Predictor L3"] <--> Q3["Planner L3"]
  end
  subgraph L2["Level 2: skill, 1 s"]
    E2["Encoder L2"] --> P2["Predictor L2"] <--> Q2["Planner L2"]
  end
  subgraph L1["Level 1: contact, 0.1 s"]
    E1["Encoder L1: ViT-S on all sensors"] --> P1["Predictor L1"] <--> Q1["Planner L1"]
  end
  S["Sensors: 2 cameras, joints, servo current"] --> E1
  E1 --> E2 --> E3
  G --> Q3 --> Q2 --> Q1 --> M["SO-101 servos: targets + stiffness"]
  M --> W["World: sim or real"] --> S
```

| Part | What | Why (task fact) |
|---|---|---|
| Sensors | Wrist camera, scene camera, 6 joint positions, 6 servo currents | The wrist view makes small parts large. The current shows contact. |
| Encoder L1 | Trainable ViT-S on all sensors, output: a few latent tokens | Many parts at the same time. Frozen tokens did not show a 2.5 cm cube (skunkworks). |
| Predictor L1 | Action-conditioned transformer, 0.1 s steps | Contact dynamics. |
| Actions L1 | Joint moves and servo stiffness (P gain or torque limit), in chunks | No inverse kinematics. Compliance for insertion. |
| Levels 2 and 3 | Encoders pool the latents of the level below; learned macro-actions (H-JEPA) | About 100 operations: skills (1 s) and assembly steps (10 s). |
| Goal | CAD state of each step, rendered with randomization; order from the official guide, checked by disassembly in sim | We do not learn what we already know. |
| Training | All levels end-to-end: prediction + SIGReg + inverse dynamics; an ensemble of predictors | Contacts jam or slip: the ensemble gives uncertainty. |
| Planner | Top-down (H-JEPA). The actor proposes candidates, with random ones; multi-start gradient descent (or Gauss-Newton) refines them; the lowest cost plus an uncertainty penalty wins; receding horizon | Fast plans on a small GPU. |
| Safety | An independent client checks each move with its own arm model and fixed limits | The world model can be wrong. |

Learning loop: randomized sim, community data, your demos and your assembly videos (no actions)
→ train → practice in sim (tasks with the most ensemble disagreement; hindsight goals; the robot
disassembles its work to reset the task) → the real arm, after the build. Every new episode goes
back into the data.

## 5. Assumption register

| # | Assumption | Test that can kill it | Status |
|---|---|---|---|
| A1 | A hierarchy helps assembly. | Flat LeWM against 2- and 3-level H-JEPA, same sim trials, rungs 2 to 6. | Open |
| A2 | A trainable encoder is better for control than a frozen one. | Frozen V-JEPA 2.1 tokens against a trainable ViT-S, rung 2. | Open |
| A3 | A few latent tokens are better than one vector when many parts are present. | One CLS vector against K tokens, rungs 2 and 6. | Open |
| A4 | Servo current adds contact information. | With and without current, rung 3. | Open |
| A5 | Models trained in sim transfer to the real SO-101. | Offline on community real data; then real rungs 1 and 2. | Open |
| A6 | CAD renders are good goals for real images. | Latent distance between a render and a photo of the same state, against a 1 cm change. | Open |
| A7 | Multi-start gradient descent plans well on our models. | Gradient descent, CEM and Gauss-Newton: same model, same trials. | Open |
| A8 | The actor makes plans faster with no loss of success. | Planner with and without actor proposals, same compute. | Open |
| A9 | Ensemble disagreement selects useful practice. | Against uniform tasks, same compute (paper 2610.02159). | Open |
| A10 | Stiffness in the action helps insertion. | With and without stiffness control, rung 3. | Open |
| A11 | Closed-loop vision and chamfers give sub-mm alignment with this arm. | Rung 3 success against chamfer size, in sim, then real. | Open |
| A12 | One arm and printed fixtures are sufficient. | Rungs 4 to 6 with fixtures. If they fail for lack of a second hand, we build a second follower. | Open |
| A13 | Assembly videos with no actions improve the upper levels. | L3 prediction of step results, with and without video pretraining. | Open |
| A14 | Dense Gaussian latents (SIGReg) are a good choice for planning; sparse latents (LpWM: RDMReg to a rectified Laplace, RepReLU heads) are not better. | LpWM against LeWM: same data, encoder, predictor and trials; each with gradient descent and CEM; offline tests and rungs 1 and 2 (rule 6, paired). If LpWM wins, a dense control with the same heads (Identity link, Gaussian target) tells if sparsity or the heads give the gain. | Open (set up 2026-10-06) |
| A15 | The world-model loss makes the image latent keep what the actions change (the arm, and the cube when the arm moves it). | Linear probe of the image latent on held-out episodes (hjepa/probe.py). Pass mark (set 2026-10-07, before the variants): grasp point and resting cube both within 2 cm RMS. | Killed for LeWM and LpWM with joint angles in the latent (12 to 13 cm; chance 12.8). In test: variant A (vision only) and E (vision only plus endpoint inverse dynamics, arXiv 2610.07540). |

## 6. The ladder

Each rung passes in sim first, then on the real arm. The pass marks are proposals. We fix each one
before its first test.

| Rung | Task | Sim pass mark (proposal) |
|---|---|---|
| 1 | Reach to a goal image | 90 of 100 test trials within 1 cm (fixed 2026-10-06, see below) |
| 2 | Pick and place with only the final goal image | 30 of 50 (fixed 2026-10-06, see below) |
| 3 | Put a servo into its housing | 30 of 50 fully in (within 1 mm of the CAD pose) |
| 4 | Press a horn onto a servo | 30 of 50 |
| 5 | One fastener (snap-fit first, then a screw with a tool) | 30 of 50 |
| 6 | One joint sub-assembly | 10 of 20 |
| 7 | The full robot-friendly arm | 5 of 10 |
| 8 | Rungs 1 to 7 on the real arm | Set before each test |
| 9 | The stock SO-101 | Set before the test |

Sim tests of rungs 1 and 2 (fixed 2026-10-06, before the first test; sim/closed_loop.py):
- Rung 1: the goal is an observation (scene and wrist views, joint angles) of the arm at a random
  reachable pose, gripper pointing down, 4 to 20 cm above the table. Success: the grasp point ends
  within 1 cm of the goal's grasp point after 100 steps (20 s). Test trials: seeds 5000 to 5099.
- Rung 2: the goal is an observation of the cube at a new place (6 cm or more away) and the arm at
  its rest pose. No sub-goals. Success: after 300 steps (60 s) the cube is within 2 cm (xy) of the
  goal place and rests on the table. Test trials: seeds 5000 to 5049.
- We tune on seeds 1000 to 4999. We compare flat LeWM and 2-level H-JEPA on the same test trials.
- Change (2026-10-06, before any test): cubes and goal places are 15 to 25 cm from the arm's base
  (was 15 to 28 cm), within +-60 degrees. Past 26 cm the arm is almost straight, and the scripted
  expert, which knows the true state, lifted the cube in only 43 of 99 tries (sim/expert_eval.py).

## 7. Phases

### Phase 0: sim only, while the parts print (2026-10-06 to 2026-10-20)
1. Simulator test. MuJoCo, ManiSkill and Isaac Lab each put an STS3215 into its SO-101 housing
   (clearance from the CAD), 100 random starts, scripted insertion. Measure: stable contact,
   physics steps per second, render speed, and setup work on the fleet. Select one.
2. SO-101 scene: official MJCF, table, cubes, wrist and scene cameras. Randomize textures,
   light, camera pose, servo backlash (0.9 degrees) and friction.
   Deviation (2026-10-06): the scene is in MuJoCo before step 1 ends. Rungs 1 and 2 do not depend
   on insertion physics, and MuJoCo has the official SO-101 model. If another simulator wins
   step 1, rungs 3 and higher move to it.
3. Scripted data: 1,000 or more reach and pick-and-place episodes. Make the episodes long, because
   the upper levels need long windows (H-JEPA failed on short Push-T episodes).
4. Train a flat LeWM, a flat LpWM (A14) and a 2-level H-JEPA (H-JEPA code). Offline tests: the
   rank of the expert's move among random moves, and the plan direction (skunkworks tests).
5. Community data: train on the 23k SO-100 episodes. Test on held-out episodes (A5, offline part).
6. Closed loop in sim: rung 1, then rung 2 with only the final goal image. The flat models plan
   with gradient descent and with CEM (A7, A14).
7. You: print the parts (gauge files first). Record a video when you assemble the arm (scene
   camera and top camera). This is real assembly data (A13).

### Phase 1: the real arm (after the build)
1. Safety client and arm model for the SO-101. Fault tests in sim first.
2. Teleoperation data with the leader arm: the rung 1 and 2 tasks, and free play.
3. Real rungs 1 and 2. Then the next rung that passed in sim.

## 8. Decisions and reviews

| Date | Decision |
|---|---|
| 2026-10-06 | North star: an SO-101 assembles an SO-101. The robot-friendly SO-101 first, the stock arm last. |
| 2026-10-06 | An actor is permitted, but only as a proposer. The world model and the cost select the action. |
| 2026-10-06 | New repository (this one). The lessons of skunkworks carry over as methods, not as code. |
| 2026-10-06 | Noah: try LpWM again. The skunkworks test (2026-10-03) was not decisive: frozen V-JEPA 2.1 encoder, 100 real episodes (it overfit), gradient planner only, offline tests only. Here it is trained end to end, in sim, with both planners and the closed-loop rungs (A14). |
| 2026-10-07 | Rung 2 failed (try 1): the image latent is blind (probe, A15). H-JEPA and community training are held, since they share the encoder. Noah: the goal is an SO-101 that picks up a block and moves it to a target position, with a full JEPA stack (no VLM or VLA); I decide the experiments. Next: encoder variants against A15's pass mark, then rung 2, then H-JEPA. |

## 9. Results log

| Test | Date | Result |
|---|---|---|
| Step 1: CAD geometry (bakeoff/geometry.py) | 2026-10-06 | The STS3215 slides into the SO-101 base along one axis. The CAD has zero clearance on 4 faces (0.05 mm overlap from tessellation), so the test makes the servo smaller by 0, 0.2 and 0.4 mm per side. Convex pieces (CoACD, threshold 0.01, at most 64 vertices; 191 base pieces) reach up to 0.19 mm into the pocket (p99 0.13 mm); threshold 0.03 reached 0.59 mm. |
| Step 1: MuJoCo 3.14 insertion (600 trials, bakeoff/insert_mujoco.py) | 2026-10-06 | 0.4 mm, aligned hand: 100 of 100 in (median error 0.13 mm). 0.4 mm, perturbed (1.5 mm, 2 degrees): 4 of 100. 0.2 and 0 mm: 0 of 400 (the servo jams 5.5 mm in; the 0.19 mm decomposition error is the probable cause at 0.2 mm). No unstable trial; deepest penetration 0.34 mm. 7,100 physics steps/s (4,800 when seated, one laptop core). Render on Iris Xe (EGL): 160 fps at 64x64, 148 fps at 480x640. |
| Step 1: ManiSkill engine, SAPIEN 3 PhysX CPU (600 trials, bakeoff/insert_maniskill.py) | 2026-10-06 | 0 of 600 in, at every clearance. With the PhysX defaults, velocity spikes (unstable trials) and 2-7 mm penetration; the best of 4 settings (TGS, 0.5 mm contact offset, slow push-out) removed the spikes, but PhysX then reported contact depths of 31-35 mm, which are not physical for a 45 mm servo: a fault in our PhysX setup or in its convex-piece handling, not found yet. 6,000 physics steps/s. Render 1,393 fps at 64x64 (8 times MuJoCo), 168 fps at 480x640. |
| Step 1: Isaac Lab (exact-mesh SDF contact; bakeoff/insert_isaaclab.py) | 2026-10-06 | Not run: needs a Linux RTX GPU (Brev paused by Noah). Script and setup are written, not tested. |
| Step 1: decision | 2026-10-06 | MuJoCo, provisionally: the only simulator that inserted the servo and stayed physical. Open: Isaac Lab's SDF contact, and an exact-mesh contact in MuJoCo (SDF plugin), to remove the decomposition error that blocks the 0.2 mm case. |
| Step 1: why MuJoCo jammed at 0.2 mm (bakeoff/path_check.py) | 2026-10-06 | A static check along the insertion path: the CAD meshes never overlap at 0.2 mm clearance, but the convex pieces overlapped by 0.18 mm from 6 mm in, where the servo jammed (0.05 mm from the base split, 0.09 mm from the servo split). A finer base split alone (CoACD 0.005, 984 pieces) removed the base part and still jammed (0 of 20). With the servo split also finer (0.01, 616 pieces) the pieces overlap 0.001 mm, and MuJoCo inserts at 0.2 mm: 20 of 20, aligned hand (median error 0.07 mm). The cause was the convex split, not MuJoCo's contact. These parts are now the default (bakeoff/parts). Physics: 1,000-3,200 steps/s (more pieces). The perturbed hand (1.5 mm, 2 degrees) still fails at every clearance: a hand-strategy problem for rung 3. |
| Step 1: both simulators again with the fine parts (100 trials per case) | 2026-10-06 | MuJoCo: 0.2 and 0.4 mm with the aligned hand, 100 of 100 each (median error 0.07 mm); perturbed hand 2 and 1 of 100; 0 mm (the CAD's zero clearance) 0 of 100; 1 unstable trial in 600. In 14 of the 100 trials at 0.2 mm, one pair of pieces (base_41, servo_166) reports a contact depth of 6.4 mm in the last 0.3 mm before the seat, where the CAD has zero gap at the pocket's end wall; the median deepest contact per trial is 0.011 mm and every trial seats, so this is a depth artifact of that pair, not a physics failure. 1,000-3,200 steps/s. ManiSkill (PhysX): 0 of 600 again, now with 9 mm reported depths; the PhysX fault is not in the parts. Decision unchanged: MuJoCo. Open: Isaac Lab (GPU), a hand that corrects 1.5 mm errors. |
| Step 2: SO-101 scene (sim/scene.py, env.py) | 2026-10-06 | Official SO-101 MJCF. The jaw collision hulls fill the gap between the fingers (the cube cannot enter), so box pads on the measured finger faces replace them. Wrist camera at the front of the camera module, aimed between the fingers; scene camera above and in front. Randomized per episode: camera pose (+-2 cm), light, table and cube colour, cube friction, joint offsets (+-0.45 degrees, calibration and backlash), servo gain (+-20%). |
| Step 3: expert and collector (sim/expert.py, collect.py) | 2026-10-06 | 60 s play episodes at 5 Hz: pick-and-place (60%), reach (25%), free motion (15%). In the 3 checked episodes the expert lifts the cube 11-12 cm and places it. Data: 64x128 frames (scene and wrist), joints, joint-target changes, and the true state for tests; about 10 KB per step. Speed: 40 steps/s per process on a Linux laptop, 280 on kat-pc (GPU rendering). |
| Step 3: expert test at scale (sim/stats.py, expert_eval.py) | 2026-10-06 | The first 1,300-episode dataset was faulty. Over 5,063 pick-and-place tries the expert lifted the cube in 66% and placed it within 2 cm in 40% (the 3 checked episodes had hidden this). Causes: (1) the grasp target was 1.2 cm too low: the finger pads reach 25 mm past the grasp point, so the arm pressed the pads into the table and the cube slipped; (2) past 26 cm the arm is almost straight, and lifts failed in 56 of 99 tries. Fixes: grasp with the pad tips 3 mm over the table, place with the cube 2.5 mm over it, and cubes and goals 15 to 25 cm out (also for the rung 1 and 2 tests, see section 6). Fast test (100 episodes, no rendering): 88% lifted, 79% placed within 2 cm (median error 1.1 cm). The dataset was made again with the fixed expert. |
| Step 3: dataset, second build (sim/make_dataset.sh, stats.py) | 2026-10-06 | 1,200 train episodes (seeds 1,000,000 and up) and 100 val episodes (2,000,000 and up), 300 steps each: 360,000 and 30,000 steps, 3.5 and 0.3 GB (10-frame image chunks). Expert in the train data: 5,366 pick-and-place tries, 91% lifted, 83% placed within 2 cm; val: 90% and 79%. Share of steps: pick-and-place 88%, reach 7%, free motion 5%. No NaN; no dark or flat frame in 600 sampled frames per split. In 2 train episodes the cube leaves the table. 50 min on samsung-2 (14 processes). |
| Step 5: community data (community/build.py, verify.py) | 2026-10-06 | lerobot/community_dataset_v3 (revision ab92ac3f), a sample of 441 SO-100/SO-101 datasets (105 GB of video), one camera each, 5 Hz, 64x64. Test = 43 whole datasets (setups not in training). Train: 388 datasets, 13,692 episodes, 1,246,544 steps (13.5 GB); test: 1,092 episodes, 128,640 steps (1.4 GB). 10 datasets left out for data errors (black frames, frozen video, joints wrapped past 360 degrees, leader and follower calibrations that differ by 45 degrees or more). Normalization per dataset: z-scores of the joint angles (proprio) and of leader minus follower (action), with the difference wrapped to +-180 degrees, a std floor (5 and 1 degrees) and a clip at +-10; without these, 0.03% of the steps had z up to 38. Frames: 1 blank and 2 green ones among 5,000 random train frames, none in test. |
| GPU for steps 4-6 | 2026-10-06 | Brev paused by Noah. Kaggle's weekly GPU quota is used up (32.8 h); Noah chose to wait for its reset (about 2026-10-10). Steps 4-6 wait. |
| GPU: WSL on kat-pc | 2026-10-06 | Noah approved Ubuntu 24.04 in WSL2 on kat-pc (RTX 4060 Ti, 8 GB); the fleet maintainer installed it. Jobs run through wsl.exe (hjepa/wsl_job.sh) with --gpu-gb 7 --ram-gb 20 --cpus 20. The GPU is shared with SailingGame's Unreal work, so jobs wait in the queue while it runs. |
| Training speed on kat-pc (hjepa/smoke.sh, speed.sh) | 2026-10-06 | WSL: CUDA works; MuJoCo renders only 4-5 frames/s there (no GPU EGL), enough for the closed loop (70 s per reach trial with gradient descent, 37 s with CEM). At batch 128 the models filled the 8 GB GPU and spilled into shared system memory: 70 samples/s for LeWM, 11 for H-JEPA. ViT gradient checkpointing (same recipe, extra compute) brings LeWM to 298 samples/s (2.7 GB), LpWM to 293 (2.6 GB) and H-JEPA to 180 (4.4 GB): about 3.3 h for 10 LeWM epochs and 5.3 h for H-JEPA. |
| H-JEPA on Windows | 2026-10-06 | Training stops on kat-pc: stable-pretraining uses POSIX signals (SIGUSR1). Training and planning run on Linux only. The full chain (data, training, offline tests, closed loop) passed a smoke test with a 3-step model. |
| kat-pc sharing | 2026-10-06 | LeWM training died at epoch 6.2 ("CUDA error: unknown error") while a SailingGame Unreal build (queued, 8 threads and 10 GB declared) ran beside it. Cause not found (no driver reset in the Windows log; RAM is probable). Noah's decision: fleet ranks projects, SailingGame first; fleet stops lower-ranked jobs and requeues them. So our jobs are stoppable and resumable: training resumes from a checkpoint (every 500 steps), the WSL side ends on a stop (tested), closed-loop runs skip finished seeds. |
| Step 4: LeWM training (so101_lewm) | 2026-10-06 | 10 epochs, 27,750 steps, batch 128, about 3.5 GPU hours on kat-pc (resumed once). Held-out prediction loss: 0.021 after epoch 1, 0.017 after epoch 3, then flat (0.017 at epoch 10; train 0.015): no overfit, no gain after epoch 3. Inverse dynamics 0.014 (train 0.008). SIGReg 5.4 to 3.1. The latent has mean 0 and std 1, but values up to +-15 to 20 (a Gaussian gives about +-5). |
| Step 4: LeWM offline tests (hjepa/offline.py; val episodes, 500 cases) | 2026-10-06 | Rank of the expert's move among 100 random moves (0 best, 0.5 chance): 0.012 +- 0.006 for a goal 0.2 s ahead, 0.21 +- 0.02 at 1 s, 0.42 +- 0.03 at 3 s. Plan direction (cosine with the expert's move, 250 cases): gradient descent 0.35 +- 0.06 at 1 s, 0.16 +- 0.06 at 3 s; CEM 0.40 +- 0.06 and 0.16 +- 0.06 (CEM minus gradient descent at 1 s: +0.05 [+0.01, +0.10], paired). The one-step signal is strong at 0.2 s and almost gone at 3 s. |
| Step 4: LpWM offline tests (A14; same 500 cases, paired with LeWM) | 2026-10-07 | LpWM trained 10 epochs (stopped and resumed 4 times by fleet ranks). Expert rank: 0.002 at 0.2 s (LeWM 0.012; difference -0.010 [-0.015, -0.005]), 0.19 at 1 s (-0.019 [-0.036, -0.001]), 0.44 at 3 s (+0.016 [-0.005, +0.036]). Plan direction with CEM at 1 s: 0.45 (LeWM 0.40; +0.042 [+0.004, +0.079]); at 3 s 0.12 (LeWM 0.16; -0.044 [-0.088, -0.000]). Gradient descent: no difference. Small LpWM gains at short horizons, a small loss at 3 s. |
| Step 6: rung 1, flat models (sim/closed_loop.py, seeds 5000-5099) | 2026-10-07 | PASS (mark 90/100): LeWM gradient descent 97/100 (median error 0.30 cm), LeWM CEM 97/100, LpWM gradient descent 100/100 (0.17 cm; LpWM minus LeWM +3% [0, +7%], McNemar p = 0.25). Not a vision test: the goal has joint angles, and the probe (below) shows the planner reaches by them. |
| Step 6: rung 2, flat models (seeds 5000-5049) | 2026-10-07 | 0/50 for LeWM (gradient descent and CEM) and LpWM (gradient descent). The cube was lifted in 1 of 150 trials; median error 12.9 cm (the cube stays where it starts). Fail, try 1 of 3 (rule 4). |
| Probe of the latents (hjepa/probe.py; ridge, 20 held-out val episodes) | 2026-10-07 | The pixel latent is almost blind. RMS error of the grasp point from the pixel latent: 12.1 cm (LeWM), 12.4 cm (LpWM), against 12.8 cm for no features (R^2 0.10 and 0.05). Cube on the table: 12.6 and 13.0 cm against 12.7 cm (R^2 below 0). The full latent finds the grasp point (1.3 cm) only through the joint angles. So the encoders learned image features that do not follow the arm or the cube; rung 2 cannot work. Probable cause: a shortcut. The joint angles already solve prediction and inverse dynamics, and per-episode features (light, colours, camera pose) are constant in an episode, so easy to predict, and spread enough for SIGReg. To test next, before H-JEPA (same encoder): within-episode variance of the pixel latent, and encoders that must use vision. |
| Image ceiling (hjepa/ceiling.py) | 2026-10-07 | A small supervised CNN (6 epochs on 120,000 train frames, 12 min on samsung-2's CPU) finds, in the 64x128 frames of the val episodes: the cube to 1.3 cm RMS, the resting cube to 1.5 cm, the grasp point to 1.3 cm (spread 12.5 cm). The frames show the cube; the blind latent comes from the objective, not the resolution. 1.5 cm is near the 2 cm marks, so resolution is a later lever. |

## 10. Prior evidence

- Skunkworks (frozen V-JEPA 2.1, WidowX AI, sim): the bilinear model with Gauss-Newton reached 97
  of 100 (1-move plans); 3-move plans 24 of 100. Pick-and-place grasped in 9 of 17 trials, with
  hand-made sub-goals. With only the final goal, the gripper never came near the cube. The L1 token
  distance does not measure progress. The frozen tokens do not show a 2.5 cm cube.
- H-JEPA ([2610.06805](https://arxiv.org/abs/2610.06805)): learned hierarchy, multi-start gradient
  descent planner. AntMaze 18 to 73%, Cube 32 to 63%. DROID (open-loop) 33.6 to 38.4. Deep
  hierarchies failed on Push-T (short episodes). Closed loop on a real robot: not yet shown.
- LeVJEPA ([2608.27395](https://arxiv.org/abs/2608.27395)): video encoder with SIGReg, 5.6 to 20.8
  times less compute than V-JEPA 2. No robot results. Its gripper probe errors were larger than
  those of V-JEPA 2.1 in skunkworks.
- LpWM ([2608.22764](https://arxiv.org/abs/2608.22764)): sparse latents help small predictors;
  plans with CEM. In skunkworks, an LpWM-type model overfit 100 real episodes.
- The planning-limits paper (2609.39235): a latent distance plans reliably only 5 to 10 steps ahead.
- EP-IDM ([2610.07540](https://arxiv.org/abs/2610.07540), Toso, LeCun et al.): prediction plus
  SIGReg can reach its minimum while the encoder drops directions the actions control (Lemma 2; our
  probe shows this). Reconstructing the H actions from the first and last latents keeps every
  direction reachable in H steps. CartPole latent LQR 0 to 100%; Walker2D 0.08 to 2.91 m/s. Same
  small architecture as ours (ViT-tiny from scratch, AdaLN predictor, CEM 300/30). Our variant E.
- EpicWorldModel ([2610.05996](https://arxiv.org/abs/2610.05996)): stochastic predictor (flow
  matching, 1-2 steps) samples several futures; CEM adds an exploration bonus from their spread.
  Big gains under occlusion (RoboCasa navigation 45 to 72-90%, PointMaze Giant 65 to 86%), none on
  the OGBench manipulation scene (64 vs 60-62%). Its encoder objective is LeWM's, so it does not fix
  A15. Later use: one-model uncertainty (planner cost, A9 practice selection) and grasps that can
  succeed or slip.
- NVIDIA Factory, IndustReal and AutoMate: contact-rich assembly learned in sim transferred to real
  arms (reinforcement learning policies).
