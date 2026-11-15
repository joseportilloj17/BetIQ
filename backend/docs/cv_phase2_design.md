# Computer Vision Phase 2 -- Technical Design Document

**Date:** 2026-11-07
**Status:** Planning / Research -- no video processing or model training yet
**Data source:** BDB (NFL Big Data Bowl) 2023 season tracking data at `data/bdb/`

---

## Table of Contents

1. [Training/Validation Labels from BDB Tracking Data](#1-trainingvalidation-labels-from-bdb-tracking-data)
2. [Field Registration and Jersey ID Approach](#2-field-registration-and-jersey-id-approach)
3. [Video Source Legality](#3-video-source-legality)
4. [Appendix: Key Repositories and Papers](#appendix-key-repositories-and-papers)

---

## 1. Training/Validation Labels from BDB Tracking Data

### 1.1 Data Inventory

The BDB dataset contains 18 weeks of 2023 NFL season pass-play tracking data:

| File(s) | Description |
|---------|-------------|
| `input_2023_w01.csv` ... `input_2023_w18.csv` | Per-frame tracking for all players on each pass play |
| `output_2023_w01.csv` | Target prediction output (future x,y positions for select players) |
| `supplementary_data.csv` | Play-level metadata: formation, route, coverage, EPA, etc. |
| `bdb_closest_cb.csv` | Pre-computed closest-CB-to-targeted-receiver per play |

**Input file columns:**
`game_id, play_id, player_to_predict, nfl_id, frame_id, play_direction, absolute_yardline_number, player_name, player_height, player_weight, player_birth_date, player_position, player_side (Defense/Offense), player_role (Targeted Receiver / Defensive Coverage / Other Route Runner / Passer), x, y, s (speed), a (acceleration), dir, o (orientation), num_frames_output, ball_land_x, ball_land_y`

**Supplementary columns of interest:**
`offense_formation, receiver_alignment, route_of_targeted_receiver, team_coverage_man_zone, team_coverage_type, pass_result, pass_length, play_action, dropback_type`

**Scale estimate:** With ~14K pass plays across 18 weeks, ~22 players per play, and ~20-30 frames per play, the dataset contains roughly **6-9 million player-frame observations**.

### 1.2 Label Type 1: Player Position Labels (Supervised Detection/Tracking)

**Purpose:** If we can detect and track players in broadcast video, we need ground-truth (x, y) positions in field coordinates to evaluate accuracy. BDB tracking data provides this.

**Extraction method:**
- Each row in `input_2023_w*.csv` is one player at one frame: `(game_id, play_id, nfl_id, frame_id)` -> `(x, y)` in NFL field coordinates (0-120 yards x 0-53.3 yards).
- Group by `(game_id, play_id, frame_id)` to get a full field snapshot: all 22 players plus the ball at each 0.1-second interval.
- Each frame becomes a labeled "scene" with 22 bounding-box-equivalent positions.

**Estimated label count:**
- ~14,000 plays x ~25 frames/play = **~350,000 labeled frames**
- Each frame has ~22 player positions = **~7.7 million player-position labels**
- These are field-coordinate labels (not pixel labels), so they serve as validation targets once a homography maps broadcast pixels to field coordinates.

**Usability notes:**
- Labels are in field coordinates, not pixel coordinates. To use them for training a pixel-space detector, you would need to establish frame-to-field correspondence first (i.e., solve the homography problem). This makes them most useful as *validation* labels rather than direct training labels for detection.
- However, they are excellent *training* labels for a trajectory prediction model or a tracking-quality evaluator: given a CV system's estimated player tracks, compare against BDB ground truth.

### 1.3 Label Type 2: CB-WR Matchup Labels

**Purpose:** Identify which cornerback is covering which wide receiver on each play. This is a core analytics question for NFL coverage analysis.

**Extraction method (two approaches):**

**Approach A -- Pre-computed (`bdb_closest_cb.csv`):**
- Already computed. One row per play: targeted receiver paired with closest CB at some reference point.
- Columns: `wr_name, wr_nfl_id, closest_cb_name, closest_cb_id, closest_cb_dist, team_coverage_man_zone, team_coverage_type`
- Quick but limited: only covers the *targeted* receiver, not all WR-CB pairs on the play.

**Approach B -- Derived from tracking data (more complete):**
1. For each play, identify all players with `player_position in (WR, TE)` on offense and `player_position in (CB, SS, FS)` on defense.
2. At snap (frame_id=1), compute pairwise distances between all offensive skill players and all defensive backs.
3. For man coverage (`team_coverage_man_zone = MAN_COVERAGE`): assign each DB to their nearest WR/TE. Track minimum average distance over the route to confirm assignment.
4. For zone coverage: assignment is less meaningful, but you can still compute the closest DB to each receiver at the catch point or at each frame.
5. Output: `(game_id, play_id, wr_nfl_id, cb_nfl_id, avg_separation, min_separation, coverage_type)`

**Estimated label count:**
- ~14,000 plays x ~3-5 WR/TE per play = **~42,000-70,000 WR-CB pairing labels**
- Man coverage plays (~40-50% of plays) yield the cleanest assignments: **~6,000-7,000 plays with clear man assignments**
- The pre-computed `bdb_closest_cb.csv` has one label per play for the targeted receiver only: **~14,000 labels**

### 1.4 Label Type 3: Formation/Alignment Labels

**Purpose:** Classify the offensive formation from pre-snap player positions. A CV system could learn to read formations from broadcast video.

**Extraction method:**
- `supplementary_data.csv` provides `offense_formation` and `receiver_alignment` per play, human-labeled by NFL charting.
- Formation values observed: `SHOTGUN, SINGLEBACK, I_FORM, PISTOL, EMPTY, JUMBO, WILDCAT` (and possibly others).
- Receiver alignment values: `3x1, 2x2, 3x2, 2x1, 4x1, 2x0, 3x0` etc.
- Combine tracking positions at frame_id=1 (pre-snap) with the formation label for a supervised classification dataset.

**Estimated label count:**
- **~14,000 formation labels** (one per play)
- **~14,000 receiver alignment labels** (one per play)
- Each label comes with the full (x, y) positions of all 22 players at the snap -- a rich feature set for classification.

**Label quality:** These are NFL-charted labels, as reliable as it gets. The combination of position coordinates + formation label is directly trainable for a classifier that maps pre-snap spatial configuration to formation type.

### 1.5 Label Type 4: Route Labels

**Purpose:** Classify the route run by a receiver. Route recognition from video is a high-value CV task.

**Extraction method:**
- `supplementary_data.csv` provides `route_of_targeted_receiver` per play.
- Route values observed: `IN, OUT, POST, CORNER, HITCH, CROSS, FLAT, GO, SLANT, COMEBACK, SCREEN, WHEEL, ANGLE, DIG` and others.
- Pair each route label with the full trajectory (x, y over all frames) of the targeted receiver from the tracking data.

**Estimated label count:**
- **~14,000 route labels** (one per play, for the targeted receiver only)
- Other route runners on the same play are in the tracking data but do NOT have route labels -- only the targeted receiver's route is labeled.
- If you need multi-receiver route labels, you would need to build a route classifier trained on the targeted-receiver data and then apply it to the other receivers' trajectories (a natural bootstrap).

**Potential enrichment:**
- From tracking data, extract route features: break point (frame where direction changes sharply), depth of route, lateral displacement, speed at break, etc.
- These features + the route label make a clean supervised dataset for route classification from trajectory data.

### 1.6 Label Summary Table

| Label Type | Source | Count (est.) | Granularity | Quality |
|------------|--------|--------------|-------------|---------|
| Player (x,y) positions | input CSVs | ~7.7M player-frames | Per player per frame | GPS-grade (RFID chips) |
| CB-WR matchups (targeted only) | bdb_closest_cb.csv | ~14K | Per play | Proximity-based heuristic |
| CB-WR matchups (all receivers) | Derived from tracking | ~42-70K pairs | Per play | Needs man/zone filtering |
| Formation classification | supplementary_data.csv | ~14K | Per play | NFL-charted, high quality |
| Receiver alignment | supplementary_data.csv | ~14K | Per play | NFL-charted, high quality |
| Route of targeted receiver | supplementary_data.csv | ~14K | Per play | NFL-charted, high quality |
| Coverage scheme | supplementary_data.csv | ~14K | Per play | NFL-charted (man/zone + type) |

### 1.7 What Labels We Cannot Extract

- **Pixel-space bounding boxes:** BDB has field coordinates, not video frames. We cannot produce labeled bounding boxes in broadcast footage from this data alone.
- **Jersey number labels in image space:** No image data in BDB. Jersey number recognition must be trained on separate annotated video datasets.
- **Non-pass plays:** BDB 2023 is pass plays only. Run plays, special teams, etc. are not covered.
- **Pre-snap motion:** The tracking data starts at or near the snap. Extended pre-snap motion (shifts, jet motions) may not be fully captured depending on frame range.

---

## 2. Field Registration and Jersey ID Approach

### 2.1 Homography / Field Registration

**The problem:** Map each pixel in a broadcast camera frame to a (x, y) coordinate on the standardized 120 x 53.3-yard NFL field. This is the prerequisite for connecting anything detected in video to the BDB tracking data.

#### State of the Art

| Method/Project | Year | Sport | Approach | Accuracy | Open Source |
|---------------|------|-------|----------|----------|-------------|
| **Roboflow `sports`** | 2024-26 | Soccer, adaptable | YOLOv8 keypoint detection on field landmarks + `cv2.findHomography` | mAP@50 ~99%+ on soccer keypoints | Yes (MIT) -- `github.com/roboflow/sports` |
| **narya** | 2021 | Soccer | Direct homography estimation + keypoint refinement with UNet | Good when penalty box visible; degrades on wide shots | Yes -- `github.com/DonsetPG/narya` |
| **SoccerNet Camera Calibration** | 2023-26 | Soccer | Benchmark with keypoint + line detection; JaC@5 metric | Top methods: IoU 93-97%, reprojection error < 5px | Yes -- `github.com/SoccerNet/sn-calibration` |
| **PnLCalib** | 2026 | Multi-sport | Points + lines optimization; handles lens distortion | State-of-the-art on SoccerNet benchmark | Yes -- `github.com` (paper: arxiv 2404.08401) |
| **BroadTrack** | 2025 | Soccer | Camera + tripod model for temporal consistency | Improved temporal stability over single-frame methods | Yes (WACV 2025) |
| **sports-field-homography** | 2022 | Basketball/soccer | UNet + Spatial Transformer Network (end-to-end) | Reasonable; less accurate than keypoint methods | Yes -- `github.com/darkAlert/sports-field-homography` |
| **Unified Sports Field Registration** | 2026 | Multi-sport | Lens distortion + multi-sport calibration | Cross-sport generalization | CVPR 2026 Workshop paper |

#### NFL-Specific Challenges

Soccer homography is the most mature because:
1. The field has many distinctive landmarks (penalty box, center circle, corner arcs).
2. SoccerNet provides large annotated benchmarks.

**NFL is harder** because:
- Fewer distinctive field landmarks in a typical broadcast frame (yard lines are parallel and repetitive; hash marks are small).
- The camera angle is typically higher and more zoomed than soccer broadcasts.
- End zone text and logos vary by team and are not standardized keypoints.
- The field is narrower (53.3 vs 68-75 yards for soccer) with less geometric diversity per frame.

#### Recommended Approach for NFL

1. **Start with Roboflow `sports` + `supervision` library.** It is the most actively maintained, has the best documentation, and the keypoint detection approach is adaptable.
2. **Create an NFL field keypoint dataset.** Define ~30-40 keypoints on an NFL field template: corners of yard-line markings, hash mark intersections, sideline-yardline intersections, goal post bases. Manually annotate 200-500 frames across different stadiums/camera angles.
3. **Train a YOLOv8-pose model** on these NFL keypoints. This is the same approach Roboflow demonstrates for soccer.
4. **Compute homography** from detected keypoints to the standard field template using `cv2.findHomography` with RANSAC.
5. **Validate** against BDB tracking data: project detected player positions through the homography and compare to BDB (x, y) for the same play.

#### Expected Accuracy

- **Soccer state of the art:** 93-97% IoU, <5px reprojection error.
- **NFL realistic target (Phase 2):** Expect worse initially -- perhaps 80-90% IoU -- due to fewer landmarks and no existing benchmark. The yard-line repetition problem (is that the 30 or the 40?) requires reading yard numbers or using game context.
- **Acceptable for analytics:** 1-2 yard accuracy in player localization is sufficient for formation recognition and coverage analysis. Sub-yard accuracy matters for separation metrics.

#### Compute Requirements

- **Keypoint annotation:** Manual, ~2-4 hours for 200 frames using Roboflow or CVAT.
- **YOLOv8-pose training:** 1-2 hours on Apple Silicon M-series (MPS backend) for a small keypoint dataset. No GPU cluster needed.
- **Inference:** YOLOv8 on Apple M2 via CoreML/Neural Engine: 21-25 FPS. Sufficient for offline analysis.

### 2.2 Jersey Number Recognition

**The problem:** Given a detected player bounding box in broadcast video, identify which player it is by reading the jersey number and/or team colors.

#### State of the Art

| Method/Project | Year | Approach | Accuracy | Notes |
|---------------|------|----------|----------|-------|
| **mkoshkina/jersey-number-pipeline** | 2024 | Legibility classifier + PARSeq scene-text recognition | 87.5% (soccer), 91.4% (hockey) at tracklet level | Best open-source general framework |
| **Roboflow basketball pipeline** | 2025-26 | RF-DETR detection + SAM2 segmentation + SmolVLM2 OCR | Practical for basketball; adapts to sports | Uses VLM for OCR -- interesting approach |
| **Football_Player_Recognition** | 2020 | Mask R-CNN detection + Faster R-CNN digit extraction | Proof of concept for American football | Older, less accurate, but NFL-specific |
| **Pose-Guided R-CNN** | 2020 | Pose estimation to locate jersey region + OCR | Improved over naive crop | Academic, limited code |
| **chinmayarvind jersey-number-recognition-boosted** | 2025 | Enhanced pipeline with pose-guided ROI + re-ID filtering | Improved over base mkoshkina pipeline | Built on jersey-number-pipeline |

#### NFL-Specific Challenges

- **Broadcast angle:** NFL broadcasts are typically sideline shots where the jersey number may be on the back (visible) or chest (occluded). Numbers are most readable during replays and All-22 film.
- **Helmet obstruction:** Helmets cover part of the upper body in many frames.
- **Motion blur:** NFL players move at 15-22 mph; motion blur at broadcast frame rates (30fps) is significant.
- **Resolution:** In wide shots with 22 players, each player may only be 30-60 pixels tall. Jersey digits at that scale are 5-10 pixels.
- **Uniform similarity:** Same-team players have identical uniforms except for the number.

#### Recommended Approach

1. **Player detection first:** Use YOLOv11 or YOLO26 fine-tuned on sports broadcast data (Roboflow has pre-trained models for football player detection).
2. **Team color segmentation:** Before OCR, classify each detection as Team A, Team B, or Referee based on dominant jersey color. This is high-accuracy (>95%) and halves the identity search space.
3. **Jersey number OCR:** Use the mkoshkina pipeline adapted for NFL:
   - Legibility classifier: skip frames where the number is not readable (back turned, too far, blur).
   - Scene text recognition: PARSeq or TrOCR on the jersey region crop.
   - Tracklet aggregation: across multiple frames of the same tracked player, take the mode of recognized numbers.
4. **Fallback: VLM approach.** For difficult cases, use SmolVLM2 or a similar small vision-language model on the player crop. This handles partial occlusion better than pure OCR.

#### Expected Accuracy

- **Team classification:** >95% from color alone.
- **Jersey number (ideal conditions):** 80-90% per frame on close-up/replay shots.
- **Jersey number (wide broadcast):** 40-60% per frame. Tracklet-level aggregation over 10+ frames can push to 70-80%.
- **Jersey number (All-22 film):** Higher angle, smaller players, but more consistent: expect 50-70% per frame, 75-85% with aggregation.

#### Compute Requirements

- **YOLOv11 detection:** Real-time on Apple Silicon via CoreML (25+ FPS).
- **PARSeq OCR:** Lightweight, <10ms per crop on CPU.
- **SmolVLM2 (fallback):** ~100-200ms per crop on Apple Silicon. Not real-time but fine for offline.
- **Full pipeline per frame:** ~50-100ms on M-series Mac. A full game (~5,400 frames at 30fps for 3 hours) processes in ~5-10 minutes.

### 2.3 Player Detection and Tracking

**The problem:** Detect all players in each frame and maintain consistent identity across frames (multi-object tracking / MOT).

#### State of the Art

| Method | Year | Approach | Notes |
|--------|------|----------|-------|
| **YOLOv11 / YOLO26** | 2025-26 | Single-shot detection | Current best for speed/accuracy tradeoff |
| **RT-DETR** | 2024-26 | Transformer-based detection | Better on small objects; slower |
| **RF-DETR** | 2025 | Roboflow's detection transformer | Optimized for sports; good accuracy |
| **DeepSORT** | 2017+ | Detection + appearance + Kalman filter tracking | Still widely used; robust |
| **StrongSORT** | 2023 | Improved DeepSORT with better appearance model | Better identity maintenance |
| **ByteTrack** | 2022 | Simpler tracking using detection confidence | Faster than DeepSORT, competitive accuracy |
| **BoT-SORT** | 2022 | Camera-motion-compensated tracking | Handles broadcast camera pans well |
| **SAM2 + YOLO** | 2024-26 | YOLO init + SAM2 tracking | Reduces per-frame compute; good for consistent IDs |
| **Roboflow `trackers`** | 2025-26 | Modular tracker library (Apache 2.0) | Plug-and-play with any detector; supports sportsmot |
| **GTATrack** | 2025 | Deep-EIoU + global tracklet association | Winner of SoccerTrack 2025 challenge |

#### Recommended Approach

1. **Detection:** YOLOv11-l or YOLO26-m fine-tuned on football broadcast data. Roboflow Universe has pre-annotated football player detection datasets.
2. **Tracking:** Start with **BoT-SORT** (handles camera motion, critical for NFL broadcasts) via the Roboflow `trackers` library. Fallback to ByteTrack if BoT-SORT is too slow.
3. **Re-identification:** Team color + jersey number form a composite ID. When OCR fails, fall back to appearance embedding (e.g., from a small ResNet trained on player crops).

#### Expected Performance

- **Detection mAP@50:** >90% for players; ~70-80% for the ball.
- **Tracking MOTA:** 70-80% on broadcast (lower than soccer benchmarks due to camera cuts, replays, and huddles).
- **Identity switches:** Expect 5-15 per play sequence. Most occur during pile-ups and camera transitions.

#### Compute Requirements

- **YOLOv11-l inference on M-series Mac:** ~15-20 FPS via CoreML. Sufficient for offline processing.
- **BoT-SORT overhead:** <5ms per frame. Negligible compared to detection.
- **Full game processing (detection + tracking):** ~15-30 minutes for a 3-hour broadcast at 30fps on Apple M2.
- **Storage:** ~50-100 GB per game at 1080p. Processed tracks (JSON): <10 MB per game.

### 2.4 End-to-End Pipeline Architecture

```
Broadcast Video
       |
       v
[1. Field Registration]  -->  Homography matrix per frame
       |                       (keypoint detection + cv2.findHomography)
       v
[2. Player Detection]    -->  Bounding boxes per frame
       |                       (YOLOv11/YOLO26)
       v
[3. Player Tracking]     -->  Consistent track IDs across frames
       |                       (BoT-SORT via roboflow/trackers)
       v
[4. Team Classification] -->  Team A / Team B / Ref per track
       |                       (color histogram on jersey region)
       v
[5. Jersey OCR]          -->  Jersey number per track (when legible)
       |                       (PARSeq + tracklet aggregation)
       v
[6. Field Projection]    -->  (x, y) field coordinates per player per frame
       |                       (homography @ bbox center)
       v
[7. Validation]          -->  Compare to BDB ground truth
                               (matched by game_id, play_id, frame alignment)
```

**Key dependency:** Step 7 requires temporal alignment between the broadcast video and the BDB frame IDs. BDB frames are at 10 Hz (0.1s intervals) starting from the snap. Broadcast video is at 30fps. Aligning the snap event in both streams is a manual step for validation but could be automated via snap detection (ball movement onset).

---

## 3. Video Source Legality

### 3.1 Overview

This section evaluates what video a solo developer can legally use for personal, non-commercial analytics (no redistribution of video, models trained on the video, or derived highlights). The key distinction throughout is between **Terms of Service violations** (breach of contract, civil matter) and **copyright infringement** (statutory, potentially criminal).

### 3.2 NFL Game Pass / NFL+

**What it is:** NFL+ ($6.99/mo) and NFL+ Premium ($14.99/mo) provide live games (mobile only), replays, and All-22/Coaches Film (Premium only).

**Terms of Service (nfl.com/legal/subscriptions_terms):**

The NFL subscription terms explicitly prohibit:

> - "Systematic retrieval of data or other content from the Services, whether to create or compile, directly or indirectly, a collection, compilation, database, or directory"
> - "Monitoring, gathering, copying, or distributing content on the Service by using any robot, rover, 'bot', or other automated technology"
> - Use is limited to "your own individual non-commercial and informational purposes only"

**What this means:**
- **Watching and taking manual notes:** Permitted. This is the intended use.
- **Screen recording individual plays for personal review:** Gray area. The ToS does not explicitly address screen recording, but the broadcast copyright notice prohibits "any pictures, descriptions, or accounts of the game without the NFL's consent." However, legal scholars have noted this language is overly broad and unenforceable for personal use (see Sony v. Universal, 1984 -- "time-shifting" for personal use is fair use).
- **Automated frame extraction / scraping:** Clearly prohibited by ToS. This is "systematic retrieval" and "automated technology."
- **Training a CV model on extracted frames:** Prohibited by ToS (creating a "compilation" or "database" from the content).

**Risk assessment:** ToS violation (breach of contract). The NFL could terminate your subscription. They are unlikely to pursue legal action against a solo developer doing personal analytics, but automated access could trigger account detection and banning. Copyright infringement is a separate, more serious risk if you redistribute anything.

**Verdict:** PROHIBITED for automated processing. Manual observation and note-taking only.

### 3.3 YouTube (NFL Channel, Team Channels, Highlights)

**YouTube Terms of Service (youtube.com/t/terms), Section 5(B):**

> "You are not allowed to access, reproduce, download, distribute, transmit, broadcast, display, sell, license, alter, modify or otherwise use any part of the Service or any Content except: (a) as expressly authorized by the Service; or (b) with prior written permission from YouTube and, if applicable, the respective rights holders."

**What this means:**
- **Watching on YouTube:** Permitted.
- **Downloading via yt-dlp or similar tools:** Violates YouTube ToS. Not authorized by the Service (only YouTube Premium offline mode is authorized).
- **Automated frame extraction:** Violates YouTube ToS ("access... by automated means" is prohibited except for public search engines following robots.txt).
- **Using downloaded content for ML training:** Violates ToS. Also potentially violates the copyright of the content owner (NFL, teams, or uploaders).

**Fair use considerations:**
- Extracting frames for personal research/commentary is potentially defensible as fair use under U.S. law, but this is separate from the ToS violation.
- Fair use factors: (1) Purpose: personal research/non-commercial -- favorable. (2) Nature: factual content -- favorable. (3) Amount: frames from full games -- less favorable. (4) Market effect: no redistribution -- favorable.
- Fair use is an affirmative defense, not a permission. You assert it in court after being sued.

**Verdict:** PROHIBITED by YouTube ToS. Fair use is a possible defense for personal research but does not prevent account termination. In practice, millions of people download YouTube videos for personal use without consequence, but automated large-scale downloading can trigger account/IP bans.

### 3.4 Broadcast Recordings (DVR / Cable)

**Legal basis:**
- **Sony Corp v. Universal City Studios (1984):** The Supreme Court ruled that recording broadcast TV for personal "time-shifting" is fair use. This is settled law.
- **DMCA (1998):** Does not prohibit personal recording of broadcast content you have lawfully received.

**What this means:**
- **Recording games via DVR/cable for personal viewing:** Legal. Established fair use.
- **Extracting frames from your DVR recording for personal analysis:** Likely fair use. You lawfully received the broadcast, you are not redistributing, and the use is personal/non-commercial. This is the strongest legal position among all video sources.
- **Automated processing of your own DVR recordings:** No ToS applies to your local recordings. The only question is copyright fair use, and the factors favor personal non-commercial analysis.
- **Redistributing extracted data (frames, models, highlights):** NOT fair use. Keep everything local.

**Practical considerations:**
- DVR recordings are typically MPEG-TS or similar compressed format. Quality is 720p-1080p depending on your provider.
- You need a way to extract the video file from your DVR. Some providers (YouTube TV, Hulu Live) store recordings in the cloud with DRM. Traditional cable DVR boxes store locally but extraction requires HDMI capture or similar.
- **HDMI capture devices** (Elgato HD60, AVerMedia) can capture HDMI output at 1080p60. Cost: $100-200. This is the most practical path.

**Verdict:** BEST LEGAL OPTION for automated processing. Record via DVR/HDMI capture, process locally, never redistribute.

### 3.5 All-22 / Coaches Film

**Availability:**
- All-22 film is available through **NFL+ Premium** ($14.99/mo) and **NFL Pro** (analytics platform).
- Available 24-36 hours after each game.
- Shows the full field from an elevated end-zone and sideline camera -- ideal for CV analytics because all 22 players are visible.

**Terms:** Same as NFL+ (Section 3.2 above). Automated access is prohibited.

**Alternative access:**
- All-22 clips sometimes appear on YouTube (uploaded by analysts, coaches, film reviewers). Same YouTube ToS restrictions apply.
- Some analysts share All-22 breakdowns on Twitter/X with short clips embedded. These are too short and low-resolution for CV training.

**Why All-22 matters for CV:**
- Broadcast cameras follow the ball, showing only a subset of players. All-22 shows the entire formation and all routes.
- The elevated, relatively static camera angle is much easier for homography estimation than the dynamic broadcast camera.
- All-22 is the ideal video source for validating against BDB tracking data because both show the full field.

**Verdict:** PROHIBITED for automated extraction via NFL+. If captured via HDMI from a legitimate subscription playback, the DVR fair-use argument from Section 3.4 may apply, but it is weaker here because NFL+ is a streaming service with explicit ToS, not a broadcast.

### 3.6 NCAA / College Footage

**Key differences from NFL:**
- The **NCAA** owns championship footage copyright but does NOT own regular-season game rights. Those belong to conferences and their broadcast partners (ESPN, FOX, CBS, etc.).
- **Catapult/Hudl** provides game film to college programs via a closed platform. Unauthorized access is actively investigated by the NCAA (as of the 2023 Catapult investigation).
- College game broadcasts are subject to the same fair use and ToS considerations as NFL broadcasts.
- There is **no equivalent of NFL+ All-22** for college. Coaches film is locked behind Hudl/Catapult and not available to the public.

**Verdict:** Same rules as NFL broadcasts -- DVR recording of public broadcasts is the safest path. No access to coaches film unless you are affiliated with a program.

### 3.7 Synthetically Generated Data

**Game engines (Madden, etc.):**
- **EA EULA** prohibits "alter, modify, enhance, or create a derivative work or improvement of any part of the Software" and prohibits "data mining."
- Screen capture of Madden gameplay for personal use is not explicitly prohibited, but using captured frames as ML training data likely falls under "derivative work."
- **Risk:** Very low enforcement risk for personal use. EA has never pursued action against individuals capturing gameplay footage for non-commercial purposes.

**Custom synthetic generation (Unreal Engine / Unity):**
- **SoccerSynth** (2025-26) demonstrated generating synthetic soccer player detection datasets using Unreal Engine with commercially licensed character models and animations.
- **Unreal Engine license** permits using rendered output for research and non-commercial purposes.
- You can create a synthetic NFL-like field (no trademarked logos) with generic player models running routes.
- This is the **cleanest legal path** for generating unlimited training data, but requires significant upfront effort.

**Practical assessment for a solo developer:**
- Building a full synthetic environment is 40-100+ hours of work.
- The SoccerSynth approach (Unreal Engine + marketplace assets) is the most practical starting point.
- Synthetic data works best for pre-training (field detection, player detection) and requires fine-tuning on real data for production accuracy.

**Verdict:** LEGAL for Unreal/Unity custom generation. Madden screen capture is a gray area (ToS violation, near-zero enforcement risk for personal use). Synthetic data is excellent for bootstrapping but insufficient alone.

### 3.8 Legality Summary Table

| Source | Automated Processing | Manual Analysis | Redistribution | Best For |
|--------|---------------------|-----------------|----------------|----------|
| NFL+ / Game Pass | PROHIBITED (ToS) | OK | PROHIBITED | Manual scouting only |
| YouTube | PROHIBITED (ToS) | OK (watching) | PROHIBITED | Reference viewing |
| DVR / HDMI capture of broadcast | LIKELY FAIR USE | OK | PROHIBITED | **Primary CV data source** |
| All-22 via NFL+ | PROHIBITED (ToS) | OK | PROHIBITED | Manual formation analysis |
| All-22 via HDMI capture during playback | GRAY AREA | OK | PROHIBITED | Best quality for CV, but riskier |
| NCAA broadcasts (DVR) | LIKELY FAIR USE | OK | PROHIBITED | Additional training data |
| NCAA coaches film (Hudl) | PROHIBITED | N/A (no access) | PROHIBITED | Not available |
| Synthetic (Unreal/Unity) | LEGAL | N/A | OK (your work) | **Pre-training, augmentation** |
| Madden screen capture | GRAY AREA (ToS) | OK | PROHIBITED | Quick prototyping |

### 3.9 Recommended Legal Strategy

For a solo developer doing personal analytics:

1. **Primary training data:** HDMI capture of broadcast games you watch live or via DVR. Fair use is strongest here. Process locally, never redistribute frames or trained models.
2. **Supplemental / pre-training:** Synthetic data from Unreal Engine. No legal risk. Use for field detection and generic player detection pre-training.
3. **Validation ground truth:** BDB tracking data (already have it, freely distributed by NFL for research).
4. **Do not:** Scrape NFL+, YouTube, or any streaming service. Do not redistribute any captured frames, trained model weights (if trained on copyrighted video), or derived highlights.
5. **Paper trail:** If you ever publish research, cite the data sources and assert fair use. Do not include copyrighted frames in publications (use synthetic visualizations instead).

---

## Appendix: Key Repositories and Papers

### Field Registration
- [roboflow/sports](https://github.com/roboflow/sports) -- MIT license, keypoint-based field registration + player detection. Most practical starting point.
- [SoccerNet/sn-calibration](https://github.com/SoccerNet/sn-calibration) -- Benchmark and baseline methods for camera calibration.
- [DonsetPG/narya](https://github.com/DonsetPG/narya) -- End-to-end soccer tracking + homography. Older but influential.
- [darkAlert/sports-field-homography](https://github.com/darkAlert/sports-field-homography) -- UNet + STN approach.
- PnLCalib (arxiv 2404.08401) -- Points + lines optimization, multi-sport.
- BroadTrack (WACV 2025) -- Temporal camera tracking for broadcasts.

### Jersey Number Recognition
- [mkoshkina/jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline) -- General framework, 87-91% accuracy. Best starting point.
- [chinmayarvind23/jersey-number-recognition-boosted](https://github.com/chinmayarvind23/jersey-number-recognition-boosted) -- Enhanced pipeline with pose guidance.
- [AlexRamey/Football_Player_Recognition](https://github.com/AlexRamey/Football_Player_Recognition) -- NFL-specific, older.

### Player Detection and Tracking
- [roboflow/trackers](https://github.com/roboflow/trackers) -- Apache 2.0, modular tracker library. Supports sportsmot.
- [supervision](https://supervision.roboflow.com/) -- MIT license, core CV building blocks.
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) -- YOLOv8/v11/YOLO26. AGPL-3.0 (note: AGPL requires source distribution if you deploy as a service; fine for personal use).

### Synthetic Data
- SoccerSynth-Detection (arxiv 2501.09281) -- Synthetic soccer player dataset via Unreal Engine.
- SoccerSynth Field (arxiv 2503.13969) -- Synthetic field detection dataset.

### Benchmarks
- [SoccerNet](https://www.soccer-net.org/) -- Multi-task benchmark including tracking, calibration, jersey numbers.
- [sportsmot](https://github.com/MCG-NJU/SportsMOT) -- Multi-object tracking benchmark for sports.

### Legal References
- *Sony Corp. of America v. Universal City Studios, Inc.*, 464 U.S. 417 (1984) -- Fair use for time-shifting.
- NFL Subscription Terms: https://www.nfl.com/legal/subscriptions_terms
- YouTube Terms of Service: https://www.youtube.com/t/terms
- EA User Agreement: https://tos.ea.com/legalapp/WEBTERMS/US/en/PC/
- Public Knowledge, "The NFL Wants You To Think These Things Are Illegal" (2007): https://publicknowledge.org/the-nfl-wants-you-to-think-these-things-are-illegal/

---

## Next Steps (Phase 2 Execution Checklist)

When ready to move from planning to implementation:

- [ ] **Label extraction scripts:** Write Python to extract all label types from BDB CSVs (Section 1). Output: structured JSON/parquet with labels.
- [ ] **NFL field template:** Define keypoint coordinates on a standardized NFL field SVG/JSON. ~35-40 keypoints.
- [ ] **Annotate 200 frames:** Capture 200 diverse broadcast frames (HDMI capture), annotate field keypoints in CVAT or Roboflow.
- [ ] **Train keypoint model:** YOLOv8-pose on the NFL keypoint dataset.
- [ ] **Set up detection pipeline:** YOLOv11 + BoT-SORT using roboflow/trackers on a single game.
- [ ] **Validate against BDB:** Align one game's CV output to BDB tracking data, measure position error.
- [ ] **Synthetic environment (optional):** If real-data volume is insufficient, build UE5 synthetic field for pre-training.
