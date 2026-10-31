# ORBIT-PRISM: speech, slide by slide (about 10 minutes)

Based on `ORBIT-PRISM_Final.pptx` (18 slides). Numbers marked **[measured]** come from the real run on the held-out Bolivia event (`results/summary.csv`, `results/dial.csv`, `results/memory_table.csv`, `results/demo_case.json`). The deck's slides 10, 11 and 15 still say "to be measured": update them to the numbers below before presenting, or say the numbers aloud and keep the slides as they are. Speak at about 140 words a minute.

Words we do not use: outperforms (unless the number is on screen), guarantees (without its condition), real-time on satellite hardware, danger score, disaster management system.

---

## Slide 1: Title (0:00 to 0:30)

Good morning. We are team ORBIT-PRISM, six of us, working on problem statement 1.4 from Kepler Aerospace. A small satellite captures far more than it can ever send down. ORBIT-PRISM is a tiny machine-learning layer on the satellite that decides what goes first. It cuts each image into sixteen tiles, scores them, and, like a prism, splits the important ones into three layers: a 32-byte alert first, then a small preview, then the full tile. In one line: you set the miss rate, we minimise the bytes, even through cloud.

## Slide 2: The problem (0:30 to 1:10)

Start with the numbers. A satellite captures gigabytes every day. A 256-kilobit link, open for ten minutes, carries 19.2 megabytes. One raw image chip is 6.8 megabytes, so fewer than three fit in a pass. And the chip that does the thinking has only 4.2 megabytes of working memory.

Today that link is wasted three ways: on cloudy frames, on empty frames, and on urgent tiles that wait for the next pass. So we read this as a choice problem, not only a compression problem. The satellite must decide what goes first, and say honestly how many important events it might miss.

## Slide 3: Six objectives (1:10 to 1:35)

That gives six objectives, each with a number we measure. One: fit inside 4.2 megabytes. Two: let the operator promise a miss rate, and test it on a flood the model has never seen. Three: still see floods under cloud. Four: land an alert before any full image. Five: spend every byte on value. Six: prove all of it against four simpler baselines, on the same budget.

## Slide 4: The idea, on a real flood (1:35 to 2:10)

Here is the idea on a real flood from the Sen1Floods11 dataset, shown in infrared so the water is dark blue. We cut the image into sixteen tiles. Six of them hold at least thirty percent flood water, according to human labels. Sending only those six, in our 8-bit format, costs about 0.39 megabytes instead of the 6.8 of the raw chip.

To be clear, those six tags come from human labels, not from our model. The model has to earn that ranking, and the next slides show how we measure whether it does.

## Slide 5: How the water is found (2:10 to 2:50)

A fair question: in true colour, on the left, this flood is hard to see, because muddy water looks like soil. The satellite does not rely on our eyes. Water absorbs infrared light, so in the infrared view it turns dark blue. Radar works differently: smooth water bounces the signal away, so flooded fields come back black, even through cloud.

The bottom row is measured on this chip. Flood pixels reflect less than half the near-infrared of dry land, and return much weaker radar, minus 17 decibels against minus 9. The hand label on the right is what analysts traced. It is what we train and test against.

## Slide 6: Five ideas, three of them core (2:50 to 3:25)

Five features. One, the core idea: a miss-rate dial. A raw confidence score promises nothing, so the operator says how many misses they accept, we calibrate to it, and we show what it costs in bytes. Two, when cloud blocks the camera, the tile switches to radar, which still sees the flood. Three, tiles are ranked by value per byte, so the best fill each pass. Four, a 32-byte alert goes down before any full tile. Five, skipped tiles are logged with a single bit, so the ground knows what was checked, not just what was sent. The first three are our contribution; the last two are supporting design.

## Slide 7: How it works (3:25 to 4:00)

This is one tile's path. The raw chip is bigger than our memory, so it stays in flash and streams in tile by tile. A cheap check asks whether the optical view is usable. If it is, a small 8-bit optical model runs. If the cloud fraction is half or more, an 8-bit radar model runs instead. We subtract permanent water such as rivers, so a river is never called a flood. Then the miss-rate dial keeps or skips the tile, priority is probability times extent, and the scheduler fills the pass. Everything outlined in lime is what we integrate. The last box, the dashboard, runs on the ground.

## Slide 8: Radar under cloud (4:00 to 4:35)

Why the radar fallback matters. Same tile, four views. On a clear day the optical camera sees the flood. Under cloud it cannot. Radar passes through cloud and still shows the flooded ground as black. The hand label is the truth. A discard-cloudy system throws this tile away; ours switches to radar when the cloud fraction passes one half.

Be clear about the limit: the cloud here is simulated, because the dataset has no cloud mask, and this picture is an illustration. Our measured result for it comes on slide 15.

## Slide 9: Scheduler and layered packets (4:35 to 5:10)

Now the scheduler. Instead of sending a whole image or nothing, we send in layers. First a 32-byte alert: where, how likely, how large, with a checksum. It takes about one millisecond on the link. Then a 2-kilobyte preview, a 32 by 32 picture, about 64 milliseconds. Then the full tile, 64 kilobytes, about two seconds.

Tiles are ranked by priority per byte until the 19.2 megabytes are spent, and whatever does not fit carries over to the next pass. About 293 full optical tiles fit in a pass, so the choice of which ones is what matters.

## Slide 10: The miss-rate dial (5:10 to 5:55) **[measured]**

This is our core idea, in a toy example. Take the critical tiles from a calibration set the model never trained on, here nineteen of them. For a five percent target, the cut-off is the lowest of those scores. A new critical tile is missed only if it scores below all nineteen: one chance in twenty, five percent. We do this separately for optical and radar.

Now the real result, on the Bolivia flood the model has never seen. On the optical path, a 5 percent target gave 4.8 percent missed, and a 10 percent target gave 9.5. The dial held. On the radar path it did not: 5 percent gave 9.5 percent, and 10 percent gave 38. Bolivia is a different flood from the calibration data, and the radar scores are weaker after compression, so the promise only holds when test data resembles calibration data. We report what we measured. The 1 percent target needs at least 99 critical calibration tiles; we have 76 optical and 42 radar, so it falls back to sending everything.

## Slide 11: Models and compression (5:55 to 6:35) **[measured]**

The models are deliberately tiny: four small convolution blocks, global pooling, and two outputs, flood probability and flood extent. The radar model is the same design with two input bands. We compress with a ladder: the trained float model, then 8-bit integer quantisation, which makes the weights four times smaller.

The memory table is measured. Optical weights are about 61 kilobytes in float and 16 in 8-bit. Radar is the same. The largest total, including the working activations and one staged tile, is about half a megabyte, far inside 4.2. The peak activation figure is an estimate from the model graph, and we label it that way. The accuracy cost of 8-bit is small: optical recall at our reference threshold goes from 89 to 88 percent. Radar's 8-bit scores are noticeably weaker, which is why we calibrate the dial on the 8-bit scores themselves.

## Slide 12: Data and preparation (6:35 to 7:10)

For data we use Sen1Floods11, a public set of paired radar and optical flood images from eleven real floods. We use the hand-labelled part: about 440 chips, cut into 6,825 tiles of 128 by 128. We remove permanent water, and call a tile critical when at least thirty percent of it is flood. We split by whole flood event, never by tile, and hold one event, Bolivia, completely out.

Two limits up front. The dataset has no cloud mask, so clouds are simulated, as opaque white, which is the harder case. And optical and radar come from two different satellites, so the paired payload is a simulation too.

## Slide 13: Evaluation plan (7:10 to 7:50) **[measured]**

How we prove it: five systems, the same chips, the same budget, one whole flood held out. Capture order. Discard cloudy. Optical-only tile triage. A colour-index threshold with no neural network. And ORBIT-PRISM. Our budget is a scaled pass, twenty percent of the event's data, about 3 megabytes, because Bolivia is small enough to fit in one real pass. We always label it scaled.

The sketch on the slide was a hypothesis. The measured result is on slide 15.

## Slide 14: What is new, and what is not (7:50 to 8:20)

We want to be straight about novelty. Onboard cloud screening, tile selection and confidence thresholds all exist: ESA's Phi-sat-2, Ubotica, TargetFuse, Earth Plus, CloudScout, NASA's EO-1. We do not claim any of those. Our contribution is combining three things in one memory-bounded system, and measuring it: a miss rate the operator sets and we check on an unseen flood, a radar fallback chosen per tile, and a byte-budget scheduler that sends in layers.

## Slide 15: Interface and demo (8:20 to 9:05) **[measured]**

This is the operator interface, and our ninety-second demo has three moves. One: a flood hides under cloud, and the discard-cloudy system loses it. Two: ORBIT-PRISM switches to radar, and the tile is flagged and sent. Three: we move the dial and watch bytes against recall.

The measured comparison, on the held-out Bolivia flood at the scaled pass, out of 42 critical tiles: ORBIT-PRISM delivers 88 percent, optical-only triage 81, the colour index 79, capture order 45, discard cloudy 36. Under cloud, ORBIT-PRISM delivers 19 of 21 critical tiles, optical-only 15, capture order 4, discard cloudy none. Two honest notes. The network beats the colour index in clear conditions by only about one tile, so the gain comes from radar and the dial. And ORBIT-PRISM ends at 93 percent recall, because the dial skips some critical tiles; it never reaches 95. The demo runs from saved results, so it never depends on one laptop, and a backup video is ready.

## Slide 16: Execution plan (9:05 to 9:25)

Six people, one shared tile table that connects every part, so everyone builds at once. Madhavan integrates and evaluates. Agnel built the data pipeline and the dial. Shenbaga trained the optical model, Harish the radar model and the 8-bit compression. Akhilesh built the scheduler and link simulator, and Sriram the website. Every part merged through one gate that checks the file formats and the claims.

## Slide 17: Limits, claims and fallbacks (9:25 to 9:50)

Our limits. Clouds are simulated. The link and memory are simulated, not flight hardware. The paired payload is a simulation. One held-out flood is thin evidence. The miss-rate promise holds only when test data resembles calibration data, and on radar it did not, as we showed. Forest fire is a second hazard we have scoped but not loaded: the fire dataset is real, but we have not measured anything on it. So there are words we do not use: outperforms without a number, guarantees without its condition, real-time on satellite hardware.

## Slide 18: The crux (9:50 to 10:00)

So: you set the miss rate, we minimise the bytes, even through cloud. A calibrated promise, radar under cloud, and byte-budget ranking, all inside 4.2 megabytes, and measured honestly, including where it fell short. Thank you. We are happy to take questions.

---

## If a judge asks (30-second answers)

- **Is it real hardware?** No. It simulates the link and the memory limit. The models are small enough that the weights are about 16 KB and the working memory about 0.4 MB, so the budget is plausible, but we have not run it on flight hardware.
- **Why did radar miss more than the target?** Bolivia is a different flood from the calibration floods, and the 8-bit radar scores are compressed, so the calibration does not transfer cleanly. We report the achieved rate rather than the target.
- **Does the network beat a simple water index?** Slightly, in clear conditions (one tile out of 42 at the scaled pass). The larger gain comes from radar under cloud and from the dial.
- **Why only floods?** Floods are fully measured. Fire has real data and a feasibility memo, but nothing is measured, and radar for fire is untested.
