# MP2 Reflection

## What worked

The retrieval pipeline worked best when the question had a strong match to the relevant source file and a small set of expected facts. In the validation results, the questions where the source and expected file aligned clearly produced higher fact-match scores, such as q1_red_headed_league_assistant (1/5) and q2_speckled_band_murder_weapon (3/6). The source-match accuracy was good overall, and the retrieval system correctly identified the expected source files for all validated questions.

## What didn't work

The fact-match score did not improve much even when I increased the temperature or adjusted prompt behavior for some questions. In particular, some questions still had low match rates for example id=q4.. because the system was retrieving a relevant file but not enough of the key supporting facts. This suggests the retrieval step was often identifying the right story but not consistently selecting the best evidence sentences.

## What I'd change

I would test more combinations of expected facts, including both relevant and irrelevant facts, to better measure whether the fact-match score is driven by retrieval quality or by the validation logic. I would also compare a hybrid retrieval approach with reranking.

## One surprise

One surprise was that the match rate could differ even when the expected facts and the source file were aligned. In a few cases, the system cited a different file name or returned results from a neighboring story for example id=q5.., which suggests the retrieval signal may be using some overlap in entity names or themes rather than only the strongest document-level match. This makes the importance of better filtering and reranking more obvious.

# Output of '$python mp2_rag.py validate'

```text
  Validating 2 questions from predefined_questions.jsonl…

  ✓ q1_red_headed_league_assistant
      Q: Who was the assistant working at Jabez Wilson's pawnshop, and what was his real identity?
      Cited: 01_red_headed_league.txt
      Expected: 01_red_headed_league.txt
      Facts matched: 1/5
      Latency: 2717ms

  ✓ q2_speckled_band_murder_weapon
      Q: What killed Julia Stoner at Stoke Moran, and how was the method discovered?
      Cited: 02_speckled_band.txt
      Expected: 02_speckled_band.txt
      Facts matched: 3/6
      Latency: 2328ms

  Source-match: 2/2

  Validating 3 questions from learner_questions.jsonl…

  ✓ q3_red_headed_league_interviewer
      Q: Who is the interviewer to fill the vacancy in the red headed league?
      Cited: 01_red_headed_league.txt
      Expected: 01_red_headed_league.txt
      Facts matched: 2/2
      Latency: 1813ms

  ✓ q4_red_coining_operation
      Q: How long Holmes took to solve the puzzle and when in engineers thumb?
      Cited: 04_engineers_thumb.txt
      Expected: 04_engineers_thumb.txt
      Facts matched: 2/4
      Latency: 1781ms

  ✓ q5_hatherley_arrivat_at_house
      Q: What is the house name Hatherley took the last train at the night from Reading?
      Cited: 04_engineers_thumb.txt, 05_scandal_in_bohemia.txt
      Expected: 04_engineers_thumb.txt
      Facts matched: 2/3
      Latency: 1734ms

  Source-match: 3/3
```
