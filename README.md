# Social Media Mining for Health Applications Shared Task

Will be updated soon with the code. Unfortunately, we are not permitted to release the datasets, but they will soon appear in a Codalab competition where people can try their models' performance. The model weights have been uploaded.

The relevant paper can be found [here](https://aclanthology.coli.uni-saarland.de/papers/W18-5910/w18-5910). Below you can find the `.bib` entry for the paper.

```
@InProceedings{W18-5910,
  author = 	"Xherija, Orest",
  title = 	"Classification of Medication-Related Tweets Using Stacked Bidirectional LSTMs with Context-Aware Attention",
  booktitle = 	"Proceedings of the 2018 EMNLP Workshop SMM4H: The 3rd Social Media Mining for Health Applications Workshop and Shared Task",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"38--42",
  location = 	"Brussels, Belgium",
  url = 	"http://aclweb.org/anthology/W18-5910"
}
```

## Sub-task 1 Leaderboard (11 participants)

Distinguish tweets that mention names of medications or dietary supplements from those that do not. The definitions of drugs and dietary supplements is taken from the FDA.

|                  | Precision | Recall    |  F1       | 
| :---             |   :---:   |   :---:   | :---:     |
| `THU_NGN`        | 0.933     | 0.904     | **0.918** |
| `UChicagoCompLx` | **0.937** | 0.891     | 0.914     |
| `IRISA`          | 0.922     | 0.906     | 0.914     |
| `Tub-Oslo`       | 0.917     | **0.907** | 0.912     |
| `CIC-NLP`        | 0.920     | 0.899     | 0.910     |
| `UZH`            | 0.927     | 0.878     | 0.902     |
| `Techno`         | 0.905     | 0.855     | 0.879     |
| `IIT_KGP`        | 0.918     | 0.840     | 0.877     |
| `LILU`           | 0.841     | 0.860     | 0.850     |
| `ART`            | 0.785     | 0.880     | 0.830     |
| `ClaC`           | 0.788     | 0.769     | 0.778     |

## Sub-task 2 Leaderboard (8 participants)

Distinguish tweets that mention personal medication intake, possible medication intake or no intake.

|                  | Precision | Recall    |  F1       | 
| :---             |   :---:   |   :---:   | :---:     |
| `UChicagoCompLx` | **0.654** | **0.783** | **0.713** |
| `Light`          | 0.492 	   | 0.467 	   | 0.479     |
| `Tub-Oslo`       | 0.464	 	 | 0.466	   | 0.465     |
| `IRISA`          | 0.434     | 0.501	   | 0.465     |
| `IIT_KGP`        | 0.408     | 0.407	   | 0.408     |
| `UZH`            | 0.371	   | 0.437     | 0.401     |
| `CLaC`           | 0.402	   | 0.366     | 0.383     |
| `Techno`         | 0.327	   | 0.432     | 0.372     |

## Sub-task 3 Leaderboard (9 participants)

Distinguish tweets that contain mentions of adverse drug reaction those that do not.

|                  | Precision | Recall    |  F1       | 
| :---             |   :---:   |  :---:    | :---:     |
| `THU_NGN`        | 0.442 	   | 0.636 	   | **0.522** |
| `IRISA`          | 0.378	   | **0.649** | 0.478     |
| `UZH`            | 0.455		 | 0.436     | 0.445     |
| `Tub-Oslo`       | **0.638** | 0.317	   | 0.424     |
| `ART`            | 0.332	   | 0.547     | 0.413     |
| `UChicagoCompLx` | 0.370	 	 | 0.464	   | 0.411     |
| `CIC-NLP`        | 0.314		 | 0.529     | 0.394     |
| `Techno`         | 0.434		 | 0.344	   | 0.383     |
| `IIT_KGP`        | 0.189     | 0.643     | 0.292     |

## Sub-task 4 Leaderboard (9 participants)

Distinguish tweets that mention behavior related to influenza vaccination from those that do not. Data annotators labeled tweets to answer the binary question _Does this message indicate that someone received, or intended to receive, a flu vaccine?_

|                  | Precision | Recall    |  F1       | 
| :---             |   :---:   |  :---:    | :---:     |
| `CARRDS`         | **0.918** | 0.859     | **0.887** |
| `Techno`         | 0.870     | 0.859     | 0.865     |
| `Light`          | 0.824     | 0.897     | 0.859     |
| `Tub-Oslo`       | 0.840     | 0.872     | 0.855     |   
| `UChicagoCompLx` | 0.791     | **0.923** | 0.852     |
| `IRISA`          | 0.867     | 0.833     | 0.850     |
| `LILU`           | 0.829     | 0.808     | 0.818     |
| `ClaC`           | 0.700     | 0.897     | 0.787     |
| `IIT_KGP`        | 0.800     | 0.769     | 0.784     |
