# LATTE

This repository is the official implementation of Tabular Anomaly Detection vai Reconstruction with Attention-based Bottleneck. 

## Requirements

conda env create -f environment.yml

We use the publicly available ADBench and OODS dataset.



We use the publicly available ETT, Weather, Solar, ECL, and Traffic datasets exactly as described in the paper. You can download these datasets from the following repository: [https://github.com/thuml/iTransformer/tree/main](https://github.com/thuml/iTransformer/tree/main)
For detailed information about each dataset, please refer to Appendix A of the paper.

## Train and evaluate

To train and evaluate the model in the paper, we provide the scripts for all benckmarks under the folder `./scripts/`.  You can run the scripts using the following commands.
If you wish to customize any settings, we recommend modifying the corresponding files inside the `./scripts/` directory.

```
# ETT
bash ./scripts/ETT/TimePerceiver_ETTh1.sh
bash ./scripts/ETT/TimePerceiver_ETTh2.sh
bash ./scripts/ETT/TimePerceiver_ETTm1.sh
bash ./scripts/ETT/TimePerceiver_ETTm2.sh

# Weather
bash ./scripts/Weather/TimePerceiver.sh

# Solar
bash ./scripts/Solar/TimePerceiver.sh

# ECL
bash ./scripts/ECL/TimePerceiver.sh

# Traffic
bash ./scripts/Traffic/TimePerceiver.sh
```
접기














Cheolseok Kang
  오후 5:38
감사합니다!


Jaebin Lee
  오후 5:43