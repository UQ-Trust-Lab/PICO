# PICO
## Folder Structure

```
├── README.md
├── LICENSE.txt
├── src
│   ├── P1
│   │	├── topic_clustering.py
│   ├── P2
│       ├── consistency_checking.ipynb
├── Dataset
├── Result
│   ├── P1
│   │	├── topic_clustering_reassigned_result.csv
└── ├── P2
        ├── consistency_checking_result


```
***Note:*** This tree includes only main files. 

## Description:

Below we describe each main file in our folder below. The two phases are detailed in Section 4 (Phase 1) and Section 5 (Phase 2).

### Phase 1 

```topic_clustering.py```: Run this file to obtain the full outputs of topic clustering. It should run on a linux server and under the enviroment of rapids (see the installation instructions from https://rapids.ai/). This program takes the input of skill's functional documents embeddings, the input dataset can be obtained from https://drive.google.com/file/d/14-UVQKRiLXQ6Cs9RTerp8XdSf2C3ujNk/view?usp=drive_link. After downloading the file, put it in the ```/Dataset``` folder.


### Phase 2

```consistency_checking.ipynb```: Conduct anomaly detection in each topic cluster. It takes the topic clustering results and the skill's requested data as input (see documented permissions examples in /Result/P1/topic_clustering_reassigned_result.csv) and finds the outliers in skill requested data. For requested data of privacy policy and runtime, we refer to the SKIPPER dataset https://github.com/UQ-Trust-Lab/SKIPPER/ and follow the same procedure.


