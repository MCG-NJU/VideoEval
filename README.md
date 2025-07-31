# 🐼VideoEval

<div align="center">

<h2><a href="https://arxiv.org/abs/2407.06491">VideoEval: Comprehensive Benchmark Suite for Low-Cost Evaluation of Video Foundation Model</a></h2>

[Xinhao Li](https://scholar.google.com.hk/citations?user=evR3uR0AAAAJ&hl=zh-CN), Zhenpeng Huang, Jing Wang, [Kunchang Li](https://scholar.google.com/citations?user=D4tLSbsAAAAJ), and [Limin Wang](https://scholar.google.com/citations?user=HEuN8PcAAAAJ).

</div>

<img src="img/image-20240607232318559.png" alt="image-20240607232318559" style="zoom: 67%;" />

<!-- ## 📺 Task Example

<!-- ## :fire: Updates
- **2024/06/12**: Release annotations and evaluation codes of **VideoEval**, which includes VidTAB and VidEB. --> 

## 🛠️ Requirements and Installation

For VidTAB, we base on  [MMAction2](https://github.com/open-mmlab/mmaction2) for training and evaluation:

```cmd
pip install -U openmim
mim install mmengine 'mmcv>=2.0.0rc1'
mim install "mmdet>=3.0.0rc5"
mim install "mmpose>=1.0.0rc0"
git clone https://github.com/leexinhao/VideoEval.git
cd VidTAB
pip install -v -e .
```

## :bar_chart: Benchmark

### Data Preparation

Due to potential copyright issues, please refer to [DATA.md](https://github.com/leexinhao/VideoEval/blob/main/DATA.md) to download the original videos of each dataset separately, and we will share our version of the dataset after we confirm that there are no copyright issues. 

For VidTAB, you could directly use the [annotations](https://github.com/leexinhao/VideoEval/tree/main/VidTAB/annotations) we prepared.

### Video Task Adaptation Benchmark (VidTAB)

#### Few-Shot Evaluation

For training and evaluation, you could refer to [here](https://mmaction2.readthedocs.io/en/latest/user_guides/train_test.html), and we provide [configs](https://github.com/leexinhao/VideoEval/tree/main/VidTAB/configs) of diffenent VFMs for your reference.

In brief, you can use `tools/train.py` to train a model on a single machine with **a CPU and optionally a GPU** (Our experiment also uses only one GPU).

```bash
python tools/train.py ${CONFIG_FILE} [ARGS]
```

And I provide my train scripts `tools/my_train.sh` for avoiding setting [ARGS], then you could begin to use VidTAB by execute a bash file like this:

```bash
bash tools/my_train.sh configs/video_eval/AR_in_Dark/Internvideo2/frozen_tuning/InternVideo2-1B-stage1-pt_16_shot_bs16.py
bash tools/my_train.sh configs/video_eval/AR_in_Dark/Internvideo2/frozen_tuning/InternVideo2-1B-stage1_100_shot_bs16.py
bash tools/my_train.sh configs/video_eval/AR_in_Dark/Internvideo2/frozen_tuning/InternVideo2-1B-stage1-pt_100_shot_bs16.py
...
bash tools/my_train.sh configs/video_eval/Fake_face/ViCLIP/frozen_tuning/ViCLIP-200M_16_shot_bs16.py
bash tools/my_train.sh configs/video_eval/Fake_face/ViCLIP/frozen_tuning/ViCLIP-10M_100_shot_bs16.py
bash tools/my_train.sh configs/video_eval/Fake_face/ViCLIP/frozen_tuning/ViCLIP-10M_16_shot_bs16.py
bash tools/my_train.sh configs/video_eval/Fake_face/ViCLIP/frozen_tuning/ViCLIP-200M_100_shot_bs16.py
bash tools/my_train.sh configs/video_eval/Fake_face/ZeroI2V/linear_adapter0d125/ZeroI2V-CLIP-L_100_shot_bs16.py
```

Then you can go to the work dir to find the corresponding log file to see the result, In all our experiments, we conducted validation during the training process to select the epoch with the highest accuracy. Consequently, there was no need for additional performance testing after the training was completed. Furthermore, please note that we used **a single clip rather than three clips** to obtain the final performance metrics.

#### Zero-Shot Evaluation

**Prompts for  Zero-Shot Evaluation**: see [prompts for image backbones](https://github.com/leexinhao/VideoEval/blob/main/VidTAB_Zeroshot/img_prompt_gen.py), [prompts for video backbones](https://github.com/leexinhao/VideoEval/blob/main/VidTAB_Zeroshot/vid_prompt_gen.py).

```bash
bash exp/vid_zs.sh #for video language models
bash exp/img_zs.sh #for image language models
```

### Video Embed Benchmark (VidEB)

For evaluation, we provide [example](https://github.com/leexinhao/VideoEval/blob/main/VidEB/example.ipynb) as a demonstration of the pipeline of embedding extraction and evaluation.

## 🧰 How to build VideoEval?

### Collecting diverse dataset from public source

We conducted a months-long investigation, integrating insights from multiple researchers and engineers in the field of video understanding, and selected evaluation tasks by leveraging available high-quality public video data. Our selection was guided by two core principles:

1. The selected tasks are **meaningful real-world tasks**; the practical application value of these tasks ensures the validity of the capability dimensions we evaluate.
2. The selected tasks are often **unaddressed in previous evaluations**, thereby ensuring our assessment of the out-of-domain generalization of Video Foundation Models (VFMs).
We elaborate below on the application value and evaluative significance of each scenario in our assessment:

- **Action**: While numerous previous benchmarks have evaluated action recognition tasks, we still include it as part of our assessment given its status as one of the most critical video understanding tasks. We focus on specialized scenarios that have received relatively little attention in traditional action recognition tasks, specifically selecting action recognition in dark scenes and long-video scenarios. This aims to examine the action recognition performance of VFMs in relatively out-of-domain test scenarios (as most training data for existing VFMs consists of short videos in relatively bright scenes).

- **Science**: "AI for Science" has emerged as a topic of growing interest in recent years but has often been overlooked in previous video understanding benchmarks. We chose animal and medical video understanding for evaluation in this domain, with specific application scenarios including animal behavior monitoring, behavioral surveillance during medical surgeries, and safety inspections.

- **Safety**: Using AI for video content moderation is a key application of video understanding, encompassing tasks such as detection of AI-generated content and detection of harmful information. These tasks require models to possess more fine-grained video understanding capabilities.

- **Quality**: Low-level evaluation of video quality has also been generally neglected in previous VFM assessments. We argue that a sufficiently general VFM representation should not only support high-level semantic understanding but also enable low-level visual understanding.
Emotion: Understanding human emotions is a crucial capability for AI models. Human emotions are complex and often require capturing microexpressions and subtle movements for accurate recognition, making this an important application scenario for VFMs.

### Constructing the adaptation task based on the existing annotations
#### 1. Remove Low Quality Video Datasets

We manually exclude datasets with videos that have low resolution (below 240p), low frame rate (below 15fps), insufficient quantity (fewer than 150 videos per category), or low annotation accuracy (below 90%).

- We use [remove_low_quality_video.py](tools/remove_low_quality_video.py) to filter low quailty videos.
- We manually filter out datasets that lack sufficient high-quality videos, and at the same time, conduct partial sampling on the initially selected video datasets to check their annotation quality (with a human accuracy rate exceeding 90%).

#### 2. Select Discriminative Tasks

 For task difficulty screening, we first evaluate zero-shot classification performance using [VidTAB_Zeroshot](VidTAB_Zeroshot). We then classify samples as follows: 
 - Easy: Samples that are correctly classified by three or more models. Spatial: Samples that are correctly classified by both CLIP and EVA
 - Temporal: Samples that are correctly classified by at least one of ViCLIP or Internvideo2-1B, but not by CLIP and EVA. 
 - Hard: Samples that are incorrectly classified by all models. We use the zero-shot classification accuracy of the models and the aforementioned proportions as references for task selection. Based on this, we choose tasks with lower zero-shot classification accuracy, higher proportions of Hard and Temporal samples, and lower proportions of Easy samples

We also provide [evaluation_with_Gemini2.5](tools/evaluation_with_Gemini2.5.py) to use Gemini2.5-Pro/Flash to evaluate the datasets.

#### 3. Control the Number of Categories

For datasets that originally include category labels, such as ARID and Animal Kingdom, we select categories with sufficient samples to ensure evaluation accuracy and stability. We also control the final number of categories to avoid making the adaptation task overly difficult. We observed that both zero-shot testing and few-shot experiments based on current VFMs show that when the number of categories is too high, models often perform no better than random guessing. Although this issue may be mitigated as VFMs improve, we currently need to control the number of categories to effectively showcase differences between models. We select the main categories for each task and limit the number of categories to around 10 (based on few-shot experiments).

- We use [mmaction2](VidTAB) to conduct few-shot experiments to determine the number of categories. We control the number of categories so that the current strongest VFMs can achieve a TA-score higher than random. Specifically, we remove overly detailed category branches and retain relatively coarse-grained key categories.

#### 4. Handling Multi-label and Regression Tasks

For datasets that are not originally classification tasks, we transform the tasks into classification tasks. For example, for DOVER, which is used for video aesthetics and technical quality assessment (a regression task), we assume that videos with quality scores in the top 40% are "high-quality videos" and those with scores in the bottom 40% are "low-quality videos", thus converting the original task into a binary classification task.

- We use [convert_to_classify.py](tools/convert_to_classify.py) to convert regression task to classification task.

### Determining the evaluation protocol
 We use [mmaction2](VidTAB) to conduct few-shot experiments to determine the evaluation protocol in `VidTAB\configs\video_eval`.
### Identifying efficient adaptation method for evaluation
 We use [mmaction2](VidTAB) to conduct few-shot experiments to identify efficient adaptation method (e.g. `VidTAB\mmaction\models\backbones\v_jepa.py`).



## :dizzy: Acknowledgement

Thanks to the open source of the following projects: [ARID](https://xuyu0010.github.io/arid.html), [Breakfast](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/), [Animal Kingdom](https://github.com/sutdcv/Animal-Kingdom),  [SurgicalActions160](http://ftp.itec.aau.at/datasets/SurgicalActions160/index.html), [FaceForensics++](https://github.com/ondyari/FaceForensics), [MOB](https://github.com/syedhammadahmed/mob), [DOVER](https://github.com/VQAssessment/DOVER), [CAER](https://caer-dataset.github.io/), [vsc2022](https://github.com/facebookresearch/vsc2022/tree/main), [FIVR-200K](https://github.com/MKLab-ITI/FIVR-200K), [Ask-Anything](https://github.com/OpenGVLab/Ask-Anything/tree/main), [UMT](https://github.com/OpenGVLab/unmasked_teacher), [EVA](https://github.com/baaivision/EVA/tree/master), [InternVideo](https://github.com/OpenGVLab/InternVideo/tree/main?tab=readme-ov-file), [SigLIP](https://github.com/google-research/big_vision), [CLIP](https://github.com/openai/CLIP), [jepa](https://github.com/facebookresearch/jepa), [dinov2](https://github.com/facebookresearch/dinov2), [VideoMAE](https://github.com/MCG-NJU/VideoMAE), [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2), [MMAction2](https://github.com/open-mmlab/mmaction2).

## :page_facing_up: Citation

If you find this project useful in your research, please consider cite:
```BibTeX
@article{li2024videoeval,
  title={VideoEval: Comprehensive Benchmark Suite for Low-Cost Evaluation of Video Foundation Model},
  author={Li, Xinhao and Huang, Zhenpeng and Wang, Jing and Li, Kunchang and Wang, Limin},
  journal={arXiv preprint arXiv:2407.06491},
  year={2024}
}
```
