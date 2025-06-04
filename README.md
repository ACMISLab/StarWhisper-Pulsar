<h1 align="center"> <img src="https://github.com/ACMISLab/StarWhisper-Pulsar/blob/main/images/StarRipple.png" alt="StarWhisper-Pulsar Logo" style="width: 100px; height: auto; vertical-align: middle; margin-right: 5px;"> StarRipple </h1>

This is the official repository of **"StarRipple"**.

## 🆕 News
- \[**December 2024**\] Our paper **《Pulsar Candidate Classification with Multimodal Large Language Models》** has been accepted to NeurIPS 2024 Workshop Foundation Models for Science: Progress, Opportunities, and Challenges (FM4Science).
- \[**August 2024**\] We have released the first version 1.0 and are very excited to share our research and insights into Pulsar Candidate Classification!

## 🚀 Usage
We strongly recommend using SWIFT from the modelscope community to reproduce or use our model, which runs in a Python environment. Please ensure that your Python version is higher than 3.8.

- Install SWIFT using the pip command：

```shell
# all
pip install 'ms-swift[all]' -U
# only LLM
pip install 'ms-swift[llm]' -U
```

If you wish to reproduce the corresponding PICS-ResNet or HCCNN on the HTRU Medlat dataset, you may need to use the training and evaluation codes under the 'pics_ccnn_model' file. 

At the same time, you will need to install the corresponding PyTorch environment. If you wish to reproduce our feature selection process, please use the code under the 'feature_choose' file.

## Download papers and datasets
### V1 version
Paper link：[Pulsar Candidate Classification with Multimodal Large Language Models](https://openreview.net/pdf?id=8SKgWpZiDL)<br>
Overall framework:
![image](https://github.com/ACMISLab/StarWhisper-Pulsar/blob/main/images/framework.png)

Due to the large size of some weights, experimental data, and inference results, we have placed the complete experimental content on Hugging Face storage. It can be accessed through the following link:
1) Download the train and test dataset [train_and test dataset](https://huggingface.co/zfy1041264242/StarWhisper-Pulsar/tree/main/train_test_data)<br>
2) Download some of the best model weights mentioned in the paper. If you need any other model weights mentioned in the paper, please feel free to contact us. [some_best_result_mode](https://huggingface.co/zfy1041264242/StarWhisper-Pulsar/tree/main/some_best_result_mode)<br>
3) Download the experimental results records mentioned in the paper. [experiment_infer_result](https://huggingface.co/zfy1041264242/StarWhisper-Pulsar/tree/main/experiment_infer_result)<br>

## 💡 Prompt
Below are the prompts we use in traditional task.

<img src="https://github.com/ACMISLab/StarWhisper-Pulsar/blob/main/images/tra_instructions.png" alt="Traditional Task Instructions" style="width: 300px; height: auto; vertical-align: middle; margin-right: 5px;">

##  📖 Experiment Results
We fine-tuned all layers of the MLLM, optimizing both its textual and visual elements, to effectively address the multimodal demands of radio signal classification.  You can find detailed experimental information in the table below.

<div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
    <div style="flex: 0 0 48%; margin-bottom: 20px;">
        <img src="https://github.com/ACMISLab/StarWhisper-Pulsar/blob/main/images/ExperimentResults1.png" alt="ExperimentResults1" style="width: 80%; height: auto;">
    </div>
    <div style="flex: 0 0 48%; margin-bottom: 20px;">
        <img src="https://github.com/ACMISLab/StarWhisper-Pulsar/blob/main/images/ExperimentResults2.png" alt="ExperimentResults2" style="width: 80%; height: auto;">
    </div>
</div>


## 🤗 Citation
If you find the code and testset are useful in your research, please consider citing
```
@inproceedings{zhao2024pulsar,
  title={Pulsar Candidate Classification with Multimodal Large Language Models},
  author={Zhao, Fuyong and Li, Yuyang and Wang, Yanhao and Li, Hui and Chen, Mei and Chen, Panfeng and Sun, Ningchen and Wang, Cunshi and Liu, Jifeng},
  booktitle={Neurips 2024 Workshop Foundation Models for Science: Progress, Opportunities, and Challenges}
}
```
## 🤗 Contact us
Fuyong Zhao: gs.fyzhao24@gzu.edu.cn

Yuyang Li: liyuyang22@mails.ucas.ac.cn

## License
The dataset is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

