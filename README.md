# NurValues Benchmark

Hugging Face: https://huggingface.co/datasets/Ben012345/NurValues

This **NurValues** benchmark aims to evaluate the alignment of nursing values in LLMs. It consists of two datasets of different difficulties, namely the Easy-Level dataset and the Hard-Level dataset.

This repository has 3 folders:

- ***Data*** folder: includes the NurValues dataset file _nursing_value_CN+EN.csv_, and the raw date _original_data.csv_.
- ***data_generate_code*** folder: indludes the code used to build the NurValues dataset.
- ***exp_code*** folder: Includes the code and prompts used for I/O, CoT, SC, and K-shot experiments. We only take the Llama-3.1 model and the Qwen-2.5 model as examples for the English and Chinese versions of the prompts, respectively.



## The NurValues dataset file (_nursing_value_CN+EN.csv_) includes:
- The columns for **Easy-Level dataset**:
  - **text_CN** column: The simple cases in Chinese.
  - **text_EN** column: The simple cases in English.
- The columns for **Hard-Level dataset**:
  - **complicated_text_CN** column: The generated complicated dialogue in Chinese.
  - **complicated_text_EN** column: The generated complicated dialogue in English
- **index** column: The index of the sample.
- **Nursing_Value** column: The related nursing value of the sample, including _Altruism_, _Human_Dignity_, _Integrity_, _Justice_, and _Professionalism_.
- **Alignment** column: Clarifies whether the nurse's behavior in the sample aligns with the corresponding nursing value.
  - 0: Not aligned
  - 1: Aligned

### Usage

```Python
import pandas as pd

df = pd.read_csv("nursing_value_CN+EN.csv")
```

# Article and Citation

[_**NurValues: Real-World Nursing Values Evaluation for Large Language Models in Clinical Context**_](https://arxiv.org/abs/2505.08734)

```
@misc{yao2025nurvaluesrealworldnursingvalues,
      title={NurValues: Real-World Nursing Values Evaluation for Large Language Models in Clinical Context}, 
      author={Ben Yao and Qiuchi Li and Yazhou Zhang and Siyu Yang and Bohan Zhang and Prayag Tiwari and Jing Qin},
      year={2025},
      eprint={2505.08734},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.08734}, 
}