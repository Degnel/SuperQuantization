# @inproceedings{Huang2024OpenCoderTO,
#   title = {OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models},
#   author = {Siming Huang and Tianhao Cheng and Jason Klein Liu and Jiaran Hao and Liuyihan Song and Yang Xu and J. Yang and J. H. Liu and Chenchen Zhang and Linzheng Chai and Ruifeng Yuan and Zhaoxiang Zhang and Jie Fu and Qian Liu and Ge Zhang and Zili Wang and Yuan Qi and Yinghui Xu and Wei Chu},
#   year = {2024},
#   url = {https://arxiv.org/pdf/2411.04905}
# }

from datasets import load_dataset

educational_instruct = load_dataset(
    "OpenCoder-LLM/opc-sft-stage2", "educational_instruct"
)
evol_instruct = load_dataset("OpenCoder-LLM/opc-sft-stage2", "evol_instruct")
mceval_instruct = load_dataset("OpenCoder-LLM/opc-sft-stage2", "mceval_instruct")
package_instruct = load_dataset("OpenCoder-LLM/opc-sft-stage2", "package_instruct")
