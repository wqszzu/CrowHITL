# CrowHITL: A Human-in-the-Loop Visual Interaction System for Spatial Crowdsourcing
Machine learning models, particularly reinforcement learning models (RLMs), have become increasingly crucial for a variety of Spatial Crowdsourcing (SC) platforms such as Uber, DiDi and Amazon. 
In this repository, we open source a system called CrowHITL, a novel human-in-the-loop visual interaction system for improving the learning efficiency and quality of RLMs. CrowHITL can enable users (e.g., model developers or data management experts) to manage and analyze large-scale spatiotemporal data from SC platforms, and interact with RLMs to guide their learning. CrowHITL consists of three core components: data manager, data visualizer, and human-computer interface. First, data manager allows users to easily collect, clean, and store spatiotemporal data from the SC platforms and learning data from their customized RLMs. Then, data visualizer offers three powerful visualization tools (i.e., worker visualization, task visualization, and RLMs visualization), which enable users to visually analyze the spatiotemporal information of workers and tasks, and the learning performance of RLMs in the SC platforms. Further, users can use the human-computer interface to convey their guidance to RLMs, helping RLMs achieve more efficient and high-quality learning. Overall, CrowHITL provides a user-friendly and intuitive solution for users to manage, analyze and guide RLMs in the SC platforms for their customized tasks. 

# Requirements
```
* PyTorch >= 1.0.0
* Python 3.6+
* Conda (suggested for building environment etc)
* tensorboardx==1.9
* tensorflow==1.14.0 (non-gpu version will do, only needed for tensorboard)
* matplotlib==3.3.4 （draw training results）
```

