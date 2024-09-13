# CrowHITL: A Human-in-the-Loop Visual Interaction System for Spatial Crowdsourcing
Machine learning models, particularly reinforcement learning models (RLMs), have become increasingly crucial for a variety of Spatial Crowdsourcing (SC) platforms such as Uber, DiDi and Amazon. 
In this repository, we open source a system called CrowHITL, a novel human-in-the-loop visual interaction system for improving the learning efficiency and quality of RLMs. CrowHITL can enable users (e.g., model developers or data management experts) to manage and analyze large-scale spatiotemporal data from SC platforms, and interact with RLMs to guide their learning. CrowHITL consists of three core components: data manager, data visualizer, and human-computer interface. First, data manager allows users to easily collect, clean, and store spatiotemporal data from the SC platforms and learning data from their customized RLMs. Then, data visualizer offers three powerful visualization tools (i.e., worker visualization, task visualization, and RLMs visualization), which enable users to visually analyze the spatiotemporal information of workers and tasks, and the learning performance of RLMs in the SC platforms. Further, users can use the human-computer interface to convey their guidance to RLMs, helping RLMs achieve more efficient and high-quality learning. Overall, CrowHITL provides a user-friendly and intuitive solution for users to manage, analyze and guide RLMs in the SC platforms for their customized tasks. The demonstration video of CrowHITL has been uploaded to [Google Drive](https://drive.google.com/file/d/1ZNRNR-qG-au1A4wThBnMIS1LEqtfRBKk/view).


# Requirements
```
* dash==1.0.2
* dash-bootstrap-components==1.2.1
* dash-daq==0.1.7
* gunicorn>=19.9.0
* numpy>=1.16.2
* pandas>=0.24.2
```

