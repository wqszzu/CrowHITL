# CrowHITL: A Human-in-the-Loop Visual Interaction System for Reinforcement Learning in Spatial Crowdsourcing  

**CrowHITL** is an open-source system designed to enhance the learning efficiency and quality of **reinforcement learning models (RLMs)** in **spatial crowdsourcing (SC) platforms** like Uber, DiDi, and Amazon.  

## 🚀 Features  
- **Data Management**: Efficiently collect, clean, and store spatiotemporal data from SC platforms and learning data from RLMs.  
- **Interactive Visualization**: Gain insights through three powerful visualization tools:  
  - 📍 **Worker Visualization**: Analyze spatiotemporal patterns of workers.  
  - 🎯 **Task Visualization**: Explore task distributions and trends.  
  - 📊 **RLMs Visualization**: Monitor and evaluate model learning performance.  
- **Human-in-the-Loop Interaction**: Guide RLMs via an intuitive interface, enabling **better model adaptation and optimization**.  

## 📌 Why CrowHITL?  
✔ **User-Friendly**: Designed for model developers and data management experts.  
✔ **Scalable**: Supports large-scale spatiotemporal datasets.  
✔ **Efficient**: Enhances model learning through human feedback and interactive analysis.  

## 🔧 Getting Started  

### 📋 Requirements  
Ensure you have the following dependencies installed:  

```bash
dash==1.0.2  
dash-bootstrap-components==1.2.1  
dash-daq==0.1.7  
gunicorn>=19.9.0  
numpy>=1.16.2  
pandas>=0.24.2
```

## 📥 Installation  

Follow these steps to install **CrowHITL** on your local machine:  

### 1️⃣ 🛠 Clone the Repository  
First, clone the repository from GitHub and navigate to the project directory:  

```bash
git clone https://github.com/your-repo/CrowHITL.git
cd CrowHITL
```

### 2️⃣ 📦 Install Dependencies  

Before running **CrowHITL**, you need to install the required dependencies.  

#### 🛠 Install via `pip`  
Run the following command in the project directory:  

```bash
pip install -r requirements.txt
```

### 🚀 Running the System  

Once the dependencies are installed, follow these steps to run **CrowHITL**:  

#### 1️⃣ ▶️ Start the System  
To launch **CrowHITL**, run the following command:  

```bash
python CrowHITL_main.py
```

#### 2️⃣ 🌐 Access the Interface  

After starting **CrowHITL**, open your web browser and navigate to the following URL:

```bash
http://127.0.0.1:8050/
```


This will open the **CrowHITL** interface, allowing you to interact with the system in real-time. You can use the interface to manage and visualize spatiotemporal data, as well as guide your RLMs.  





