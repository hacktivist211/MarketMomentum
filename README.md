# MarketMomentum
Stock Price Prediction Using XGBoost		

# MarketMomentum - Stock Market Prediction Using Machine Learning

## Installation and Setup

Follow these steps to set up the stock market prediction model on your local system:

### Step 1: Install Anaconda

First, you'll need to install **Anaconda** if it isn't already installed on your system. Anaconda is a popular distribution for data science, machine learning, and deep learning workflows. It includes Python, package management, and environment management tools.

- **Download**: Go to the [Anaconda download page](https://www.anaconda.com/products/individual) and download the version that matches your operating system (Windows, macOS, or Linux).
- **Install**: Follow the installation instructions on the website.

### Step 2: Set Up a Virtual Environment

It's good practice to create a virtual environment for your project to isolate its dependencies from other Python projects.

- Open **Anaconda Prompt** (or your terminal if you're on macOS/Linux) and run the following command to create a new environment:
  
  ```bash
  conda create --name stock-prediction python=3.9
  ```

  This will create a new environment named `stock-prediction` with Python 3.9. You can specify a different version of Python if needed.

- To activate the environment, run:
  
  ```bash
  conda activate stock-prediction
  ```

### Step 3: Clone the Repository

Clone the repository to your local system so you can work with the code.

- Navigate to the directory where you want to clone the project and run:
  
  ```bash
  git clone https://github.com/hacktivist211/MarketMomentum.git
  ```

- After cloning, go to the project folder:

  ```bash
  cd MarketMomentum
  ```

### Step 4: Install Required Packages

Navigate to the project directory and install the required Python packages. You can create a `requirements.txt` file in your repository with the necessary dependencies.

For example, your `requirements.txt` file could look something like this:

```
pandas==1.3.3
numpy==1.21.2
matplotlib==3.4.3
xgboost==1.5.1
scikit-learn==0.24.2
seaborn==0.11.2
tensorflow==2.6.0
keras==2.6.0
```

To install these dependencies, run the following command in your terminal (make sure you're in the `MarketMomentum` directory):

```bash
pip install -r requirements.txt
```

Alternatively, you can install each package manually with:

```bash
pip install pandas numpy matplotlib xgboost scikit-learn seaborn tensorflow keras
```

### Step 5: Download the Dataset

Before running the model, you need the dataset. In this case, you’ll have a folder containing **CSV files** with stock data. 

- Place the dataset in the appropriate directory as per the project structure. The model will look for the data at this location.

### Step 6: Run the Model

Once the environment is set up and the packages are installed, you’re ready to run the model.

- Navigate to the Python script that contains your model (e.g., `train_model.py`).
- Run the script by executing the following in your terminal:

  ```bash
  python train_model.py
  ```

- The script will train the model on the stock market data, and you'll see logs of the training process. The model will output predictions that can be visualized as `Actual vs Predicted` graphs.

### Step 7: Visualize Results

Once the model completes the training, you’ll want to visualize the predictions. The script might generate graphs or save the results as images, which can be found in a directory like `output/`.

To view the graphs, you can use tools like **matplotlib** or **seaborn** to plot the predictions against the actual values.

### Step 8: Optional: Evaluate the Model

After training, you may want to evaluate the model’s performance by using various metrics like **accuracy**, **RMSE (Root Mean Squared Error)**, or **MAE (Mean Absolute Error)**. These are typically calculated in the evaluation phase of the code, but you can always add additional evaluation techniques based on your needs.
