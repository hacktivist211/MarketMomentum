# MarketMomentum
Stock Price Prediction Using XGBoost
	Stock Market Prediction Using Machine Learning    

This project has been one of the most rewarding yet challenging experiences of my journey in machine learning. The idea started simple: build a model that could predict stock prices and reveal hidden patterns in the chaotic stock market. But as I got deeper into it, I realised the true complexity of the task. It wasn’t just about coding or choosing the right algorithm—it was about solving a series of interconnected problems, learning from mistakes, and growing at every step.

      

 The Journey    

  Starting Out    
The first step was setting up my machine—a process that turned out to be way harder than expected. From installing TensorFlow and NVIDIA drivers to dealing with cryptic CUDA errors like   “DLL load failed”  , I quickly found myself buried in debugging forums. Setting up the environment felt like solving a puzzle, but every solved issue brought a sense of triumph.  

  Handling the Data    
I had over     2613 CSV files    , each with 10,000 rows of data. Merging and cleaning such a huge dataset was a daunting task. Missing values, inconsistent formats, and memory errors taught me the importance of data wrangling. I spent a lot of time engineering features like     moving averages (SMA  200)     and     volume trends    —and honestly, seeing those features work well in the final model was one of the most satisfying moments.

  Building the Model    
Initially, I decided to try     Temporal Fusion Transformers (TFT)     because they seemed perfect for time  series data. But they turned out to be too resource  intensive and tricky to tune, so I pivoted to     LSTMs     before finally settling on     XGBoost    . Each algorithm brought its own challenges, but I learned so much about hyperparameter tuning, overfitting, and how even small tweaks can make a huge difference in performance.

      

  The Highs and Lows    

This project wasn’t without its share of struggles:
  Highs    :  
     Successfully achieving an accuracy of     97.45%     after countless iterations.
     Visualising the importance of features like `sma_200` and seeing the     Actual vs Predicted     plot closely align.  
     Overcoming obstacles like memory errors, complex debugging, and handling 26 million rows of data.

  Lows    :  
     Facing crashes when my system couldn’t handle the massive data size.  
     Spending hours on bugs that turned out to be simple fixes.  
     Watching models fail repeatedly during early training phases.  

Despite these struggles, the feeling of accomplishment after each breakthrough was unmatched. It’s incredible how much I’ve grown through this process—not just as a developer but as a problem  solver.

      

  What’s Inside?    

This repository contains:
1. Data Preparation Scripts    : Code to merge, clean, and preprocess large datasets.  
2. Feature Engineering    : Techniques for creating impactful features like moving averages and cyclical encodings.  
3. Model Training    : Implementation of XGBoost with detailed hyperparameter tuning.  
4. Visualisations    : Plots of feature importance and actual vs predicted values to showcase model performance.

      

   What I Learned    
This project taught me:
       Patience    : Debugging and training models require time and effort.  
       Flexibility    : Switching between algorithms and adapting to new challenges is essential.  
       Community Matters    : Forums, blogs, and open  source contributions were lifesavers.
