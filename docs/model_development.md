# Model Development

## Training Stages 

1. **Experiment**
- Iteratively select different models and assess their performance on the training data. The most promising models will be taken forward to the tuning stage.
2. **Tuning**
- The most promising models will them have their hyperparamters tuned using ramdonised cross validation search. 
3. **Training**
- The best model from the tuning stage is then trained on the training data.
4. **Training Evaluation**
- The trained model's performance is then evaluated using relevant metrics such as F1 score, precision, and recall.
4. **Offline Promotion**
- If the model passes the required threshold, then it will be promoted to the `Champion` model.
5. **Offline Inference**
- The `Champion` model is then used to generate predictions on unseen test data.
6. **Inference Evaluation**
- The inference results are then evaluated using F1 score, recall, and precision.
7. **Online Promotion**
- If the model passes the required threshold for production, then the model will be promoted into production.