# Identifying Ironic Content Spreaders on Twitter using Psychometrics, Contextual and Ironic Features with Gradient Boosting Classifier

The study of irony detection on social networks has gained much attention in recent years. As part of this task, a collection of users’ tweets is provided, and the goal is to determine if these users are spreaders of irony or not. As we hypothesized that user-generated content is affected by the author’s psychometric characteristics, contextual information, and irony features in the text, we investigated the effects of incorporating this information to identify ironic spreaders. Using the emotion and personality detection module, we were able to extract the author’s psychometric features. A pre-trained language model based on SBERT and T5-based architecture has been employed to extract context-based features. Our paper describes a framework using the author’s psychometric, contextual, and ironic features in a Gradient Boosting classifier based on our theory. Experimental results in this paper demonstrate the importance of this combination in identifying ironic spreader users. As a result, we were able to achieve an accuracy of 95.00% and 93.81% with 5-fold and 10-fold cross-validation respectively on the IROSTEREO training dataset. However, on the official PAN test set, our system attained a 88.89% score.

##### Proposed Method
![Screenshot from 2023-02-22 16-20-34](https://user-images.githubusercontent.com/86873813/220624599-1d66ce5f-8e6e-4a60-997b-33d6cbd1cb9d.png)

##### Personality Embedding Module

![Screenshot from 2023-02-22 16-20-22](https://user-images.githubusercontent.com/86873813/220624619-258d6cbf-b8d1-46f5-acaf-155f118377cb.png)




# Usages
#### Train Model:

> tweet_level_trainer.py
> ouser_level_trainer.py

#### Test Model

> python inferencer.py

