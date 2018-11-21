# Can you mock emoji?
We want to develop a game that makes use of the power of facial expression detection. The game is based on different levels. The goal of user is to pass as many levels as possible. In each level, user will try to mimic an emoji popup on the screen before timeout. The camera continuously captures user's face and dectects if the facial expression corresponds to the emoji. User is rewarded by points if he correctly minics the emoji in time.

## Team members
* Heng Zhang, zhan2614@purdue.edu
* Yajie Geng, geng18@purdue.edu

## Goals
* Use existing available deep neural network architectures to perform facial detection.
* Propose a new architecture that would perform better.
* Analyze accuracy vs speed of detection and optimize the network for real time detection.
* Develop a game interface for playing.

## Challenges
* For a game, the accuracy of facial detection is important. Because if the user correctly imitates the emoji but the CNN fails to detect it, that will gives wrong feedback to the user and the game experience is negatively affected.
* The game interface should be clear and simple and easy to learn

## References
* [some APIs](https://nordicapis.com/20-emotion-recognition-apis-that-will-leave-you-impressed-and-concerned/)
* [emotion dection](https://www.kaggle.com/c/emotion-detection-from-facial-expressions)
* [dataset1](http://vis-www.cs.umass.edu/lfw/#download)
* [dataset2](http://www.kasrl.org/jaffe.html)
* [dataset3](http://cvit.iiit.ac.in/projects/IMFDB/)
* [dataset4](http://vision.ucsd.edu/~iskwak/ExtYaleDatabase/ExtYaleB.html)
* [Existing proj1](https://github.com/atulapra/Emotion-detection)
