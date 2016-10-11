# Shakespearebot_9000

Shakespeare Bot 9000 is a Shakespearian sonnet generator based on training hidden Markov Models. Unsupervised training is performed by the EM procedure and sonnet generation is performed using the Viterbi algorithm.

Additional constraints are imposed to ensure propery rhyme scheme (abab cdcd efef) as well as proper meter when possible. Basic parts of sentence (POS) tagging helps ensure reasonable sentence structure.

### Required Packages

- Numpy:
  - `sudo pip install numpy`
- NLTK and PyHyphen (for NLP):
  - `sudo pip install pyhyphen`
  - `sudo pip install nltk`

### How To Run

`python hmm.py <number of sonnets to generate>`

##Sample sonnet:

**Eye of the Beholder**

Eye that spoil and my checked thou her treasure;
Die dreams the next death to me memory,
Perforce where there and disgrace in pleasure,
But outward and shall hidden spheres who sky,
All to sue extremity slave unfathered,
Them woman that thou to some bear around,
Yea but dulling use me and my gathered.
Are I and soul a game of coals of ground,
Thou to like music am the number taste,
Made esteem and this sight do shall in side,
Tanned offender's there that thoughts a should the last,
Those breath an nor losing and raven hide,
     Doubt storm-beaten report you thou is staineth,
     She and his hell the proud loud they disdaineth.
