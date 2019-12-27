# Words2vec
wordstovec report 
Word2vec 

The Word2vec algorithm has two different implementations: Skip-Gram and Continuous Bag of Words (CBOW). They are completely opposite in terms of their implementation. Word2vec takes a large corpus of text as input and produces a vector space (feature space), typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space.

Skip-Gram
Dataset and Structure
Skip-gram attempts to predict the context words from an input word. Input is target word and output(s) are context words.

Figure 1 Skip gram architecture (Source: https://arxiv.org/pdf/1301.3781.pdf Mikolov el al.)
As we observe in the figure of the Skip-Gram model w(t) is the target word for which we require context in the form of w(t-2), w(t-1), w(t+1) and w(t+2). 

For example, in the above figure we can see how context-target word pairs are made to be used for the training of Skip-Gram model. We use a window_size parameter (window_size is 2 in this case) which looks to the left and right of the context word for as many as window_size(=2) words.
Given a specific word in the middle of a sentence (the input word), look at the words nearby and pick one at random. The network is going to tell us the probability for every word in our vocabulary of being the “nearby word” that we chose.
The network is going to learn the statistics from the number of times each pairing shows up. So, for example, the network is probably going to get many more training samples of (“Soviet”, “Union”) than it is of (“Soviet”, “Sasquatch”). When the training is finished, if you give it the word “Soviet” as input, then it will output a much higher probability for “Union” or “Russia” than it will for “Sasquatch”.

Computation Model
Text cannot be directly fed to a neural network, to combat this issue we first form a vocabulary of words from our training samples. Suppose we have a vocabulary of 10,000 different words. We will represent an input word like "ants" as a one-hot vector. This vector will have 10,000 parts (one for each word in our vocabulary) and we'll put a "1" in the position relating to "ants", and 0s in the rest of the different positions.

There is no activation function on the hidden layer neurons, but the output neurons use softmax. When training this network on word pairs, the input is a one-hot vector representing the input word and the training output is also a one-hot vector representing the output word. But when you evaluate the trained network on an input word, the output vector will actually be a probability distribution (i.e., a bunch of floating-point values, not a one-hot vector).
Each of which are passed to an embedding layer (initialized with random weights) of its own. Once we obtain the word embeddings for the target and the context word, we pass it to a merge layer where we compute the dot product of these two vectors. Then we pass on this dot product value to a dense sigmoid layer which predicts either a 1 or a 0 depending on if the pair of words are contextually relevant or just random words (Y’). We match this with the actual relevance label (Y), compute the loss by leveraging the mean_squared_error loss and perform backpropagation with each epoch to update the embedding layer in the process. 
Softmax is a multinomial regression classifier. It means that it classifies multiple labels, such as predicting if an hand-written digit is 0,1,2,...8 or 9. In case of binary classification (True or False), such as classifying fraud or not-fraud in bank transactions, binomial regression classifier called Sigmoid function is used
We are using tf.nn.sampled_softmax_loss . This is a faster way to train a softmax classifier over a huge number of classes. This operation is for training only. It is generally an underestimate of the full softmax loss. 


Training Session
We run the algorithm for 10,000 epochs and the final average loss is 3.367473.


Each training example consisting of one input target word having a unique numeric identifier and one context word having a unique numeric identifier. If it is a positive sample the word has contextual meaning, is a context wordand our label Y=1, else if it is a negative sample, the word has no contextual meaning, is just a random word and our label Y=0. We will pass each of them to an embedding layer of their own, having size (vocab_size x embed_size) which will give us dense word embeddings for each of these two words (1 x embed_size for each word). Next up we use a merge layer to compute the dot product of these two embeddings and get the dot product value. This is then sent to the dense sigmoid layer which outputs either a 1 or 0. We compare this with the actual label Y (1 or 0), compute the loss, backpropagate the errors to adjust the weights (in the embedding layer) and repeat this process for all (target, context) pairs for multiple epochs.

Factors affecting efficiency
The size of the training corpus (T) is very large thus updating 3M neurons for each training sample is unrealistic in terms of computational efficiency. Negative sampling addresses this issue by updating only a small fraction of the output weight neurons for each training sample. 
In negative sampling, K negative samples are randomly drawn from a noise distribution. K is a hyper-parameter that can be empirically tuned, with a typical range of [5,20]. For each training sample (positive pair: w and cpos), you randomly draw K number of negative samples from a noise distribution Pn(w), and the model will update (K+1)×N neurons in the output weight matrix (Woutput). N is the dimension of the hidden layer (h), or the size of a word vector. +1 accounts for a positive sample.
With the above assumption, if you set K=9, the model will update (9+1)×300=3000 neurons, which is only 0.1% of the 3M neurons in Woutput. This is computationally much cheaper than the original Skip-Gram, and yet maintains a good quality of word vectors.

The below figure has 3-dimensional hidden layer (N=3), 11 vocabs (V=11), and 3 negative samples (K=3).




CBOW
Dataset and Structure
The CBOW model architecture tries to predict the current target word (the center word) based on the source context words (surrounding words).

Considering a simple sentence, “the quick brown fox jumps over the lazy dog”, this can be pairs of (context_window, target_word) where if we consider a context window of size 2, we have examples like ([quick, fox], brown), ([the, brown], quick), ([the, dog], lazy) and so on. Thus the model tries to predict the target_word based on the context_window words.
We can model this CBOW architecture now as a deep learning classification model such that we take in the context words as our input, X and try to predict the target word, Y. In fact building this architecture is simpler than the skip-gram model where we try to predict a whole bunch of context words from a source target word.
First corpus vocabulary is built, where we extract out each unique word from our vocabulary and map a unique numeric identifier to it.

We need pairs which consist of a target centre word and surround context words. In our implementation, a target word is of length 1 and surrounding context is of length 2 x window_size  where we take window_size words before and after the target word in our corpus. 

Computation model
The loss calculated is propagated to the softmax layer and embedding layer of each word. We have input context words of dimensions (2 x window_size), we will pass them to an embedding layer of size (vocab_size x embed_size) which will give us dense word embeddings for each of these context words (1 x embed_size for each word). Next up we use a lambda layer to average out these embeddings and get an average dense embedding (1 x embed_size) which is sent to the dense softmax layer which outputs the most likely target word. We compare this with the actual target word, compute the loss, backpropagate the errors to adjust the weights (in the embedding layer) and repeat this process for all (context, target) pairs for multiple epochs. The following figure tries to explain the same.

We are now ready to train this model on our corpus using our data generator to feed in (context, target_word) pairs.

We are using tf.nn.sampled_softmax_loss . This is a faster way to train a softmax classifier over a huge number of classes. This operation is for training only. It is generally an underestimate of the full softmax loss. 
Loss calculation is done using the method below.

Average of 2000 epochs is taken and loss is displayed every 2000 epochs.
Training Session
We run the algorithm for 10,000 epochs and the final average loss is 2.79.

The input layer and the target, both are one- hot encoded of size [1 X V]. There are two sets of weights. one is between the input and the hidden layer and second between hidden and output layer.
Input-Hidden layer matrix size =[V X N] , hidden-Output layer matrix  size =[N X V] : Where N is the number of dimensions we choose to represent our word in. It is arbitrary and a hyper-parameter for a Neural Network. Also, N is the number of neurons in the hidden layer. Here, N=4.There is a no activation function between any layers.. The input is multiplied by the input-hidden weights and called hidden activation. It is simply the corresponding row in the input-hidden matrix copied. The hidden input gets multiplied by hidden- output weights and output is calculated. Error between output and target is calculated and propagated back to re-adjust the weights. The weight between the hidden layer and the output layer is taken as the word vector representation of the word.
Factors affecting efficiency
The objective function in CBOW it is negative log likelihood of a word given a set of context i.e -log(p(wo/wi)), where p(wo/wi) is given as

The gradient of error with respect to hidden-output weights and input-hidden weights are calculated using backpropagation and CBOW has linear activations. Being probabilistic is nature, it is supposed to perform superior to deterministic methods(generally). It requires low amount of memory. It does not need to have huge RAM requirements like that of co-occurrence matrix where it needs to store three huge matrices.

The loss initially and finally is much lower than Skip-Gram, though training a CBOW from scratch can take forever if not properly optimized.
 





