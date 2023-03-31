# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Predicting Parts-of-Speech with an LSTM
#
# Let's preview the end result. We want to take a sentence and output the part-of-speech for each word in that sentence. Something like this:
#
# **Code**
#
# ```python
# new_sentence = "I is a teeth"
#
# ... # Preprocessing the sentence
#
# # Acting on the preprocessed sentence
# predictions = model(word_indices)
#
# ... # Formatting the output
# ```
#
# **Output**
#
# ```text
# I     => Noun
# is    => Verb
# a     => Determiner
# teeth => Noun
# ```

# %% [markdown]
# ## Tasks
#
# 1. Add two additional sentences (for every member of your group) to the [list on this google sheet](https://docs.google.com/spreadsheets/d/1HJmlehaYhGWclDo1t0k6i1VHxN15zr8ZmJj7Rf_VEaI/edit#gid=1031300490). You can thank previous semesters for the existing dataset below.
#
# 1. **Do not run all cells in the notebook.** You will need to make some predictions prior to running cells. Read through the notebook as a group, stopping and answering each question on gradescope as you go.
#
# 1. After you work through the notebook once, you should try to improve accuracy by changing hyperparameters (including the network parameters and network architecture--extending the classification part of the network might be a good idea).
#
# 1. (Optional) Try changing the model out for a
#     + fully connected network,
#     + convolutional neural network, or
#     + transformer.

# %% [markdown]
# ## Imports

# %%
from random import shuffle

import torch
from torchsummary import summary

from fastprogress.fastprogress import progress_bar, master_bar

import matplotlib.pyplot as plt
from jupyterthemes import jtplot

jtplot.style(context="talk")

# %% [markdown]
# ## Data Processing

# %% [markdown]
# ### Raw dataset
#
# - Add two sentences and corresponding parts of speech **per group member**
# - You can use this utility for double checking your parts of speech: https://parts-of-speech.info/
# - I will put them into the notebook (you will need to pull the updates)
# - Do not include any punctuation
# - Your sentences must only include nouns, verbs, and determiners
#     + N for noun
#     + V for verb
#     + D for determiner
# - You can mark pronouns as nouns
#
# You can find an exmaples in the [spreadsheet](https://docs.google.com/spreadsheets/d/1HJmlehaYhGWclDo1t0k6i1VHxN15zr8ZmJj7Rf_VEaI/edit#gid=1031300490).

# %%
# I will update this once everyone has submitted their sentences in the sheet.
# You will then need to pull the latest version of this notebook.

raw_dataset = [
    ("The dog ate the apple", "D N V D N"),
    ("Everybody read that book", "N V D N"),
    ("Trapp is sleeping", "N V V"),
    ("Everybody ate the apple", "N V D N"),
    ("Cats are good", "N V D"),
    ("Dogs are not as good as cats", "N V D D D D N"),
    ("Dogs eat dog food", "N V N N"),
    ("Watermelon is the best food", "N V D D N"),
    ("I want a milkshake right now", "N V D N D D"),
    ("I have too much homework", "N V D D N"),
    ("Zoom won't work", "N D V"),
    ("Pie also sounds good", "N D V D"),
    ("The college is having the department fair this Friday", "D N V V D N N D N"),
    ("Research interests span many areas", "N N V D N"),
    ("Alex is finishing his Ph.D", "N V V D N"),
    ("She is the author", "N V D N"),
    ("It is almost the end of the semester", "N V D D N D D N"),
    ("Blue is a color", "N V D N"),
    ("They wrote a book", "N V D N"),
    ("The syrup covers the pancake", "D N V D N"),
    ("Harrison has these teeth", "N V D N"),
    ("The numbers are fractions", "D N V N"),
    ("Yesterday happened", "N V"),
    ("Caramel is sweet", "N V D"),
    ("Computers use electricity", "N V N"),
    ("Gold is a valuable thing", "N V D D N"),
    ("This extension cord helps", "D D N V"),
    ("It works on my machine", "N V D D N"),
    ("We have the words", "N V D N"),
    ("Trapp is a dog", "N V D N"),
    ("This is a computer", "N V D N"),
    ("I love lamps", "N V N"),
    ("I walked outside", "N V N"),
    ("You never bike home", "N D V N"),
    ("You are a wizard Harry", "N V D N N"),
    ("Trapp ate the shoe", "N V D N"),
    ("Jett failed his test", "N V D N"),
    ("Alice won the game", "N V D N"),
    ("The class lasted a semester", "D N V D N"),
    ("The tree had a branch", "D N V D N"),
    ("I ran a race", "N V D N"),
    ("The dog barked", "D N V"),
    ("Toby hit the wall", "N V D N"),
    ("Zayn ate an apple", "N V D N"),
    ("The cat fought the dog", "D N V D N"),
    ("I got an A", "N V D N"),
    ("The A hurt", "D N V"),
    ("I jump", "N V"),
    ("I drank a yerb", "N V D N"),
    ("The snake ate a fruit", "D N V D N"),
    ("I played the game", "N V D N"),
    ("I watched a movie", "N V D N"),
    ("Clark fixed the audio", "N V D N"),
    ("I went to Frary", "N V D N"),
    ("I go to Pomona", "N V D N"),
    ("Food are friends not fish", "N V N D N"),
    ("You are reading this", "N V D N"),
    ("Wonderland protocol is amazing", "D N V D"),
    ("This is a sentence", "D V D N"),
    ("I should be doing homework", "N V V V N"),
    ("Computers are tools", "N V N"),
    ("The whale swims", "D N V"),
    ("A cup is filled", "D N V V"),
    ("This is a cat", "D V D N"),
    ("These are trees", "D V N"),
    ("The cat is the teacher", "D N V D N"),
    ("I ate food today", "N V N N"),
    ("I am a human", "N V D N"),
    ("The cat sleeps", "D N V"),
    ("Whales are mammals", "N V N"),
    ("I like turtles", "N V N"),
    ("A shark ate me", "D N V N"),
    ("There are mirrors", "D V N"),
    ("The bus spins", "D N V"),
    ("Computers are machines", "N V N"),
    ("Beckett is a dancer", "N V D N"),
    ("Networks are things", "N V N"),
    ("The lady killed a cat", "D N V D N"),
    ("Summer is tomorrow", "N V N"),
    ("A girl cries", "D N V"),
    ("I am a dog", "N V D N"),
    ("Orange is the fruit", "N V D N"),
    ("Mary had a lamb", "N V D N"),
    ("She died yesterday", "N V N"),
    ("The dog jumped", "D N V"),
    ("The man ran", "D N V"),
    ("The sun slept", "D N V"),
    ("the computer is dying", "D N V V"),
    ("Alan likes pears", "N V N"),
    ("I am the octopus", "N V D N"),
    ("This is a sentence", "D V D N"),
    ("The dog walked", "D N V"),
    ("The wind was blowing yesterday", "D N V V N"),
    ("The laptop cried", "D N V"),
    ("I like running", "N V V "),
    ("He hates cats", "N V N"),
    ("Alan wants food", "N V N"),
    ("It is a baby", "N V D N"),
    ("I had a donut", "N V D N"),
    ("Blotto is game", "N V N"),
    ("Game math win", "N N N"),
    ("Nutella is a topping", "N V D N"),
    ("Work takes time", "N V N"),
]


# %% [markdown]
# ### Preprocess raw dataset

# %%
def process_sentence(sentence):
    """Convert a string into a list of lowercased words."""
    return sentence.lower().split()


def process_parts(parts):
    """Break the parts into individual list elements."""
    return parts.split()


dataset = [(process_sentence(s), process_parts(p)) for s, p in raw_dataset]

# %% [markdown]
# ### Prepare data for use as NN input
#
# We can't pass a list of plain text words and parts-of-speech to a NN. We need to convert them to a more appropriate format.
#
# We'll start by creating a unique index for each word and part-of-speech.

# %%
# Grab all unique words
word_to_index = {}
word_counts = {}
total_words = 0

# Grab all unique parts-of-speech
part_to_index = {}
part_counts = {}
part_list = []
total_parts = 0

for words, parts in dataset:

    # Need a part-of-speech for every word
    assert len(words) == len(parts)

    # Process words
    total_words += len(words)

    for word in words:
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)
            word_counts[word] = 0
        word_counts[word] += 1

    # Process parts
    total_parts += len(parts)

    for part in parts:
        if part not in part_to_index:
            part_to_index[part] = len(part_to_index)
            part_counts[part] = 0
            part_list.append(part)
        part_counts[part] += 1

# %%
print("Total number of words:", total_words)
print("Number of unique words:", len(word_to_index))

print()
print("       Vocabulary Indices")
print("--------------------------------")
print("          Word => Index (Count)")
print("--------------------------------")

for word in sorted(word_to_index):
    print(f"{word:>14} => {word_to_index[word]:>3} ({word_counts[word]:>2})")

# %%
print("Total number of parts-of-speech:", total_parts)
print("Number of unique parts-of-speech:", len(part_to_index))

print()
print(" Part Indices")
print("--------------")

for part, index in part_to_index.items():
    print(f" {part} => {index} ({part_counts[part]:>3} / {total_parts} = {100*part_counts[part]/total_parts:.2f}%)")


# %% [markdown]
# ### Question: What is the highest accuracy you'd expect from a "dumb" classifier (hint: look at the distribution of the targets in the output above)?

# %% [markdown]
# ## Building a Parts-of-Speech Classifier

# %% [markdown]
# ### Word embeddings
#
# Once we have a unique identifier for each word, it is useful to start our NN with an [embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding) layer. This layer converts an index into a vector of values.
#
# You can think of each value as indicating something about the word. For example, maybe the first value indicates how much a word conveys happiness vs sadness. Of course, the NN can learn any attributes and it is not limited to thinks like happy/sad, masculine/feminine, etc.
#
# This is an important concept in natual language processing. It enables the network to consider two distinct words as *similar*---synonyms would share similar embedding values.
#
# **Creating an embedding layer**. An embedding layer is created by telling it the size of the vocabulary (the number of words) and an embedding dimension (how many values to use to represent a word).
#
# **Embedding layer input and output**. An embedding layer takes a word index and return a corresponding embedding as a vector.

# %% [markdown]
# #### Question: What do you expect to see printed for the indices?

# %%
def to_indices(words, mapping):
    """Convert a word (like "apple") into an index (like 4)."""
    indices = [mapping[w] for w in words]
    return torch.tensor(indices, dtype=torch.long)


words = ["trapp", "computer"]
print("An example mapping of words to indices.")
print("Words:", words)
print("Indices:", to_indices(words, word_to_index))

# %%
# The vocab size is determined by how many words you expect to train on
vocab_size = len(word_to_index)

# We get to pick the number of parameters that represent a word
embed_dim = 6

embed_layer = torch.nn.Embedding(vocab_size, embed_dim)

# %% [markdown]
# #### Question: What is the expected shape of `embed_output`?

# %%
sentence = "The dog ate the apple"
words = process_sentence(sentence)
indices = to_indices(words, word_to_index)

# Test out our untrained embedding layer
embed_output = embed_layer(indices)
print("Indices shape:", indices.shape)
print("Indices values:", indices)
print("Embedding output shape:", embed_output.shape)
print(f"Embedding values:\n{embed_output}")

# %% [markdown]
# ### Adding an LSTM (RNN) layer
#
# The [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM) layer is in charge of processing embeddings such that the network can output the correct classification. Since this is a recurrent layer, it will take into account past words when it creates an output for the current word.
#
# **Creating an LSTM layer**. To create an LSTM you need to tell it the size of its input (the size of an embedding) and the size of its internal cell state.
#
# **LSTM layer input and output**. An LSTM takes an embedding (and optionally an initial hidden and cell state) and outputs a value for each word as well as the current hidden and cell state).
#
# If you read the linked LSTM documentation you will see that it requires input in this format: `(seq_len, batch, input_size)`.
#
# As you can see above, our embedding layer outputs something that is `(seq_len, input_size)`. So, we need to add a dimension in the middle.

# %%
hidden_dim = 10  # Hyperparameter
num_layers = 5  # Hyperparameter

lstm_layer = torch.nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers)

# %% [markdown]
# #### Question: What is the expected shape of `lstm_output`?

# %%
# The LSTM layer expects the input to be in the shape (L, N, E)
#   L is the length of the sequence
#   N is the batch size (we'll stick with 1 here)
#   E is the size of the embedding

# Add the N dimension
lstm_input = embed_output.unsqueeze(1)

# We can ignore the second output of the lstm_layer for now
lstm_output, _ = lstm_layer(lstm_input)

print("LSTM output shape:", lstm_output.shape)

# %% [markdown]
# ### Classifiying the LSTM output
#
# We can now add a fully connected, [linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear) layer to our NN to classify the word's part-of-speech.
#
# **Creating a linear layer**. We create a linear layer by specifying the shape of the input into the layer and the number of neurons in the linear layer.
#
# **Linear layer input and output**. The input is expected to be `(input_size, output_size)` and the output will be the output of each neuron.

# %%
# Out network needs an output for each possible part-of-speech
parts_size = len(part_to_index)

linear_layer = torch.nn.Linear(hidden_dim, parts_size)

# %% [markdown]
# #### Question: What is the expected shape of `linear_output`?

# %%
linear_output = linear_layer(lstm_output)

print("Linear output shape:", linear_output.shape)
print(f"Linear output:\n{linear_output}")

# %% [markdown]
# ## Training an LSTM Model

# %% [markdown]
# ### Setting all hyperparameters

# %%
# Training/validation split
valid_percent = 0.15

# Size of word embedding
embed_dim = 8

# Size of LSTM internal state
hidden_dim = 8

# Number of LSTM layers
num_layers = 1

# Optimization hyperparameters
learning_rate = 0.005
num_epochs = 20

# %% [markdown]
# ### Splitting the dataset into training and validation partitions

# %%
N = len(dataset)
vocab_size = len(word_to_index)  # Number of unique input words
parts_size = len(part_to_index)  # Number of unique output targets

# Shuffle the data so that we can split the dataset randomly
shuffle(dataset)

split_point = int(N * valid_percent)
valid_dataset = dataset[:split_point]
train_dataset = dataset[split_point:]

print("Size of validation dataset:", len(train_dataset))
print("Size of validation dataset:", len(valid_dataset))


# %% [markdown]
# ### Creating the parts-of-speech LSTM model

# %%
class POS_LSTM(torch.nn.Module):
    """Parts-of-speech LSTM model."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, parts_size):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, embed_dim)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers)
        self.linear = torch.nn.Linear(hidden_dim, parts_size)

    def forward(self, X):
        X = self.embed(X)
        X, _ = self.lstm(X.unsqueeze(1))
        return self.linear(X)


# %% [markdown]
# ### Training

# %%
def train_one_epoch(mb, dataset, model, criterion, optimizer):

    model.train()

    total_loss = 0

    for words, parts in progress_bar(dataset, parent=mb):

        mb.child.comment = "Training"
        
        word_indices = to_indices(words, word_to_index)
        part_indices = to_indices(parts, part_to_index)

        part_scores = model(word_indices)

        loss = criterion(part_scores.squeeze(), part_indices)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataset)


def validate(mb, dataset, model, criterion):

    model.eval()

    total_words = 0
    total_correct = 0
    total_loss = 0

    with torch.no_grad():

        data_iter = progress_bar(dataset, parent=mb) if mb else iter(dataset)
        for words, parts in data_iter:

            if mb:
                mb.child.comment = f"Validation"

            total_words += len(words)

            word_indices = to_indices(words, word_to_index)
            part_indices = to_indices(parts, part_to_index)

            part_scores = model(word_indices).squeeze()

            loss = criterion(part_scores.squeeze(), part_indices)
            total_loss += loss.item()

            predictions = part_scores.argmax(dim=1)
            total_correct += sum(t == part_list[p] for t, p in zip(parts, predictions))

    return total_correct * 100 / total_words, total_loss / len(dataset)


def update_plots(mb, train_losses, valid_losses, epoch, num_epochs):

    # Update plot data
    max_loss = max(max(train_losses), max(valid_losses))
    min_loss = min(min(train_losses), min(valid_losses))

    x_margin = 0.2
    x_bounds = [0 - x_margin, num_epochs + x_margin]

    y_margin = 0.1 * (max_loss - min_loss)
    y_bounds = [min_loss - y_margin, max_loss + y_margin]

    train_xaxis = torch.linspace(0, epoch + 1, len(train_losses))
    valid_xaxis = torch.linspace(0, epoch + 1, len(valid_losses))
    graph_data = [[train_xaxis, train_losses], [valid_xaxis, valid_losses]]

    mb.update_graph(graph_data, x_bounds, y_bounds)


# %%
model = POS_LSTM(vocab_size, embed_dim, hidden_dim, num_layers, parts_size)

summary(model)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_losses = []
valid_losses = []
accuracies = []

mb = master_bar(range(num_epochs))
mb.names = ["Train Loss", "Valid Loss"]
mb.main_bar.comment = f"Epochs"

accuracy, valid_loss = validate(None, valid_dataset, model, criterion)
valid_losses.append(valid_loss)
accuracies.append(accuracy)

for epoch in mb:

    # Shuffle the data for each epoch (stochastic gradient descent)
    shuffle(train_dataset)

    train_loss = train_one_epoch(mb, train_dataset, model, criterion, optimizer)
    train_losses.append(train_loss)

    accuracy, valid_loss = validate(mb, valid_dataset, model, criterion)
    valid_losses.append(valid_loss)
    accuracies.append(accuracy)

    update_plots(mb, train_losses, valid_losses, epoch, num_epochs)

# %%
plt.plot(accuracies, "--o")
plt.title(f"Accuracy (Final={accuracies[-1]:.2f}%)")
plt.xlabel("Epoch")
_ = plt.ylim([0, 100])

# %% [markdown]
# ### Examining results
#
# Here we look at all words that are misclassified by the model

# %%
print("Mis-predictions on entire dataset after training")
header = "Word".center(14) + " | True Part | Prediction"
print(header)
print("-" * len(header))

model.eval()

with torch.no_grad():
    
    for words, parts in dataset:
        
        word_indices = to_indices(words, word_to_index)
        
        part_scores = model(word_indices)
        
        predictions = part_scores.squeeze().argmax(dim=1)
        
        for word, part, pred in zip(words, parts, predictions):
            
            if part != part_list[pred]:
                print(f"{word:>14} |     {part}     |    {part_list[pred]}")

# %% [markdown]
# ## Using the Model for Inference

# %%
new_sentence = "I is a teeth"

# Convert sentence to lowercase words
words = process_sentence(new_sentence)

# Check that each word is in our vocabulary
for word in words:
    assert word in word_to_index

# Convert input to a tensor
word_indices = to_indices(words, word_to_index)

# Compute prediction
predictions = model(word_indices)
predictions = predictions.squeeze().argmax(dim=1)

# Print results
for word, part in zip(new_sentence.split(), predictions):
    print(word, "=>", part_list[part.item()])

# %%
# # !jupytext --sync PartsOfSpeechRNN.ipynb
