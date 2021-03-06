class decisionTree(object):
	"""
		DecisionTree class, instantiates a decision tree object. Requires:

		examples, an array of tuples (exa, lab) where lab is a label which can NOT be a python dictionary, and exa is a python dictionary of the form {key : value} where key is an attribute in the set of attributes and values is its value for the example. Values should not be a python dictionary.
		attributes, an array of all the attributes for the dataset
	"""

	def __init__(self, examples, attributes):
		self.exa = examples
		self.att = attributes
		self.tree = {}

	@staticmethod
	def plurality_value(exa):
		"""
			Returns the majority class in the given example set, breaking ties randomly
		"""
		import random
		labels = [labeledExa[1] for labeledExa in exa]
		freq = [(label, labels.count(label)) for label in set(labels)]
		max_f = max(freq, key = lambda pair: pair[1])[1]
		top = [lab for (lab, f) in freq if f == max_f]
		return random.sample(top, 1)[0]

	def values(self, att):
		"""
			Returns the set of unique values assumed by the attribute in the example set
		"""
		return set(labeledExa[0][att] for labeledExa in self.exa)

	@staticmethod
	def entropy(exa):
		"""
			Returns the entropy measure for the provided example set
		"""
		from math import log
		labels = [lab for (e, lab) in exa]
		count = len(labels)
		node_PS = [labels.count(l) / float(count) for l in set(labels)]
		return -sum([(x * log(x, 2) if x else x) for x in node_PS])

	def info_gain(self, exa, att):
		"""
			Returns the information gain for the possible split of the example set according to the given attribute
		"""
		node_size = len(exa)
		split_entropy = 0.0
		for val in self.values(att):
			split_exa = [e for e in exa if e[0][att] == val]
			split_entropy += (float(len(split_exa)) / node_size) * decisionTree.entropy(split_exa)
		return decisionTree.entropy(exa) - split_entropy

	def decision_tree_learning(self, examples, attributes, parent_examples = None):
		"""
			Internal recursive method implementing the decision tree learning algorithm. Requires:

			examples, the labeled example set, as specified by the class decisionTree
			attributes, the features of the example set
			parent_examples, (optional) for deeper recursion levels, the example set before the current split

			Returns the majority class of the example set before the split, if there are no more examples to learn; the of the current split if all examples have the same label; the majority class of the current example set if there are no more attributes to split according to; a subtree generated by the recursion to a lower level after splitting on the best available attribute, selected as the one providing the maximum information gain, breaking ties according to lexicographic ordering.
		"""
		if not examples:
			return decisionTree.plurality_value(parent_examples)
		if len(set(lab for (exa, lab) in examples)) == 1:
			return examples[0][1]
		if not attributes:
			return plurality_value(examples)
		# select split attribute
		split_att = max(sorted(attributes), key = lambda att: self.info_gain(examples, att))
		tree = {}
		# generate splits on splitt_att
		for value in self.values(split_att):
			exa = [labeledExa for labeledExa in examples if labeledExa[0][split_att] == value]
			sub_tree = self.decision_tree_learning(exa, [att for att in attributes if att != split_att], examples)
			tree[(split_att, str(value))] = sub_tree
		return tree

	def learn(self):
		"""
			The wrap method to start the decision tree learning algorithm. Requires the target class to be initiated with coherent example set and attribute list. Returns a tree as a python dictionary of the form {(key, val) : node} where (key, val) is a tuple present in the same level once with each value val that the attribute key assumes in the training set, and node is either an inner python dictionary as a split subtree or a label for the terminal node. Label shall never be instances of python dictionaries.
		"""
		if not self.exa or not self.att or not isinstance(self.exa[0], tuple):
			import sys
			sys.stderr.write("[ERROR] bad data provided as input for decisionTree.learn()\n")
			return
		# filter data with missing values
		nomissisng = filter(lambda labeledExa: None not in [labeledExa[0][att] for att in self.att], self.exa)
		# filter data with missing label
		self.tree = self.decision_tree_learning(filter(lambda labeledExa: labeledExa[1] is not None and not isinstance(labeledExa[1], dict), nomissisng), self.att)

	@staticmethod
	def dev(tree, exa, keyValuePair, label):
		"""
			Computes the single deviation value of the split of the example set according to the (attribute, value) pair keyValuePair, and the label label
		"""
		c = len(filter(lambda labeledExa: labeledExa[1] == label, exa))
		if not c:
			return 0
		c_k = len(filter(lambda labeledExa: labeledExa[0][keyValuePair[0]] == keyValuePair[1] and labeledExa[1] == label, exa))
		if not c_k:
			return 0
		c_k_hat = float(c) * float(len(filter(lambda labeledExa: labeledExa[0][keyValuePair[0]] == keyValuePair[1], exa))) / float(len(exa))
		return float((c_k - c_k_hat) ** 2.0) / c_k_hat

	@staticmethod
	def __prune__(tree, examples, p_func, maxPchance):
		"""
			Inner method that performs pruning traversing recursively the tree and going back upwards pruning from the lowermost nodes whose all child are leaves, if applicable given the p_value provided by the chi-squared probability distribution calculated in the total deviation delta and the degree of freedom df of the splitting attribute. Requires:

			tree, the current subtree
			examples, the example set at the current level
			p_func, a function taking delta and df and providing the cumulative distribution function of the chi-squared evaluated in delta with parameter df
			maxPchance, the tuning parameter defining the threshold at which to reject the null hypothesis, thus leaving the split
		"""
		if not isinstance(tree, dict):
			return tree
		# if there are further levels (splits)
		# go deep further into the tree first
		for keyValuePair in tree:
			if isinstance(tree[keyValuePair], dict):
				tree[keyValuePair] = decisionTree.__prune__(tree[keyValuePair], filter(lambda labeledExa: labeledExa[0][keyValuePair[0]] == keyValuePair[1], examples), p_func, maxPchance)
		# if all this subtree's childs are leaves
		# (after potential pruning in deeper levels)
		# apply pruning on this attribute
		if not reduce(lambda x, y: x or y, [isinstance(tree[keyValuePair], dict) for keyValuePair in tree], False):
			labels = set(tree[keyValuePair] for keyValuePair in tree)
			delta = sum(sum([decisionTree.dev(tree, examples, keyValuePair, label) for label in labels]) for keyValuePair in tree)
			df = (len([keyValuePair for keyValuePair in tree]) - 1) * (len(labels) - 1)
			if p_func(delta, df) > maxPchance:
				return decisionTree.plurality_value(examples)
		return tree

	def prune(self, maxPchance = 0.05):
		"""
			The wrap method to initiate the pruning algorithm on a fully-grown tree. Pruning is chi-squared pruning, and maxPchance is the threshold at which to reject the null hypothesis and leaving a split as is. Makes use of the python scipy.stat module providing the chi2.cdf() function.
		"""
		if not self.tree:
			import sys
			sys.stderr.write("[ERROR] pruning attempt on empty tree\n")
			return
		import math
		try:
			from scipy.stats import chi2
		except ImportError:
			import sys
			sys.stderr.write("[ERROR] ImportError: no module named scipy\n")
			sys.stderr.write("        aborting pruning\n")
			return
		chisq_fun = lambda x, df: 1 - chi2.cdf(x, df)
		self.tree = decisionTree.__prune__(self.tree, self.exa, chisq_fun, maxPchance)

	@staticmethod
	def printKey(tree, depth = 0):
		if not isinstance(tree, dict):
			print "--> " + str(tree)
			return
		for ruleHead in tree:
			print ("\t" * depth) + ("and" if depth else "if") + str(ruleHead)
			if not isinstance(tree[ruleHead], dict):
				print ("\t" * (depth + 1)) + "--> then [" + str(tree[ruleHead]) + "]"
			else:
				decisionTree.printKey(tree[ruleHead], depth + 1)

	def show(self):
		"""
			Prints a human-readable representation of the decision tree.
		"""
		decisionTree.printKey(self.tree)

	@staticmethod
	def __classify__(test, tree, default = None):
		"""
			Classifies a test istance traversing the decision tree across the current subtree. Returns the predicted label, if tree is a leaf; the default label, if the subtree can't be traversed due to a missing value; or the classification returned by the deeper recursion levels.
		"""
		if not isinstance(tree, dict):
			return tree
		split_att = [keyValuePair[0] for keyValuePair in tree][0]
		if test[split_att] not in [keyValuePair[1] for keyValuePair in tree]:
			import sys
			sys.stderr.write("[ERROR] Test istance has missing values on needed attribute\n")
			sys.stderr.write("        classified with default label (" + str(default) + ")\n")
			return default
		return decisionTree.__classify__(test, tree[(split_att, test[split_att])])

	def classify(self, test, default = None):
		"""
			Returns a classification for the given test, possibly the default class if the test example misses the value for a determining attribute. Result is a touple containing the original test and the applied label.
		"""
		return (test, decisionTree.__classify__(test, self.tree, default))


if __name__ == '__main__':
	# examples for tree learning must be an array or list of tuples (exa, lab)
	# where exa is a dictionary containing "key" : "value" entries for each attribute
	# and lab is the label, or class, or goal
	examples = [
		({"Alt" : "Yes", "Bar" : "No", "Fri" : "No", "Hun" : "Yes", "Pat" : "Some", "Price" : "$$$", "Rain" : "No", "Res" : "Yes", "Type" : "French", "Est" : "0-10"}, "Yes"),
		({"Alt" : "Yes", "Bar" : "No", "Fri" : "No", "Hun" : "Yes", "Pat" : "Full", "Price" : "$", "Rain" : "No", "Res" : "No", "Type" : "Thai", "Est" : "30-60"}, "No"),
		({"Alt" : "No", "Bar" : "Yes", "Fri" : "No", "Hun" : "No", "Pat" : "Some", "Price" : "$", "Rain" : "No", "Res" : "No", "Type" : "Burger", "Est" : "0-10"}, "Yes"),
		({"Alt" : "Yes", "Bar" : "No", "Fri" : "Yes", "Hun" : "Yes", "Pat" : "Full", "Price" : "$", "Rain" : "Yes", "Res" : "No", "Type" : "Thai", "Est" : "10-30"}, "Yes"),
		({"Alt" : "Yes", "Bar" : "No", "Fri" : "Yes", "Hun" : "No", "Pat" : "Full", "Price" : "$$$", "Rain" : "No", "Res" : "Yes", "Type" : "French", "Est" : ">60"} , "No"),
		({"Alt" : "No", "Bar" : "Yes", "Fri" : "No", "Hun" : "Yes", "Pat" : "Some", "Price" : "$$", "Rain" : "Yes", "Res" : "Yes", "Type" : "Italian", "Est" : "0-10"}, "Yes"),
		({"Alt" : "No", "Bar" : "Yes", "Fri" : "No", "Hun" : "No", "Pat" : "None", "Price" : "$", "Rain" : "Yes", "Res" : "No", "Type" : "Burger", "Est" : "0-10"}, "No"),
		({"Alt" : "No", "Bar" : "No", "Fri" : "No", "Hun" : "Yes", "Pat" : "Some", "Price" : "$$", "Rain" : "Yes", "Res" : "Yes", "Type" : "Thai", "Est" : "0-10"}, "Yes"),
		({"Alt" : "No", "Bar" : "Yes", "Fri" : "Yes", "Hun" : "No", "Pat" : "Full", "Price" : "$", "Rain" : "Yes", "Res" : "No", "Type" : "Burger", "Est" : ">60"}, "No"),
		({"Alt" : "Yes", "Bar" : "Yes", "Fri" : "Yes", "Hun" : "Yes", "Pat" : "Full", "Price" : "$$$", "Rain" : "No", "Res" : "Yes", "Type" : "Italian", "Est" : "10-30"}, "No"),
		({"Alt" : "No", "Bar" : "No", "Fri" : "No", "Hun" : "No", "Pat" : "None", "Price" : "$", "Rain" : "No", "Res" : "No", "Type" : "Thai", "Est" : "0-10"}, "No"),
		({"Alt" : "Yes", "Bar" : "Yes", "Fri" : "Yes", "Hun" : "Yes", "Pat" : "Full", "Price" : "$", "Rain" : "No", "Res" : "No", "Type" : "Burger", "Est" : "30-60"}, "Yes")
	]
	# attributes must be an array or list or set of the same attributes for "key"s in examples
	attributes = [key for key in examples[0][0]]

	# define object decisionTree
	tree = decisionTree(examples, attributes)
	tree.learn()
	print "*** TRAINED MODEL ***\n"
	tree.show()

	print "\n\n*** CLASSIFICATION TEST ***"
	test = {"Alt" : "No", "Bar" : "No", "Fri" : "No", "Hun" : "No", "Pat" : "Full", "Price" : "$$$", "Rain" : "No", "Res" : "No", "Type" : "Burger", "Est" : "10-30"}
	result = tree.classify(test)
	print "[" + result[1] + "]"

	print "\n\n*** PRUNED TREE ***\n"
	tree.prune(maxPchance = 0.05)
	tree.show()

	print "\n\n*** CLASSIFICATION TEST ***"
	result = tree.classify(test)
	print "[" + result[1] + "]"
