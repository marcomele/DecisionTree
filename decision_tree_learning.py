class decisionTree(object):
	def __init__(self, examples, attributes):
		self.exa = examples
		self.att = attributes
		self.tree = {}

	@staticmethod
	def plurality_value(exa):
		import random
		labels = [labeledExa[1] for labeledExa in exa]
		freq = [(label, labels.count(label)) for label in set(labels)]
		max_f = max(freq, key = lambda pair: pair[1])[1]
		top = [lab for (lab, f) in freq if f == max_f]
		return random.sample(top, 1)

	@staticmethod
	def values(exa, att):
		return set(labeledExa[0][att] for labeledExa in exa)

	@staticmethod
	def entropy(exa):
		from math import log
		labels = [lab for (e, lab) in exa]
		count = len(labels)
		node_PS = [labels.count(l) / float(count) for l in set(labels)]
		return -sum([(x * log(x, 2) if x else x) for x in node_PS])

	@staticmethod
	def info_gain(att, exa):
		node_size = len(exa)
		split_entropy = 0.0
		for val in decisionTree.values(exa, att):
			split_exa = [e for e in exa if e[0][att] == val]
			split_entropy += (float(len(split_exa)) / node_size) * decisionTree.entropy(split_exa)
		return decisionTree.entropy(exa) - split_entropy

	def decision_tree_learning(self, examples, attributes, parent_examples = None):
		if not examples:
			return plurality_value(parent_examples)
		if len(set(lab for (exa, lab) in examples)) == 1:
			return examples[0][1]
		if not attributes:
			return plurality_value(examples)
		split_att = max(sorted(attributes), key = lambda att: decisionTree.info_gain(att, examples))
		tree = {}
		for value in decisionTree.values(examples, split_att):
			exa = [labeledExa for labeledExa in examples if labeledExa[0][split_att] == value]
			sub_tree = self.decision_tree_learning(exa, [att for att in attributes if att != split_att], examples)
			tree[(split_att, str(value))] = sub_tree
		return tree

	def learn(self):
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
		c = len(filter(lambda labeledExa: labeledExa[1] == label, exa))
		if not c:
			return 0
		c_k = len(filter(lambda labeledExa: labeledExa[0][keyValuePair[0]] == keyValuePair[1] and labeledExa[1] == label, exa))
		c_k_hat = float(c) * float(len(filter(lambda labeledExa: labeledExa[0][keyValuePair[0]] == keyValuePair[1], exa))) / float(len(exa))
		return float((c_k - c_k_hat) ** 2.0) / c_k_hat

	@staticmethod
	def __prune__(tree, examples, p_func, maxPchange):
		if not isinstance(tree, dict):
			return tree
		# if there are further levels (splits)
		# go deep further into the tree first
		for keyValuePair in tree:
			if isinstance(tree[keyValuePair], dict):
				tree[keyValuePair] = decisionTree.__prune__(tree[keyValuePair], filter(lambda labeledExa: labeledExa[0][keyValuePair[0]] == keyValuePair[1], examples), p_func, maxPchange)
		# if all this subtree's childs are leaves
		# (after potential pruning in deeper levels)
		# apply pruning on this attribute
		if not reduce(lambda x, y: x or y, [isinstance(tree[keyValuePair], dict) for keyValuePair in tree], False):
			labels = set(tree[keyValuePair] for keyValuePair in tree)
			delta = sum(sum([decisionTree.dev(tree, examples, keyValuePair, label) for label in labels]) for keyValuePair in tree)
			df = len([keyValuePair for keyValuePair in tree]) - 1
			if p_func(delta, df) > maxPchange:
				return decisionTree.plurality_value(examples)[0]
		return tree

	def prune(self, maxPchange = 0.05):
		if not self.tree:
			import sys
			sys.stderr.write("[ERROR] pruning attempt on empty tree\n")
			return
		import math
		chisq_fun = lambda x, df: 1.0 / (2.0 * gamma(df / 2.0)) * (x / 2.0) ** (df / 2.0 - 1) * exp(-x / 2.0)
		try:
			from scipy.stats import chi2
			chisq_fun = lambda x, df: 1 - chi2.cdf(x, df)
		except ImportError:
			import sys
			sys.stderr.write("[WARNING] ImportError: no module named scipy\n")
			sys.stderr.write("          falling back to math module to perform chi-squared pruning\n")
		self.tree = decisionTree.__prune__(self.tree, self.exa, chisq_fun, maxPchange)

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
		decisionTree.printKey(self.tree)

	@staticmethod
	def __classify__(test, tree, default = None):
		if not isinstance(tree, dict):
			return tree
		split_att = [keyValuePair[0] for keyValuePair in tree][0]
		if test[split_att] not in [keyValuePair[1] for keyValuePair in tree]:
			import sys
			sys.stderr.write("[ERROR] Test istance has missing values on needed attribute\n")
			sys.stderr.write("        classified with default label (" + str(default) + ")\n")
			return None
		return decisionTree.__classify__(test, tree[(split_att, test[split_att])])

	def classify_example(self, test, default = None):
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
	result = tree.classify_example(test)
	print "[" + result[1] + "]"

	print "\n\n*** PRUNED TREE ***\n"
	tree.prune(maxPchange = 0.05)
	tree.show()

	print "\n\n*** CLASSIFICATION TEST ***"
	test = {"Alt" : "No", "Bar" : "No", "Fri" : "No", "Hun" : "No", "Pat" : "Full", "Price" : "$$$", "Rain" : "No", "Res" : "No", "Type" : "Burger", "Est" : "10-30"}
	result = tree.classify_example(test)
	print "[" + result[1] + "]"
